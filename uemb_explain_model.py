import os

import keras
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead
import numpy as np


def build_gru_model(params=None):
    """
    This model is based on pre-trained Word Embeddings (not BERT)

    Parameters
    ----------
    params

    Returns
    -------

    """
    if not params:  # default params
        params = {
            'batch_size': 5,
            'vocab_size': 20001,
            'user_size': 8000,
            'emb_dim': 300,
            'dp_rate': .2,
            'word_emb_path': './resources/word_emb.npy',
            'user_emb_path': './resources/user_emb.npy',
            'word_emb_train': False,
            'user_emb_train': True,
            'user_task_weight': 1,
            'word_task_weight': 1,
            'epochs': 10,
            'optimizer': 'adam',
            'lr': 0.00001,
            'max_len': 1000,
        }

    user1_input = keras.layers.Input((1,), name='user_doc_input', dtype='int32')
    doc_input = keras.layers.Input((params['max_len'],), name='doc_input', dtype='int32')

    user2_input = keras.layers.Input((1,), name='user_concept_input', dtype='int32')
    concept_input = keras.layers.Input((1,), name='concept_input', dtype='int32')

    # load weights if word embedding path is given
    if os.path.exists(params['word_emb_path']):
        weights = np.load(params['word_emb_path'])
        word_emb = keras.layers.Embedding(
            weights.shape[0], weights.shape[1],
            weights=[weights],
            trainable=params['word_emb_train'], name='word_emb'
        )
    else:
        word_emb = keras.layers.Embedding(
            params['vocab_size']+1, params['emb_dim'],
            trainable=params['word_emb_train'], name='word_emb'
        )

    # load weights if concept embedding path is given
    if os.path.exists(params['concept_emb_path']):
        weights = np.load(params['concept_emb_path'])
        concept_emb = keras.layers.Embedding(
            weights.shape[0], weights.shape[1],
            weights=[weights],
            trainable=True, name='concept_emb'
        )
    else:
        concept_emb = keras.layers.Embedding(
            params['vocab_size'], params['emb_dim'],
            trainable=True, name='concept_emb'
        )

    # load weights if user embedding path is given
    if 'pretrained_uemb' in params and os.path.exists(params['pretrained_uemb']):
        weights = np.load(params['pretrained_uemb'])
        user_emb = keras.layers.Embedding(
            weights.shape[0], weights.shape[1],
            weights=[weights],
            trainable=params['user_emb_train'], name='user_emb'
        )
    else:
        user_emb = keras.layers.Embedding(
            params['user_size'], params['emb_dim'],
            trainable=params['user_emb_train'], name='user_emb'
        )

    '''
    User & Documents
    '''
    user1_rep = user_emb(user1_input)
    doc_rep = word_emb(doc_input)

    # GRU reads through words to generate document representation
    gru_encoder = keras.layers.GRU(params['emb_dim'], )(doc_rep)
    gru_encoder = keras.layers.Dropout(params['dp_rate'])(gru_encoder)

    # dot product between document and user vectors
    user_doc_dot = keras.layers.dot(
        [user1_rep, gru_encoder], axes=-1
    )
    user_doc_dot = keras.layers.Reshape((1,))(user_doc_dot)
    user_doc_pred = keras.layers.Dense(
        1, activation='sigmoid', name='user_doc_pred'
    )(user_doc_dot)

    '''
    User & Concepts
    '''
    user2_rep = user_emb(user2_input)
    concept_rep = concept_emb(concept_input)
    user_concept_dot = keras.layers.dot(
        [user2_rep, concept_rep], axes=-1
    )
    user_concept_dot = keras.layers.Reshape((1,))(user_concept_dot)
    user_concept_pred = keras.layers.Dense(
        1, activation='sigmoid', name='user_concept_pred'
    )(user_concept_dot)

    '''Compose Multitask model'''
    # ud_model = keras.models.Model(
    #     inputs=[user1_input, doc_input],
    #     outputs=user_doc_pred
    # )
    # ud_model.compile(
    #     loss='binary_crossentropy',
    #     optimizer=keras.optimizers.RMSprop(lr=params['lr'])
    # )
    # uc_model = keras.models.Model(
    #     inputs=[user2_input, concept_input],
    #     outputs=user_concept_pred
    # )
    # uc_model.compile(
    #     loss='binary_crossentropy',
    #     optimizer=keras.optimizers.Adam(lr=params['lr'])
    # )
    loss_w_dict = {
        'user_doc_pred': params['doc_task_weight'],
        'user_concept_pred': params['concept_task_weight']*0.1,
    }
    caue_model = keras.models.Model(
        inputs=[user1_input, doc_input, user2_input, concept_input],
        outputs=[user_doc_pred, user_concept_pred]
    )
    caue_model.compile(
        loss={'user_doc_pred': 'binary_crossentropy', 'user_concept_pred': 'binary_crossentropy'},
        optimizer=keras.optimizers.RMSprop(lr=params['lr']),
        loss_weights=loss_w_dict
    )

    return caue_model


# Dual Neural Network
class CAUEgru(nn.Module):
    def __init__(self, params):
        super(CAUEgru, self).__init__()
        self.params = params

        # define user embeddings
        if 'pretrained_uemb' in self.params and os.path.exists(self.params['pretrained_uemb']):
            self.uemb = nn.Embedding.from_pretrained(self.params['pretrained_uemb'])
        else:
            self.uemb = nn.Embedding(
                self.params['user_size'], self.params['emb_dim']
            )
            # self.uemb.reset_parameters()
            torch.nn.init.kaiming_uniform_(self.uemb.weight, a=np.sqrt(5))

        # word embeddings
        if 'word_emb_path' in self.params and os.path.exists(self.params['word_emb_path']):
            self.wemb = nn.Embedding.from_pretrained(
                torch.FloatTensor(np.load(self.params['word_emb_path']))
            )
            self.wemb.requires_grad_(requires_grad=False)  # freeze the weight update
        else:
            self.wemb = nn.Embedding(
                self.params['vocab_size']+1, self.params['emb_dim']
            )
            torch.nn.init.kaiming_uniform_(self.uemb.weight, a=np.sqrt(5))

        # concept embeddings
        if params['use_concept']:
            if os.path.exists(self.params['concept_emb_path']):
                self.cemb = nn.Embedding.from_pretrained(
                    torch.FloatTensor(np.load(self.params['concept_emb_path']))
                )
                self.cemb.requires_grad_(requires_grad=False)  # freeze the weight update
            else:
                self.cemb = nn.Embedding(
                    self.params['concept_size'], self.params['emb_dim']
                )
                torch.nn.init.kaiming_uniform_(self.uemb.weight, a=np.sqrt(5))
            self.concept_projector = nn.Linear(self.cemb.embedding_dim, self.uemb.embedding_dim)

        # to encode words in documents
        self.doc_encoder = nn.GRU(
            input_size=self.wemb.embedding_dim,
            hidden_size=self.params['emb_dim'] // 2,
            bidirectional=self.params['bidirectional'],
            batch_first=True,
            dropout=self.params['dp_rate']
        )
        # self.cos = nn.CosineSimilarity(dim=-1)
        # self.att_nn = None

    def forward(self, **kwargs):
        input_doc_ids = kwargs['input_doc_ids']
        input1_uids = kwargs['input_uids4doc']
        users1 = self.uemb(input1_uids)
        doc_embs = self.wemb(input_doc_ids)
        _, gru_embs = self.doc_encoder(doc_embs)
        gru_embs = torch.cat((gru_embs[0, :, :], gru_embs[1, :, :]), -1)
        # dot product between user and docs
        user_doc_sim = torch.sum(users1 * gru_embs, -1)
        # user_doc_sim = self.cos(users1, gru_embs)

        if self.params['use_concept']:
            input2_uids = kwargs['input_uids4concept']
            input_concept_ids = kwargs['input_concept_ids']
            users2 = self.uemb(input2_uids)
            concept_embs = self.cemb(input_concept_ids)
            concept_embs = torch.relu(self.concept_projector(concept_embs))
            # dot product between user and concepts
            user_concept_sim = torch.sum(users2 * concept_embs, -1)
            # user_concept_sim = self.cos(users2, concept_embs)
        else:
            user_concept_sim = None

        return user_doc_sim, user_concept_sim


class CAUEBert(nn.Module):
    def __init__(self, params):
        super(CAUEBert, self).__init__()
        self.params = params

        # define user embeddings
        if 'pretrained_uemb' in self.params and os.path.exists(self.params['pretrained_uemb']):
            self.uemb = nn.Embedding.from_pretrained(self.params['pretrained_uemb'])
        else:
            self.uemb = nn.Embedding(
                self.params['user_size'], self.params['emb_dim']
            )
            self.uemb.reset_parameters()
            torch.nn.init.kaiming_uniform_(self.uemb.weight, a=np.sqrt(5))

        # bert
        self.bert_model = AutoModel.from_pretrained(self.params['bert_name'])

        self.dropout = nn.Dropout(self.params['dp_rate'])
        # a project layer to map 768 dimensional vectors to 300 vectors
        self.linear = nn.Linear(
            self.bert_model.config.hidden_size, self.params['emb_dim'])
        self.cos = nn.CosineSimilarity(dim=-1)
        if 'use_mlm' in self.params and self.params['use_mlm']:
            self.cls = BertLMPredictionHead(self.bert_model.config)

    def forward(self, **kwargs):
        input_doc_ids = kwargs['input_doc_ids']
        input_uids4doc = kwargs['input_uids4doc']
        input_uids4concept = kwargs['input_uids4concept']
        input_concept_ids = kwargs['input_concept_ids']

        users4doc = self.uemb(input_uids4doc)
        users4concept = self.uemb(input_uids4concept)

        doc_bert_embs = self.bert_model(input_ids=input_doc_ids)
        doc_embs = self.dropout(self.linear(doc_bert_embs[1]))

        concept_bert_embs = self.bert_model(input_ids=input_concept_ids)
        concept_embs = self.dropout(torch.sigmoid(self.linear(concept_bert_embs[1])))

        user_doc_sims = torch.sum(users4doc * doc_embs, -1)
        user_concept_sims = torch.sum(users4concept * concept_embs, -1)

        if 'use_mlm' in self.params and self.params['use_mlm']:
            mlm_scores = self.cls(doc_bert_embs[0]).view(-1, self.bert_model.config.vocab_size)
            return user_doc_sims, user_concept_sims, mlm_scores
        else:
            return user_doc_sims, user_concept_sims
