import os
import json

import keras

import torch
import torch.nn as nn
from transformers import AutoModel

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
            params['vocab_size'], weights.shape[1],
            weights=[weights],
            trainable=params['word_emb_train'], name='word_emb'
        )
    else:
        word_emb = keras.layers.Embedding(
            params['vocab_size'], params['emb_dim'],
            trainable=params['word_emb_train'], name='word_emb'
        )

    # load weights if user embedding path is given
    if 'pretrained_uemb' in params and os.path.exists(params['pretrained_uemb']):
        user_emb = keras.layers.Embedding(
            params['user_size'], params['emb_dim'],
            weights=[np.load(params['pretrained_uemb'])],
            trainable=params['user_emb_train'], name='user_emb'
        )
    else:
        user_emb = keras.layers.Embedding(
            params['user_size'], params['emb_dim'],
            trainable=params['user_emb_train'], name='user_emb'
        )


class DannGru(nn.Module):
    def __init__(self, params):
        super(DannGru, self).__init__()
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

        # word embeddings
        if 'pretrained_wemb' in self.params and os.path.exists(self.params['pretrained_wemb']):
            self.wemb = nn.Embedding.from_pretrained(self.params['pretrained_uemb'])
        else:
            self.wemb = nn.Embedding(
                self.params['vocab_size'], self.params['emb_dim']
            )
            torch.nn.init.kaiming_uniform_(self.uemb.weight, a=np.sqrt(5))

        # to encode words in documents
        self.doc_encoder = nn.GRU(
            input_size=self.wemb.embedding_dim,
            hidden_size=self.params['emb_dim'] // 2,
            bidirectional=self.params['bidirectional'],
            batch_first=True,
            dropout=self.params['dp_rate']
        )

        # define self attention networks
        self.att_nn = None
        pass

    def forward(self):
        pass


class DannBert(nn.Module):
    def __init__(self, params):
        super(DannBert, self).__init__()
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

        # to encode words in documents
        self.doc_encoder = nn.GRU(
            input_size=self.wemb.embedding_dim,
            hidden_size=self.params['emb_dim'] // 2,
            bidirectional=self.params['bidirectional'],
            batch_first=True,
            dropout=self.params['dp_rate']
        )

        # define self attention networks
        self.att_nn = None
        pass

    def forward(self):
        pass
