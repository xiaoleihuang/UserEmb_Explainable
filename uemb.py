import os
import json

import torch
import torch.nn as nn
from transformers import AutoModel

import numpy as np


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
