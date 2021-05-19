"""
Implementing
    Deep Patient:
        An Unsupervised Representation to Predict the Future of Patients from the Electronic Health Records
https://www.nature.com/articles/srep26094

input features: 300 topics from lda
encoder: two layers of autoencoder

"""
import pickle
import sys
import os

import torch
from torch import nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
import numpy as np
from tqdm import tqdm
from baseline_utils import data_loader


class ConceptCorpus(object):
    def __init__(self, concept_list, doc2id=False, dictionary=None):
        """ Load Json file
        """
        self.clist = concept_list
        self.dictionary = dictionary
        self.doc2id = doc2id

    def __iter__(self):
        for line in self.clist:
            if self.doc2id and self.dictionary:  # this is for inference
                yield self.dictionary.doc2bow(line)
            else:
                yield line


class RawCorpus(object):
    def __init__(self, docs, doc2id=False, dictionary=None):
        """ Load Json file
        """
        self.docs = docs
        self.dictionary = dictionary
        self.doc2id = doc2id

    def __iter__(self):
        for line in self.docs:
            if self.doc2id and self.dictionary:  # this is for inference
                yield self.dictionary.doc2bow(line.split())
            else:
                yield line.split()


def train_concept_lda(concept_list, output_dir='../resources/embedding/', dim=300):
    """
        The number of topics should be aligned with the dimensions of the user embedding.
    """
    # load data and build dictionary
    corpus = ConceptCorpus(concept_list)
    dictionary = Dictionary(corpus, prune_at=15000)
    dictionary.save(output_dir + 'concept_lda_dict.pkl')

    concept_matrix = ConceptCorpus(concept_list, True, dictionary)
    model = LdaMulticore(
        concept_matrix, id2word=dictionary, num_topics=dim,
        passes=10, alpha='symmetric', workers=os.cpu_count()//2
    )
    model.save(output_dir + 'concept_lda.model')


def train_lda(docs, output_dir, dim=300):
    """
        The number of topics should be aligned with the dimensions of the user embedding.
    """

    if os.path.exists(output_dir + 'lda_dict.pkl'):
        dictionary = pickle.load(open(output_dir + 'lda_dict.pkl', 'rb'))
    else:
        corpus = RawCorpus(docs)
        dictionary = Dictionary(corpus, prune_at=10000)
        dictionary.save(output_dir + 'lda_dict.pkl')

    doc_matrix = RawCorpus(docs, True, dictionary)

    model = LdaMulticore(
        doc_matrix, id2word=dictionary, num_topics=dim,
        passes=10, alpha='symmetric', workers=os.cpu_count()//2
    )
    model.save(output_dir + 'lda.model')


class AE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AE, self).__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_size, out_features=hidden_size
        )
        self.encoder_output_layer = nn.Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden_size, out_features=input_size
        )
        self.dp = nn.Dropout(0.05)

    def forward(self, topics):
        features = self.dp(topics)
        features = self.encoder_hidden_layer(features)
        features = torch.sigmoid(features)
        code_features = self.encoder_output_layer(features)
        code_features = torch.sigmoid(code_features)
        decode_features = self.decoder_hidden_layer(code_features)
        decode_features = torch.sigmoid(decode_features)
        decode_features = self.decoder_output_layer(decode_features)
        reconstructed = torch.sigmoid(decode_features)
        return reconstructed, code_features


class Lda2User(object):
    """Apply LDA model on the documents to generate user and product representation.
        Outputs will be one user/product_id + vect per line.

        Parameters
        ----------
        task_name: str
            Task name, such as amazon, yelp and imdb
        dictionary_path: str
            Path of LDA dictionary file
        lda_path: str
            Path of LDA model file
        ae_path: str
            Path of Autoencoder model
    """
    def __init__(self, **kwargs):
        self.task = kwargs['task_name']
        self.word_dict = pickle.load(open(kwargs['word_dict_path'], 'rb'))
        self.concept_dict = pickle.load(open(kwargs['concept_dict_path'], 'rb'))
        self.word_model = LdaModel.load(kwargs['word_model_path'])
        self.concept_model = LdaModel.load(kwargs['concept_model_path'])
        self.ae_path = kwargs['ae_path']
        self.device = kwargs['device']

        self.ae = AE(self.word_model.num_topics, 300)  # default value in paper
        if os.path.exists(self.ae_path):
            self.ae.load_state_dict(torch.load(self.ae_path), strict=False)

    def train_autoencoder(self, user_features):
        user_features = TensorDataset(torch.FloatTensor(user_features))
        user_features = DataLoader(user_features, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.ae.parameters(), lr=.001)
        criterion = torch.nn.BCELoss().to(self.device)
        self.ae.train()

        for _ in tqdm(range(10)):
            for idx, batch in enumerate(user_features):
                optimizer.zero_grad()
                batch = batch[0]
                batch.to(self.device)
                output, _ = self.ae(batch)  # omit encoded features
                loss = criterion(output, batch)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 0.1)
                optimizer.step()

        torch.save(self.ae.state_dict(), self.ae_path)

    def lda2item(self, data_path, opath):
        """Extract user vectors from the given data path

            Parameters
            ----------
            data_path: str
                Path of data file, tsv file
            opath: str
                Path of output path for user vectors
        """
        ofile = open(opath, 'w')
        # load the datasets from caue_gru task
        user_docs, all_docs = data_loader(data_path)
        doc_features = list()
        concept_features = list()

        print('Loading Data')
        for user_id in tqdm(list(user_docs.keys())):
            docs = [all_docs[doc_id] for doc_id in user_docs[user_id]['docs']]
            doc_vectors = []
            for idx, doc in enumerate(docs):
                output = self.word_model[self.word_dict.doc2bow(doc.split())]
                doc_vectors.append([0.] * self.word_model.num_topics)
                for item in output:
                    doc_vectors[-1][item[0]] = item[1]
            # average the lda inferred documents
            doc_vectors = np.mean(doc_vectors, axis=0)
            doc_features.append(doc_vectors)

            concept_vectors = []
            for idx, concepts in enumerate(user_docs[user_id]['concepts']):
                if len(concepts) == 0:
                    continue
                output = self.concept_model[self.concept_dict.doc2bow(concepts)]
                concept_vectors.append([0.] * self.concept_model.num_topics)
                for item in output:
                    concept_vectors[-1][item[0]] = item[1]
            # average the lda inferred documents
            concept_vectors = np.mean(concept_vectors, axis=0)
            concept_features.append(concept_vectors)

        # train autoencoder if the model does not exist
        if not os.path.exists(self.ae_path):
            self.train_autoencoder(doc_features)
            self.ae.load_state_dict(torch.load(self.ae_path), strict=False)

        # convert the doc features by the autoencoder.
        _, doc_features = self.ae(doc_features)
        doc_features = doc_features.cpu().detach().numpy()

        for idx, user_id in enumerate(list(user_docs.keys())):
            # write to file
            ofile.write(user_id + '\t' + ' '.join(map(
                str, np.concatenate((doc_features[idx], concept_features[idx]), axis=None))) + '\n')
        ofile.flush()
        ofile.close()


if __name__ == '__main__':
    task = sys.argv[1]
    task_data_path = '../resources/embedding/{}/caue_gru/user_docs_concepts.pkl'.format(task)

    baseline_dir = '../resources/embedding/'
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)

    task_dir = baseline_dir + task + '/'
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    odir = task_dir + 'deeppatient2user_concept/'
    if not os.path.exists(odir):
        os.mkdir(odir)
    opath_user = odir + 'user.txt'

    word_dict_path = odir + 'lda_dict.pkl'
    word_model_path = odir + 'lda.model'

    concept_model_path = odir + 'concept_lda.model'
    concept_dict_path = odir + 'concept_lda_dict.pkl'
    autoencoder_path = task_dir + 'ae_model.pth'

    if os.path.exists(concept_model_path) and os.path.exists(concept_dict_path):
        pass
    else:
        print('Training Concept LDA Models...')
        user_data, lda_docs = data_loader(task_data_path)
        clist = []
        for uid in user_data:
            clist.extend([concepts for concepts in user_data[uid]['concepts'] if len(concepts) > 0])
        train_concept_lda(concept_list=clist, output_dir=odir, dim=300)
        train_lda(docs=lda_docs, output_dir=odir, dim=300)

    # auto encoder 2 user
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    l2u = Lda2User(
        task_name=task, word_dict_path=word_dict_path, word_model_path=word_model_path,
        concept_dict_path=concept_dict_path, concept_model_path=concept_model_path,
        data_path=task_data_path, ae_path=autoencoder_path, device=device
    )
    l2u.lda2item(task_data_path, opath_user)
