"""
Implementation of Patient representation learning and interpretable evaluation using clinical notes

We set the dimension of doc2vec as 150, and the another 150 dimensions will be the AutoEncoder
"""
import pickle
import sys
import json
import os

import torch
from torch import nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from tqdm import tqdm

from baseline_utils import train_doc2v


def dummy_func(doc):
    if type(doc) == str:
        return word_tokenize(doc)
    return doc


def get_tfidf_vect(task_name, concept_directory, save_dir):
    opath = os.path.join(save_dir, 'tfidf_vect_{}.pkl'.format(task_name))
    if os.path.exists(opath):
        return opath
    flist = os.listdir(concept_directory)
    user_concepts = dict()

    for fname in flist:
        uid = fname.split('_')[0]
        if uid not in user_concepts:
            user_concepts[uid] = []
        concepts = pickle.load(open(concept_directory + fname, 'rb'))
        user_concepts[uid].extend([concept['preferred_name'].lower() for concept in concepts])

    tfidf_vect = TfidfVectorizer(tokenizer=dummy_func, preprocessor=dummy_func, max_features=10000)
    tfidf_vect.fit(list(user_concepts.values()))
    with open(opath, 'wb') as tfidf_file:
        pickle.dump(tfidf_vect, tfidf_file)
    return opath


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


class Doc2User(object):
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs
        """
        self.task = kwargs['task_name']
        self.doc2vec = Doc2Vec.load(kwargs['doc2vec_path'])
        self.data_path = kwargs['data_path']
        self.ae_path = kwargs['ae_path']
        self.device = kwargs['device']
        self.tf_idf_vect = pickle.load(open(kwargs['tf_idf_path'], 'rb'))
        self.concept_dir = kwargs['concept_dir']
        self.mode = kwargs['mode']

        self.ae = AE(len(self.tf_idf_vect.vocabulary_), 150)
        if not os.path.exists(self.ae_path):
            self.train_autoencoder()
        else:
            self.ae.load_state_dict(torch.load(self.ae_path), strict=False)

    def train_autoencoder(self):
        user_concepts = dict()
        # load concepts
        flist = os.listdir(self.concept_dir)
        for fname in flist:
            uid = fname.split('_')[0]
            if uid not in user_concepts:
                user_concepts[uid] = []
            concepts = pickle.load(open(self.concept_dir + fname, 'rb'))
            user_concepts[uid].extend([concept['preferred_name'].lower() for concept in concepts])

        user_features = self.tf_idf_vect.transform(
            list(user_concepts.values())
        ).toarray()
        user_features = TensorDataset(torch.FloatTensor(user_features))
        user_features = DataLoader(user_features, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.ae.parameters(), lr=.001)
        criterion = torch.nn.BCELoss().to(self.device)
        self.ae.train()

        for _ in tqdm(range(10)):  # train 10 epochs
            for batch in user_features:
                batch = batch[0]
                batch.to(self.device)
                output, _ = self.ae(batch)  # omit encoded features
                loss = criterion(output, batch)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 0.1)
                optimizer.step()

        torch.save(self.ae.state_dict(), self.ae_path)

    def inference(self, data_path, concept_directory):
        user_docs = dict()
        user_concepts = dict()
        user_features = dict()

        with open(data_path) as dfile:
            for line in dfile:
                user = json.loads(line)
                user['uid'] = str(user['uid'])
                if user['uid'] not in user_docs:
                    user_docs[user['uid']] = []
                if user['uid'] not in user_concepts:
                    user_concepts[user['uid']] = []

                # because data builder takes patient per stay as a patient,
                # instead of multiple stays
                uid = user['uid']
                if '-' in uid:
                    uid = uid.split('-')[0]

                for doc_entity in user['docs']:
                    # collect data
                    if self.mode == 'average':
                        user_docs[user['uid']].append(doc_entity['text'].split())
                    else:
                        if len(user_docs[user['uid']]) == 0:
                            user_docs[user['uid']].append(doc_entity['text'].split())
                        else:
                            user_docs[user['uid']][0].extend(doc_entity['text'].split())

                    # get concept files
                    concept_fname = '{}_{}.pkl'.format(uid, doc_entity['doc_id'])
                    if os.path.exists(concept_directory + concept_fname):
                        concepts = pickle.load(open(concept_directory + concept_fname, 'rb'))
                        user_concepts[user['uid']].extend(
                            [concept['preferred_name'].lower() for concept in concepts])

        uids = list(user_docs.keys())
        self.ae.eval()
        batch_size = 128
        steps = len(uids) // batch_size
        if len(uids) % batch_size != 0:
            steps += 1

        for idx in range(steps):
            batch_uids = uids[idx*batch_size: (idx + 1)*batch_size]
            batch_features = [user_concepts[item_uid] for item_uid in batch_uids]
            batch_features = torch.FloatTensor(self.tf_idf_vect.transform(batch_features).toarray())
            # batch_features.to(self.device)
            with torch.no_grad():
                _, batch_features = self.ae(batch_features)
            batch_features = batch_features.cpu().detach().numpy()

            for batch_idx in range(len(batch_uids)):
                user_features[batch_uids[batch_idx]] = []
                # concept features
                user_features[batch_uids[batch_idx]].extend(batch_features[batch_idx])
                docs_features = [
                    self.doc2vec.infer_vector(doc) for doc in user_docs[batch_uids[batch_idx]]
                ]
                # average the lda inferred documents
                docs_features = np.mean(docs_features, axis=0)
                user_features[batch_uids[batch_idx]].extend(docs_features)

                # release memory
                del user_docs[batch_uids[batch_idx]]
                del user_concepts[batch_uids[batch_idx]]

        del user_docs
        del user_concepts

        return user_features


if __name__ == '__main__':
    task = sys.argv[1]
    task_data_path = '../data/processed_data/{}/{}.json'.format(task, task)
    concept_dir = '../data/processed_data/{}/concepts/'.format(task)

    baseline_dir = '../resources/embedding/'
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)

    task_dir = baseline_dir + task + '/'
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    odir = task_dir + 'suisil2user/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    opath_user = odir + 'user.txt'
    doc2vec_path = odir + 'doc2v.model'
    autoencoder_path = odir + 'ae_model.pth'

    if not os.path.exists(doc2vec_path):
        doc2vec_path = train_doc2v(
            task, input_path=task_data_path, odir=odir, dim=150
        )

    # Doc2Vec + Concept
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    l2u = Doc2User(
        task_name=task, doc2vec_path=doc2vec_path,
        tf_idf_path=get_tfidf_vect(task, concept_dir, odir),
        data_path=task_data_path, ae_path=autoencoder_path,
        device=device, concept_dir=concept_dir, mode='average',
    )

    # user vectors
    ufeatures = l2u.inference(task_data_path, concept_dir)

    # write to file
    wfile = open(opath_user, 'w')
    for tid in list(ufeatures.keys()):
        wfile.write(tid + '\t' + ' '.join(map(str, ufeatures[tid])) + '\n')
    wfile.flush()
    wfile.close()
