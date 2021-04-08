"""
Implementation of Patient representation learning and interpretable evaluation using clinical notes

"""
import pickle
import sys
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from tqdm import tqdm


def get_tfidf_vect(task_name, data_path, odir):
    corpus = []
    with open(data_path) as dfile:
        for line in dfile:
            user = json.loads(line)
            for doc_entity in user['docs']:
                corpus.append(doc_entity['text'])
    tfidf_vect = TfidfVectorizer(min_df=2, max_features=3000)
    tfidf_vect.fit(corpus)
    with open(os.path.join(odir, 'tfidf_vect_{}.pkl'.format(task_name)), 'wb') as wfile:
        pickle.dump(tfidf_vect, wfile)


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
        features = F.sigmoid(features)
        code_features = self.encoder_output_layer(features)
        code_features = F.sigmoid(code_features)
        decode_features = self.decoder_hidden_layer(code_features)
        decode_features = F.sigmoid(decode_features)
        decode_features = self.decoder_output_layer(decode_features)
        reconstructed = F.sigmoid(decode_features)
        return reconstructed, code_features


class Doc2User(object):
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
        self.dictionary = pickle.load(open(kwargs['dictionary_path'], 'rb'))
        self.model = Doc2Vec.load(kwargs['doc2vec_path'])
        self.data_path = kwargs['data_path']
        self.ae_path = kwargs['ae_path']
        self.device = kwargs['device']
        self.tf_idf_vect = pickle.load(open(kwargs['tf_idf_path'], 'rb'))

        self.ae = AE(, 150)
        if not self.ae_path:
            self.train_autoencoder()
        else:
            self.ae.load_state_dict(torch.load(self.ae_path), strict=False)

    def train_autoencoder(self):
        user_features = list(self.doc2user().values())
        user_features = TensorDataset(torch.tensor(user_features))
        user_features = DataLoader(user_features, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.ae.parameters(), lr=.001)
        criterion = torch.nn.BCELoss().to(self.device)
        self.ae.train()

        for _ in tqdm(range(10)):
            for idx, batch in enumerate(user_features):
                batch.to(self.device)
                output, _ = self.ae(batch)  # omit encoded features
                loss = criterion(output, batch)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 0.1)
                optimizer.step()

        torch.save(self.ae.state_dict(), self.ae_path)

    def doc2user(self, mode='average'):
        """Extract user vectors from the given data path

            Parameters
            ----------
            mode: str
                Methods to combine document representations
        """
        item_dict = dict()

        print('Loading Data')
        with open(self.data_path) as dfile:
            for line in dfile:
                user = json.loads(line)
                if user['uid'] not in item_dict:
                    item_dict[user['uid']] = []
                for doc_entity in user['docs']:
                    # collect data
                    if mode == 'average':
                        item_dict[user['uid']].append(doc_entity['text'].split())
                    else:
                        if len(item_dict[user['uid']]) == 0:
                            item_dict[user['uid']].append(doc_entity['text'].split())
                        else:
                            item_dict[user['uid']][0].extend(doc_entity['text'].split())

        for uid in list(item_dict.keys()):
            # encode the document by lda
            for idx, doc in enumerate(item_dict[uid]):
                output = self.model[self.dictionary.doc2bow(doc)]
                item_dict[uid][idx] = [0.] # TODO
                for item in output:
                    item_dict[uid][idx][item[0]] = item[1]
            # average the lda inferred documents
            item_dict[uid] = np.mean(item_dict[uid], axis=0)

        return item_dict

    def inference(self, user_features=None):
        self.ae.eval()
        if not user_features:
            user_features = self.doc2user()
        if not torch.is_tensor(user_features):
            user_features = torch.tensor(user_features)

        _, user_embs = self.ae(user_features)
        user_embs = user_embs.cpu().detach().numpy()
        return user_embs


if __name__ == '__main__':
    task = sys.argv[1]
    task_data_path = '../data/processed_data/{}/{}.json'.format(task, task)

    baseline_dir = '../resources/embedding/'
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)

    task_dir = baseline_dir + task + '/'
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    odir = task_dir + 'doc2user/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    opath_user = odir + 'user.txt'

    dict_path = task_dir + 'lda_dict.pkl'
    model_path = task_dir + 'lda.model'
    autoencoder_path = task_dir + 'ae_model.pth'

    # Lda2User
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    l2u = Doc2User(
        task_name=task, dictionary_path=dict_path, doc2vec_path=model_path,
        data_path=task_data_path, ae_path=autoencoder_path, device=device
    )

    # user vectors
    user_topics = l2u.doc2user(mode='average')
    ufeatures = l2u.inference(list(user_topics.values()))
    user_topics = dict(zip(list(user_topics.keys()), ufeatures))

    # write to file
    wfile = open(opath_user, 'w')
    for tid in list(user_topics.keys()):
        wfile.write(tid + '\t' + ' '.join(map(str, user_topics[tid])) + '\n')
    wfile.flush()
    wfile.close()
