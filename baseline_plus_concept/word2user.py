"""Extract document representations by average word representations,
average all document vectors as user or product representations
"""
import os
import sys
import pickle
import itertools

import gensim
import numpy as np
from tqdm import tqdm
from baseline_utils import data_loader


class Word2User(object):
    """Apply Word2Vec model on the documents to generate user and product representation.
        Outputs will be one user/product_id + vect per line.

        Parameters
        ----------
        task_name: str
            Task name, such as amazon, yelp and imdb
        tknPath: str
            Path of LDA dictionary file
        modelPath: str
            Path of LDA model file
    """

    def __init__(self, **kwargs):
        self.task = kwargs['task_name']
        self.word_tkn = pickle.load(open(kwargs['word_tkn_path'], 'rb'))
        self.concept_tkn = pickle.load(open(kwargs['concept_tkn_path'], 'rb'))
        self.emb_dim = kwargs['emb_dim']
        self.word_model = self.__load_model(kwargs['word_emb_path'], self.word_tkn, kwargs['emb_dim'])
        self.concept_model = self.__load_model(kwargs['concept_emb_path'], self.concept_tkn, kwargs['emb_dim'])

    def __load_model(self, modelPath, tkn, emb_dim=300):
        # support three types, bin/txt/npy
        if type(tkn) == dict:
            emb_len = len(tkn)
        else:
            emb_len = len(self.word_tkn.word_index)
            if emb_len > self.word_tkn.num_words:
                emb_len = self.word_tkn.num_words
        model = np.zeros((emb_len + 1, emb_dim))

        if modelPath.endswith('.bin'):
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                modelPath, binary=True
            )
            for pair in zip(w2v_model.wv.index2word, w2v_model.wv.syn0):
                if pair[0] in tkn.word_index and \
                        tkn.word_index[pair[0]] < tkn.num_words:
                    model[tkn.word_index[pair[0]]] = pair[1]

        elif modelPath.endswith('.npy'):
            model = np.load(modelPath)

        elif modelPath.endswith('.txt'):
            with open(modelPath) as dfile:
                for line in dfile:
                    line = line.strip().split()
                    word = line[0]
                    vectors = np.asarray(line[1:], dtype='float32')

                    if word in tkn.word_index and \
                            tkn.word_index[word] < tkn.num_words:
                        model[tkn.word_index[word]] = vectors
        else:
            raise ValueError('Current other formats are not supported!')
        return model

    def word2item(self, data_path, opath):
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

        for tid in tqdm(list(user_docs.keys())):
            # encode the document by word2vec
            docs = list(itertools.chain.from_iterable([all_docs[doc_id] for doc_id in user_docs[tid]['docs']]))
            docs = [doc.split() for doc in docs]
            docs = list(itertools.chain.from_iterable(docs))
            concepts = list(itertools.chain.from_iterable(user_docs[tid]['concepts']))

            # average the word2vec inferred documents
            word_emb = np.mean(np.asarray([
                self.word_model[self.word_tkn.word_index[word]] for word in docs
                if word in self.word_tkn.word_index and self.word_tkn.word_index[word] < self.word_tkn.num_words
            ]), axis=0)
            # average the word2vec inferred concepts
            concept_emb = np.mean(np.asarray([
                self.concept_model[self.concept_tkn[concept]] for concept in concepts
                if concept in self.concept_tkn
            ]), axis=0)

            # write to file
            ofile.write(
                tid + '\t' + ' '.join(map(
                    str, np.concatenate((word_emb, concept_emb), axis=None))
                ) + '\n'
            )

        ofile.flush()
        ofile.close()


if __name__ == '__main__':
    dname = sys.argv[1]
    raw_dir = '../data/processed_data/'
    task_data_path = '../resources/embedding/{}/caue_gru/user_docs_concepts.pkl'.format(dname)

    data_dir = raw_dir + dname + '/'
    baseline_dir = '../resources/embedding/'
    task_dir = baseline_dir + dname + '/'
    odir = task_dir + 'word2user_concept/'
    opath_user = odir + 'user.txt'

    resource_dir = '../resources/embedding/'
    tkn_path = '../data/processed_data/' + dname + '/' + dname + '.tkn'
    model_path = resource_dir + dname + '/word_emb.npy'

    # create directories
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    if not os.path.exists(odir):
        os.mkdir(odir)

    # Word2User
    l2u = Word2User(
        task_name=dname, word_tkn_path=tkn_path, word_emb_path=model_path,
        concept_tkn_path=data_dir+'concept_tkn.pkl', emb_dim=300,
        concept_emb_path='../resources/embedding/{}/caue_gru/{}_concept_emb.npy'.format(dname, dname)
    )
    # user vectors
    l2u.word2item(
        data_path=task_data_path,
        opath=opath_user
    )
