"""Extract document representations by average word representations,
average all document vectors as user or product representations
"""
import os
import sys
import pickle
import json

import gensim
import numpy as np


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

    def __init__(self, task_name, tknPath, modelPath, emb_dim=300):
        self.task = task_name
        self.tkn = pickle.load(open(tknPath, 'rb'))
        self.emb_dim = emb_dim
        self.model = self.__load_model(modelPath, emb_dim)

    def __load_model(self, modelPath, emb_dim=300):
        # support three types, bin/txt/npy
        emb_len = len(self.tkn.word_index)
        if emb_len > self.tkn.num_words:
            emb_len = self.tkn.num_words
        model = np.zeros((emb_len + 1, emb_dim))

        if modelPath.endswith('.bin'):
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                modelPath, binary=True
            )
            for pair in zip(w2v_model.wv.index2word, w2v_model.wv.syn0):
                if pair[0] in self.tkn.word_index and \
                        self.tkn.word_index[pair[0]] < self.tkn.num_words:
                    model[self.tkn.word_index[pair[0]]] = pair[1]

        elif modelPath.endswith('.npy'):
            model = np.load(modelPath)

        elif modelPath.endswith('.txt'):
            with open(modelPath) as dfile:
                for line in dfile:
                    line = line.strip().split()
                    word = line[0]
                    vectors = np.asarray(line[1:], dtype='float32')

                    if word in self.tkn.word_index and \
                            self.tkn.word_index[word] < self.tkn.num_words:
                        model[self.tkn.word_index[word]] = vectors
        else:
            raise ValueError('Current other formats are not supported!')
        return model

    def word2item(self, data_path, opath, max_len=512):
        """Extract user vectors from the given data path

            Parameters
            ----------
            max_len
            data_path: str
                Path of data file, tsv file
            opath: str
                Path of output path for user vectors
        """
        item_dict = dict()
        ofile = open(opath, 'w')

        with open(data_path) as dfile:
            for line in dfile:
                user = json.loads(line)
                tid = user['uid']
                if tid not in item_dict:
                    item_dict[tid] = []

                for doc_entity in user['docs']:
                    text = doc_entity['text']
                    # collect data
                    item_dict[tid].extend(text.split()[:max_len])

        for tid in list(item_dict.keys()):
            # encode the document by word2vec
            item_dict[tid] = {}
            # average the word2vec inferred documents
            item_dict[tid]['word'] = np.mean(np.asarray([
                self.model[self.tkn.word_index[word]] for word in item_dict[tid]
                if word in self.tkn.word_index and self.tkn.word_index[word] < self.tkn.num_words
            ]), axis=0)
            # average the word2vec inferred concepts
            item_dict[tid]['concept'] = np.mean(np.asarray([
                self.model[self.tkn.word_index[concept]] for concept in item_dict[tid]
                if concept in self.tkn.word_index and self.tkn.word_index[concept] < self.tkn.num_words
            ]), axis=0)

            # write to file
            ofile.write(tid + '\t' + ' '.join(map(str, item_dict[tid])) + '\n')

            # save memory
            del item_dict[tid]
        ofile.flush()
        ofile.close()


if __name__ == '__main__':
    dname = sys.argv[1]
    raw_dir = '../data/processed_data/'
    task_data_path = raw_dir + dname + '/' + dname + '.json'

    data_dir = raw_dir + dname + '/'
    baseline_dir = '../resources/embedding/'
    task_dir = baseline_dir + dname + '/'
    odir = task_dir + 'word2user/'
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
    l2u = Word2User(dname, tkn_path, model_path)
    # user vectors
    l2u.word2item(
        data_path=task_data_path,
        opath=opath_user
    )
