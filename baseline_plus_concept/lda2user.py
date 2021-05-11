"""Extract topics from each document, average all topic vectors as user representations
This script is to implement methods from Multi-View Unsupervised User Feature Embedding for Social
Media-based Substance Use Prediction

PostLDA-Doc
"""
import pickle
import numpy as np
import sys
import json
import os
import itertools

from baseline_utils import data_loader
from gensim.models import LdaModel


class Lda2User(object):
    """Apply LDA model on the documents to generate user and product representation.
        Outputs will be one user/product_id + vect per line.

        Parameters
        ----------
        task_name: str
            Task name, such as amazon, yelp and imdb
        dictionary_path: str
            Path of LDA dictionary file
        mpath: str
            Path of LDA model file
    """
    def __init__(self, task_name, dictionary_path, mpath):
        self.task = task_name
        self.dictionary = pickle.load(open(dictionary_path, 'rb'))
        self.model = LdaModel.load(mpath)

    def lda2item(self, data_path, opath, mode='average'):
        """Extract user vectors from the given data path

            Parameters
            ----------
            data_path: str
                Path of data file, tsv file
            opath: str
                Path of output path for user vectors
            mode: str
                Methods to combine document representations
        """
        item_dict = dict()
        ofile = open(opath, 'w')

        print('Loading Data')
        with open(data_path) as dfile:
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

        for tid in list(item_dict.keys()):
            # encode the document by lda
            for idx, doc in enumerate(item_dict[tid]):
                output = self.model[self.dictionary.doc2bow(doc)]
                item_dict[tid][idx] = [0.] * self.model.num_topics
                for item in output:
                    item_dict[tid][idx][item[0]] = item[1]
            # average the lda inferred documents
            item_dict[tid] = np.mean(item_dict[tid], axis=0)

            # write to file
            ofile.write(tid + '\t' + ' '.join(map(str, item_dict[tid])) + '\n')

            # save memory
            del item_dict[tid]
        ofile.flush()
        ofile.close()


if __name__ == '__main__':
    task = sys.argv[1]
    task_data_path = '../data/processed_data/{}/{}.json'.format(task, task)

    baseline_dir = '../resources/embedding/'
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)

    task_dir = baseline_dir + task + '/'
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    odir = task_dir + 'lda2user/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    opath_user = odir + 'user.txt'

    dict_path = task_dir + 'lda_dict.pkl'
    model_path = task_dir + 'lda.model'

    # Lda2User
    l2u = Lda2User(task, dict_path, model_path)
    # user vectors
    l2u.lda2item(
        data_path=task_data_path, 
        opath=opath_user,
        mode='average'
    )
