"""This script implements the paper of Multi-View Unsupervised User Feature Embedding
for Social Media-based Substance Use Prediction: Post-D-DBOW

doc2vec portion
Patient representation learning and interpretable evaluation using clinical notes
"""

from gensim.models.doc2vec import Doc2Vec
import sys
import os
import json
import numpy as np


class Doc2User(object):
    def __init__(self, task, mpath):
        """Apply Doc2Vec model on the documents to generate user and product representation.
            Outputs will be one user/product_id + vect per line.

            Parameters
            ----------
            task: str
                Task name, such as amazon, yelp and imdb
            mpath: str
                Path of LDA model file
        """
        self.task = task
        self.model = Doc2Vec.load(mpath)
        self.model.delete_temporary_training_data(
            keep_doctags_vectors=True, keep_inference=True)

    def doc2item(self, data_path, opath, mode='average'):
        """Extract user vectors from the given data path

        :param data_path:
        :param opath:
        :param mode:
        :return:
        """
        item_dict = dict()
        ofile = open(opath, 'w')

        print('Loading Data')
        with open(data_path) as dfile:
            for line in dfile:
                user = json.loads(line)
                tid = user['uid']
                if tid not in item_dict:
                    item_dict[tid] = []

                for doc in user['docs']:
                    text = doc['text']

                    # collect data
                    if mode == 'average':
                        item_dict[tid].append(text.split())
                    else:
                        if len(item_dict[tid]) == 0:
                            item_dict[tid].append(text.split())
                        else:
                            item_dict[tid][0].extend(text.split())

        for tid in list(item_dict.keys()):
            print('Working on user: ', tid)
            # encode the document by doc2vec
            item_dict[tid] = np.asarray([
                self.model.infer_vector(doc) for doc in item_dict[tid]
            ])
            # average the lda inferred documents
            item_dict[tid] = np.mean(item_dict[tid], axis=0)

            # write to file
            ofile.write(tid + '\t' + ' '.join(map(str, item_dict[tid])) + '\n')

            # save memory
            del item_dict[tid]
        ofile.flush()
        ofile.close()


if __name__ == '__main__':
    dname = sys.argv[1]
    task_data_path = '../data/processed_data/{}/{}.json'.format(dname, dname)

    baseline_dir = '../resources/embedding/'
    # create directories
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)

    task_dir = baseline_dir + dname + '/'
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    odir = task_dir + 'doc2user/'
    if not os.path.exists(odir):
        os.mkdir(odir)
    opath_user = odir + 'user.txt'
    model_path = task_dir + 'doc2v.model'

    # Doc2User
    d2u = Doc2User(dname, model_path)
    # user vectors
    d2u.doc2item(
        data_path=task_data_path, 
        opath=opath_user
    )
