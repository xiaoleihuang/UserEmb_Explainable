"""This script implements the paper of Multi-View Unsupervised User Feature Embedding
for Social Media-based Substance Use Prediction: Post-D-DBOW

doc2vec portion
Patient representation learning and interpretable evaluation using clinical notes
"""

import sys
import os
import numpy as np
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from baseline_utils import data_loader


def train_concept_doc2v(concept_list, output_dir, dim=300):
    """ Build paragraph2vec model
    """
    def read_corpus(data_list):
        for line_id, line in data_list:
            yield TaggedDocument(line, [line_id])

    # load the corpus
    corpus = read_corpus(concept_list)

    # init, train and save the model
    model = Doc2Vec(
        vector_size=dim, min_count=2, epochs=30,
        workers=os.cpu_count()//2, max_vocab_size=15000
    )
    model.build_vocab(corpus)

    model.train(
        corpus, total_examples=model.corpus_count,
        epochs=model.epochs
    )

    model.save(output_dir + 'doc2v_concept.model')


class Doc2User(object):
    def __init__(self, **kwargs):
        """Apply Doc2Vec model on the documents to generate user and product representation.
            Outputs will be one user/product_id + vect per line.
        """
        self.task = kwargs['task_name']
        self.doc_model = Doc2Vec.load(kwargs['doc_model_path'])
        self.concept_model = Doc2Vec.load(kwargs['concept_model_path'])
        self.doc_model.delete_temporary_training_data(
            keep_doctags_vectors=True, keep_inference=True)
        self.concept_model.delete_temporary_training_data(
            keep_doctags_vectors=True, keep_inference=True)

    def doc2item(self, data_path, opath):
        """Extract user vectors from the given data path

        :param data_path:
        :param opath:
        :return:
        """
        user_docs, all_docs = data_loader(data_path)
        ofile = open(opath, 'w')

        for tid in tqdm(list(user_docs.keys())):
            # encode the document by doc2vec
            doc_vectors = np.asarray([
                self.doc_model.infer_vector(all_docs[doc_id].split()) for doc_id in user_docs[tid]['docs']
            ])
            concept_vectors = np.asarray([
                self.concept_model.infer_vector(concepts) for concepts in user_docs[tid]['concepts']
                if len(concepts) > 0
            ])

            # average the lda inferred documents
            doc_vectors = np.mean(doc_vectors, axis=0)
            concept_vectors = np.mean(concept_vectors, axis=0)

            # write to file
            ofile.write(tid + '\t' + ' '.join(map(
                str, np.concatenate((doc_vectors, concept_vectors), axis=None))) + '\n')

        ofile.flush()
        ofile.close()


if __name__ == '__main__':
    dname = sys.argv[1]
    task_data_path = '../resources/embedding/{}/caue_gru/user_docs_concepts.pkl'.format(dname)

    baseline_dir = '../resources/embedding/'
    # create directories
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)

    task_dir = baseline_dir + dname + '/'
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    odir = task_dir + 'doc2user_concept/'
    if not os.path.exists(odir):
        os.mkdir(odir)
    opath_user = odir + 'user.txt'
    doc_model_path = task_dir + 'doc2v.model'

    concept_model_path = odir + 'doc2v_concept.model'
    if not os.path.exists(concept_model_path):
        user_data, lda_docs = data_loader(task_data_path)
        clist = []
        for uid in user_data:
            clist.extend([(uid, concepts) for concepts in user_data[uid]['concepts'] if len(concepts) > 0])
        train_concept_doc2v(concept_list=clist, output_dir=odir, dim=300)

    # Doc2User
    d2u = Doc2User(
        task_name=dname, doc_model_path=doc_model_path, concept_model_path=concept_model_path)
    # user vectors
    d2u.doc2item(
        data_path=task_data_path, 
        opath=opath_user
    )
