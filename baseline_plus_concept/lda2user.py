"""Extract topics from each document, average all topic vectors as user representations
This script is to implement methods from Multi-View Unsupervised User Feature Embedding for Social
Media-based Substance Use Prediction

PostLDA-Doc
"""
import pickle
import numpy as np
import sys
import os
from multiprocessing import Pool

from baseline_utils import data_loader
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from tqdm import tqdm


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
    def __init__(self, **kwargs):
        self.task = kwargs['task_name']
        self.word_dict = pickle.load(open(kwargs['word_dict_path'], 'rb'))
        self.concept_dict = pickle.load(open(kwargs['concept_dict_path'], 'rb'))
        self.word_model = LdaModel.load(kwargs['word_model_path'])
        self.concept_model = LdaModel.load(kwargs['concept_model_path'])

    def lda2item_thread(self, info):
        tid, docs, concepts_list = info
        doc_vectors = []
        for idx, doc in enumerate(docs):
            output = self.word_model[self.word_dict.doc2bow(doc.split())]
            doc_vectors.append([0.] * self.word_model.num_topics)
            for item in output:
                doc_vectors[-1][item[0]] = item[1]
        # average the lda inferred documents
        doc_vectors = np.mean(doc_vectors, axis=0)

        concept_vectors = []
        for idx, concepts in enumerate(concepts_list):
            if len(concepts) == 0:
                continue
            output = self.concept_model[self.concept_dict.doc2bow(concepts)]
            concept_vectors.append([0.] * self.concept_model.num_topics)
            for item in output:
                concept_vectors[-1][item[0]] = item[1]
        # average the lda inferred documents
        concept_vectors = np.mean(concept_vectors, axis=0)
        return tid, np.concatenate((doc_vectors, concept_vectors), axis=None)

    def lda2item_parallel(self, data_path, opath):
        ofile = open(opath, 'w')
        # load the datasets from caue_gru task
        user_docs, all_docs = data_loader(data_path)
        parallel_info = []
        print('Loading Data')
        for tid in list(user_docs.keys()):
            # encode the document by lda
            # docs = list(itertools.chain.from_iterable([all_docs[doc_id] for doc_id in user_docs[tid]['docs']]))
            docs = [all_docs[doc_id] for doc_id in user_docs[tid]['docs']]
            parallel_info.append((tid, docs, user_docs[tid]['concepts']))

        pool = Pool(os.cpu_count()//2)
        results = pool.map(self.lda2item_thread, parallel_info)

        for tid, vector in results:
            # write to file
            ofile.write(tid + '\t' + ' '.join(map(str, vector)) + '\n')

        ofile.flush()
        ofile.close()

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

        print('Loading Data')
        for tid in tqdm(list(user_docs.keys())):
            # encode the document by lda
            # docs = list(itertools.chain.from_iterable([all_docs[doc_id] for doc_id in user_docs[tid]['docs']]))
            docs = [all_docs[doc_id] for doc_id in user_docs[tid]['docs']]
            doc_vectors = []
            for idx, doc in enumerate(docs):
                output = self.word_model[self.word_dict.doc2bow(doc.split())]
                doc_vectors.append([0.] * self.word_model.num_topics)
                for item in output:
                    doc_vectors[-1][item[0]] = item[1]
            # average the lda inferred documents
            doc_vectors = np.mean(doc_vectors, axis=0)

            concept_vectors = []
            for idx, concepts in enumerate(user_docs[tid]['concepts']):
                if len(concepts) == 0:
                    continue
                output = self.concept_model[self.concept_dict.doc2bow(concepts)]
                concept_vectors.append([0.] * self.concept_model.num_topics)
                for item in output:
                    concept_vectors[-1][item[0]] = item[1]
            # average the lda inferred documents
            concept_vectors = np.mean(concept_vectors, axis=0)

            # write to file
            ofile.write(tid + '\t' + ' '.join(map(
                str, np.concatenate((doc_vectors, concept_vectors), axis=None))) + '\n')

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

    odir = task_dir + 'lda2user_concept/'
    if not os.path.exists(odir):
        os.mkdir(odir)
    opath_user = odir + 'user.txt'

    word_dict_path = odir + 'lda_dict.pkl'
    word_model_path = odir + 'lda.model'

    concept_model_path = odir + 'concept_lda.model'
    concept_dict_path = odir + 'concept_lda_dict.pkl'

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

    # Lda2User
    l2u = Lda2User(
        task_name=task, word_dict_path=word_dict_path, concept_model_path=concept_model_path,
        concept_dict_path=concept_dict_path, word_model_path=word_model_path
    )
    # user vectors
    # l2u.lda2item_parallel(
    l2u.lda2item(
        data_path=task_data_path, 
        opath=opath_user,
    )
