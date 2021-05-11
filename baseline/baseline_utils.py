import os
import pickle
import json
import sys

import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore


class RawCorpus(object):
    def __init__(self, filep, doc2id=False, dictionary=None):
        """ Load Json file
        """
        self.filep = filep
        self.dictionary = dictionary
        self.doc2id = doc2id

    def __iter__(self):
        with open(self.filep) as dfile:
            for line in dfile:
                user = json.loads(line)
                for doc_entity in user['docs']:
                    if self.doc2id and self.dictionary:  # this is for inference
                        yield self.dictionary.doc2bow(doc_entity['text'].split())
                    else:
                        yield doc_entity['text'].split()


def train_lda(dname, raw_dir='../data/raw/', odir='../resources/embedding/', dim=300):
    """
        The number of topics should be aligned with the dimensions of the user embedding.
    """
    odir = odir + dname + '/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    # load data and build dictionary
    dpath = raw_dir + dname + '/' + dname + '.json'
    if os.path.exists(odir + 'lda_dict.pkl'):
        dictionary = pickle.load(open(odir + 'lda_dict.pkl', 'rb'))
    else:
        corpus = RawCorpus(dpath)
        dictionary = Dictionary(corpus, prune_at=10000)
        dictionary.save(odir + 'lda_dict.pkl')

    doc_matrix = RawCorpus(dpath, True, dictionary)

    #    if dname == 'amazon1':
    #        model = LdaModel(
    #            doc_matrix, id2word=dictionary, num_topics=300,
    #            passes=5, alpha='symmetric'
    #        )
    #    else:
    model = LdaMulticore(
        doc_matrix, id2word=dictionary, num_topics=dim,
        passes=10, alpha='symmetric', workers=os.cpu_count()//2
    )
    model.save(odir + 'lda.model')


def train_doc2v(dname, input_path, odir='../resources/embedding/', dim=300):
    """ Build paragraph2vec model
    """
    print(dname)

    def read_corpus(data_path):
        with open(data_path) as dfile:
            for line in dfile:
                user = json.loads(line)
                for idx, doc_entity in enumerate(user['docs']):
                    line = doc_entity['text'].split()
                    yield TaggedDocument(line, [user['uid'] + str(idx)])

    if not os.path.exists(odir):
        os.mkdir(odir)

    # load the corpus
    corpus = read_corpus(input_path)

    # init, train and save the model
    model = Doc2Vec(
        vector_size=dim, min_count=2, epochs=30,
        workers=8, max_vocab_size=20000
    )
    model.build_vocab(corpus)

    model.train(
        corpus, total_examples=model.corpus_count,
        epochs=model.epochs
    )

    model.save(odir + 'doc2v.model')
    return odir + 'doc2v.model'


def train_w2v(dname, raw_dir='../data/raw/', odir='../resources/embedding/', pretrained=None, dim=300):
    dpath = raw_dir + dname + '/' + dname + '.tsv'
    corpus = RawCorpus(dpath)
    if not pretrained:
        model = Word2Vec(
            corpus, min_count=2, window=5,
            size=dim, sg=0, workers=8,
            max_vocab_size=20000,
        )
    else:
        model = Word2Vec.load(pretrained)
        model.build_vocab(corpus, update=True)
        model.train(corpus, epochs=10)

    odir = odir + dname + '/'
    if not os.path.exists(odir):
        os.mkdir(odir)
    model.save(odir + 'w2v')

    odir += 'w2v.txt'
    model.wv.save_word2vec_format(odir, binary=False)


def sample_decay(count, decay=2):
    """Calculate decay number for sampling user and product
    """
    return 1 / (1 + decay * count)


def user_word_sampler(uid, sequence, tokenizer, filter_words=None, negative_samples=1):
    """This function was partially adopted from
    https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/sequence.py#L151
    """
    couples = []
    labels = []
    word_set = set(sequence)
    if filter_words:
        word_set = word_set.union(filter_words)

    sample_size = 256
    # for wid in sequence:
    for wid in set(sequence):  # to reduce number of training instances, rush for conference deadline
        couples.append([uid, wid])
        labels.append(1)

    if len(couples) > sample_size:  # rush for conference deadline
        sample_indices = list(np.random.choice(list(range(len(labels))), size=sample_size, replace=False))
        couples = np.asarray(couples)
        labels = np.asarray(labels)
        couples = list(couples[sample_indices])
        labels = list(labels[sample_indices])

    norm_probs = [0] * tokenizer.num_words
    for wid in tokenizer.index_word:
        if wid > tokenizer.num_words-1:
            continue
        if wid in word_set:
            continue
        norm_probs[wid] = tokenizer.word_counts[tokenizer.index_word[wid]]**.75
    # scaling = int(np.ceil(1./min(norm_probs)))
    # norm_probs *= scaling
    norm_probs_sum = sum(norm_probs)
    norm_probs = [item/norm_probs_sum for item in norm_probs]
    negative_set = [item for item in range(tokenizer.num_words)]
    num_negative_samples = int(len(labels) * negative_samples)

    if negative_samples > 0:
        wid_list = np.random.choice(
            negative_set,
            size=num_negative_samples,
            p=norm_probs
        )
        for wid in wid_list:
            # 0 is placeholder, starts by 1
            # ensure user did not use the word
            couples.append([uid, wid])
            labels.append(0)

    # shuffle
    # seed = np.random.randint(0, int(10e6))
    # np.random.seed(seed)
    # np.random.shuffle(couples)
    # np.random.seed(seed)
    # np.random.shuffle(labels)
    return couples, labels


def npy2tsv(npy_path, idx2id_path, opath):
    """Convert Index to item (user or product) IDs, follow with their embedding
    """
    embs = np.load(npy_path)
    idx2id = json.load(open(idx2id_path))
    idx2id = {v: k for k, v in idx2id.items()}

    with open(opath, 'w') as wfile:
        for idx in range(len(embs)):
            if idx not in idx2id:
                continue
            wfile.write('{}\t{}\n'.format(idx2id[idx], ' '.join(map(str, embs[idx]))))


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)


if __name__ == '__main__':
    save_dir = '../resources/embedding/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    data_dir = '../data/processed_data/'

    data_name = sys.argv[1]  # ['amazon', 'diabetes', 'mimic-iii']
    model_name = sys.argv[2]  # lda, word2vec, doc2vec and bert
    if data_name not in ['amazon', 'diabetes', 'mimic-iii']:
        raise ValueError('Data {} is not supported currently...'.format(data_name))

    if model_name == 'word2vec':
        print('Training Word Embeddings: ', data_name)
        train_w2v(data_name, raw_dir=data_dir, odir=save_dir)
    elif model_name == 'lda':
        print('Training LDA: ', data_name)
        train_lda(data_name, raw_dir=data_dir, odir=save_dir)
    elif model_name == 'doc2vec':
        print('Training Doc2vec: ', data_name)
        train_doc2v(
            data_name, 
            input_path=data_dir+data_name+'/{}.json'.format(data_name), 
            odir=save_dir+data_name+'/'
        )
    else:
        raise ValueError('Model name, {}, is not in supported now...'.format(model_name))
