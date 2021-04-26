import json
import os
import pickle
import sys

import keras
import numpy as np

from keras_preprocessing.text import Tokenizer
import gensim
from gensim.models.doc2vec import Doc2Vec
from keras.preprocessing.sequence import pad_sequences
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for cpu usage
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def user_doc_builder(user_docs, all_docs, user_docs_indices, negative_samples=1):
    """This function was re-implemented from the Silvio Amir
    https://github.com/samiroid/usr2vec/tree/master/code

        user_docs (dict): user docs
        sequence (list): a sequence of word indices
        emb_dim (int): document size
    """
    uids = list(user_docs.keys())
    np.random.shuffle(uids)
    couples = []
    labels = []

    for uid in uids:
        for doc in user_docs[uid]:
            couples.append([uid, doc])
            labels.append(1)

            sample_space = [idx for idx in list(range(len(all_docs))) if idx not in user_docs_indices[uid]]
            sample_space = np.random.choice(sample_space, size=negative_samples, replace=False)
            for idx in sample_space:
                labels.append(0)
                couples.append([uid, all_docs[idx]])

    couples = np.asarray(couples, dtype=object)
    labels = np.asarray(labels)

    return couples, labels


def user_doc_generator(couples, labels, batch_size):
    # shuffle the dataset
    rand_indices = list(range(len(labels)))
    np.random.shuffle(rand_indices)
    couples = couples[rand_indices]
    labels = labels[rand_indices]

    steps = len(labels) // batch_size
    if len(labels) % batch_size != 0:
        steps += 1

    for idx in range(steps):
        yield couples[batch_size * idx: batch_size * (idx + 1)], labels[batch_size * idx: batch_size * (idx + 1)]


# design model
def build_model(params=None):
    """
        params (dict): a dictionary of parameter settings
    """
    if not params:  # default params
        params = {
            'batch_size': 5,
            'vocab_size': 20001,
            'user_size': 8000,
            'emb_dim': 300,
            'dp_rate': .2,
            'word_emb_path': '../resources/word_emb.npy',
            'user_emb_path': '../resources/user_emb.npy',
            'word_emb_train': False,
            'user_emb_train': True,
            'user_task_weight': 1,
            'word_task_weight': 1,
            'epochs': 10,
            'optimizer': 'adam',
            'lr': 0.00001,
            'max_len': 1000,
        }

    # User Input
    user_input = keras.layers.Input((1,), name='user_input', dtype='int32')
    doc_input = keras.layers.Input((params['max_len'],), name='doc_input', dtype='int32')

    # load weights if word embedding path is given
    if os.path.exists(params['word_emb_path']):
        doc_emb = keras.layers.Embedding(
            params['emb_dim'], params['emb_dim'],
            weights=[np.load(params['doc_emb_path'])],
            trainable=params['word_emb_train'], name='doc_emb'
        )
    else:
        doc_emb = keras.layers.Embedding(
            params['emb_dim'], params['emb_dim'],
            trainable=params['doc_emb_path'], name='doc_emb'
        )

    # load weights if user embedding path is given
    if os.path.exists(params['user_emb_path']):
        user_emb = keras.layers.Embedding(
            params['user_size'], params['emb_dim'],
            weights=[np.load(params['user_emb_path'])],
            trainable=params['user_emb_train'], name='user_emb'
        )
    else:
        user_emb = keras.layers.Embedding(
            params['user_size'], params['emb_dim'],
            trainable=params['user_emb_train'], name='user_emb'
        )

    '''User Document Dot Production'''
    user_rep = user_emb(user_input)
    doc_rep = doc_emb(doc_input)

    user_doc_dot = keras.layers.dot(
        [user_rep, doc_rep], axes=-1
    )
    user_doc_dot = keras.layers.Reshape((1,))(user_doc_dot)
    user_pred = keras.layers.Dense(
        1, activation='sigmoid', name='user_pred'
    )(user_doc_dot)

    '''Compose model'''
    if params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(lr=params['lr'])
    else:
        optimizer = keras.optimizers.SGD(
            lr=params['lr'], decay=1e-6, momentum=0.9, nesterov=True)

    # user doc model
    ud_model = keras.models.Model(
        inputs=[user_input, doc_input],
        outputs=user_pred
    )
    ud_model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return ud_model


def build_emb_layer(tokenizer, emb_path, save_path, emb_dim=300):
    """"""
    # support three types, bin/txt/npy
    emb_len = len(tokenizer.word_index)
    if emb_len > tokenizer.num_words:
        emb_len = tokenizer.num_words

    emb_model = np.zeros((emb_len, emb_dim))
    if emb_path.endswith('.bin'):
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            emb_path, binary=True
        )
        for pair in zip(w2v_model.wv.index2word, w2v_model.wv.syn0):
            if pair[0] in tokenizer.word_index and \
                    tokenizer.word_index[pair[0]] < tokenizer.num_words:
                emb_model[tokenizer.word_index[pair[0]]] = pair[1]

        # save the extracted embedding weights
        np.save(save_path, emb_model)

    elif emb_path.endswith('.txt'):
        with open(emb_path) as dfile:
            for line in dfile:
                line = line.strip().split()
                word = line[0]
                vectors = np.asarray(line[1:], dtype='float32')
                if len(vectors) != emb_dim:
                    continue

                if word in tokenizer.word_index and \
                        tokenizer.word_index[word] < tokenizer.num_words:
                    emb_model[tokenizer.word_index[word]] = vectors
        # save the extracted embedding weights
        np.save(save_path, emb_model)
    else:
        raise ValueError('Current other formats are not supported!')


def main(data_name, encode_directory, odirectory='../resources/'):
    # load corpus data
    user_corpus = dict()
    all_docs = []
    user_docs_indices = dict()

    # load tokenizer
    if os.path.exists(encode_directory + data_name + '.tkn'):
        tok = pickle.load(open(encode_directory + data_name + '.tkn', 'rb'))
    else:
        tok = Tokenizer(num_words=20001)  # 20000 known + 1 unkown tokens
        tok.fit_on_texts(all_docs)
        pickle.dump(tok, open(encode_directory + data_name + '.tkn', 'wb'))

    params = {
        'batch_size': 16,
        'vocab_size': tok.num_words,
        'user_size': -1,  # +1 for unknown
        'emb_dim': 300,
        'dp_rate': .2,
        'emb_path': '/data/models/BioWordVec_PubMed_MIMICIII_d200.vec.bin',
        'word_emb_path': '../resources/embedding/{}/word_emb.npy'.format(data_name),
        'user_emb_path': '../resources/embedding/{}/user_emb.npy'.format(data_name),
        'word_emb_train': False,
        'user_emb_train': True,
        'user_task_weight': 1,
        'epochs': 15,
        'optimizer': 'adam',
        'lr': 1e-4,
        'negative_sample': 3,
        'max_len': 3000,
    }

    # load user encoder, which convert users into indices
    user_encoder = dict()
    if os.path.exists(encode_directory + 'user_encoder.json'):
        user_encoder = json.load(open(encode_directory + 'user_encoder.json'))

    with open(encode_directory + data_name + '.json') as dfile:
        for idx, line in enumerate(dfile):
            user_info = json.loads(line)
            if user_info['uid'] not in user_encoder:
                user_encoder[user_info['uid']] = len(user_encoder)
            user_info['uid'] = user_encoder[user_info['uid']]

            user_corpus[user_info['uid']] = []
            user_docs_indices[user_info['uid']] = []

            for doc in user_info['docs']:
                processed_doc = pad_sequences(
                    tok.texts_to_sequences([doc['text']]), maxlen=params['max_len'])[0]
                user_corpus[user_info['uid']].append(processed_doc)
                all_docs.append(processed_doc)
                user_docs_indices[user_info['uid']].append(len(all_docs) - 1)
    # update user information
    params['user_size'] = len(user_docs_indices) + 1
    if not os.path.exists(encode_directory + 'user_encoder.json'):
        json.dump(user_encoder, open(encode_directory + 'user_encoder.json', 'w'))

    if not os.path.exists('../resources/embedding/{}/'.format(data_name)):
        os.mkdir('../resources/embedding/{}/'.format(data_name))

    # build datasets
    user_docs, labels = user_doc_builder(
        user_corpus, all_docs, user_docs_indices, negative_samples=params['negative_sample']
    )

    # build embedding model
    build_emb_layer(
        tok, params['emb_path'], params['word_emb_path'], params['emb_dim']
    )

    ud_model = build_model(params)
    print()
    print(params)

    for epoch in range(params['epochs']):
        loss = 0

        train_iter = user_doc_generator(
            user_docs, labels, params['batch_size']
        )

        for step, train_batch in enumerate(train_iter):
            '''user info, uw: user-word'''
            ud_pairs, ud_labels = train_batch
            ud_pairs = [np.array(x) for x in zip(*ud_pairs)]
            ud_labels = np.array(ud_labels, dtype=np.int32)

            '''Train'''
            loss += ud_model.train_on_batch(ud_pairs, ud_labels)

            loss_avg = loss / (step + 1)
            if step % 100 == 0:
                print('Epoch: {}, Step: {}'.format(epoch, step))
                print('\tLoss: {}.'.format(loss_avg))
                print('-------------------------------------------------')

    # save the model
    ud_model.save(odirectory + 'ud_model.h5')
    # save the user embedding
    np.save(odirectory + 'user.npy', ud_model.get_layer(name='user_emb').get_weights()[0])


if __name__ == '__main__':
    dname = sys.argv[1]

    encode_dir = '../data/processed_data/'
    encode_dir = encode_dir + dname + '/'

    odir = '../resources/{}/'.format(dname)
    if not os.path.exists(odir):
        os.mkdir(odir)
    odir = odir + 'user2vec/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    main(data_name=dname, encode_directory=encode_dir, odirectory=odir)
