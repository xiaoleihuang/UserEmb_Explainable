import json
import os
import pickle
import sys

from tqdm import tqdm
import keras
import numpy as np
from keras_preprocessing.text import Tokenizer
import gensim
from baseline_utils import user_word_sampler

# from keras.preprocessing.sequence import pad_sequences
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for cpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def user_doc_builder(user_docs, tokenizer, negative_samples=1, output_dir=''):
    """This function was re-implemented from the Silvio Amir
    https://github.com/samiroid/usr2vec/tree/master/code

        user_docs (dict): user docs
        sequence (list): a sequence of word indices
        emb_dim (int): document size
    """
    if os.path.exists(output_dir + 'usr2vec_user_docs.pkl'):
        tmp = pickle.load(open(output_dir + 'usr2vec_user_docs.pkl', 'rb'))
        return tmp['couples'], tmp['labels']
    else:
        uids = list(user_docs.keys())
        np.random.shuffle(uids)
        couples = []
        labels = []

        for uid in uids:
            for doc in user_docs[uid]:
                doc_couples, doc_labels = user_word_sampler(
                    uid=uid, sequence=doc, tokenizer=tokenizer, negative_samples=negative_samples
                )
                couples.extend(doc_couples)
                labels.extend(doc_labels)

        couples = np.asarray(couples, dtype=object)
        labels = np.asarray(labels)

        tmp = {
            'couples': couples,
            'labels': labels
        }
        with open(output_dir + 'usr2vec_user_docs.pkl', 'wb') as wfile:
            pickle.dump(tmp, wfile)
        del tmp

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
            'vocab_size': 15001,
            'user_size': 8000,
            'emb_dim': 300,
            'word_emb_path': '../resources/word_emb.npy',
            'user_emb_path': '../resources/user_emb.npy',
            'word_emb_train': False,
            'user_emb_train': True,
            'user_task_weight': 1,
            'word_task_weight': 1,
            'epochs': 10,
            'optimizer': 'adam',
            'lr': 0.00001,
            'max_len': 512,
        }

    # User Input
    user_input = keras.layers.Input((1,), name='user_input', dtype='int32')
    word_input = keras.layers.Input((1,), name='word_input', dtype='int32')

    # load weights if word embedding path is given
    if os.path.exists(params['word_emb_path']):
        weights = np.load(params['word_emb_path'])
        word_emb = keras.layers.Embedding(
            params['vocab_size'], weights.shape[1],
            weights=[weights],
            trainable=params['word_emb_train'], name='word_emb',
            input_length=1,
        )
    else:
        word_emb = keras.layers.Embedding(
            params['vocab_size'], params['emb_dim'],
            trainable=params['word_emb_train'], name='word_emb',
            input_length=1,
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

    '''User Word Dot Production'''
    user_rep = user_emb(user_input)
    word_rep = word_emb(word_input)
    if word_emb.output_dim != params['emb_dim']:
        word_rep = keras.layers.Dense(
            params['emb_dim'], name='dimension_transform'
        )(word_rep)

    user_word_dot = keras.layers.dot(
        [user_rep, word_rep], axes=-1
    )
    user_pred = keras.layers.Reshape((1,))(user_word_dot)
    # remove this because of source code:
    # https://github.com/samiroid/usr2vec/blob/0a4cab05ae8a20915340e6647f74be3d75f633fb/code/usr2vec.py#L60
    # user_pred = keras.layers.Dense(
    #     1, activation='sigmoid', name='user_pred'
    # )(user_word_dot)

    '''Compose model'''
    if params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(lr=params['lr'])
    else:
        optimizer = keras.optimizers.SGD(lr=params['lr'])

    # user word model
    ud_model = keras.models.Model(
        inputs=[user_input, word_input],
        outputs=user_pred
    )
    ud_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    print(ud_model.summary())

    return ud_model


def build_emb_layer(tokenizer, emb_path, save_path):
    """"""
    # support three types, bin/txt/npy
    emb_len = len(tokenizer.word_index)
    if emb_len > tokenizer.num_words:
        emb_len = tokenizer.num_words

    if emb_path.endswith('.bin'):
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            emb_path, binary=True
        )
        emb_model = np.zeros((emb_len, w2v_model.vector_size))
        for pair in zip(w2v_model.index2word, w2v_model.vectors):
            if pair[0] in tokenizer.word_index and \
                    tokenizer.word_index[pair[0]] < tokenizer.num_words:
                emb_model[tokenizer.word_index[pair[0]]] = pair[1]

        # save the extracted embedding weights
        np.save(save_path, emb_model)

    elif emb_path.endswith('.txt'):
        line = open(emb_path).readline()
        emb_model = np.zeros((emb_len, len(line.strip().split()) - 1))  # first one is a word not vector.
        emb_dim = len(line.strip().split()) - 1

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
    # load tokenizer
    if os.path.exists(encode_directory + data_name + '.tkn'):
        tok = pickle.load(open(encode_directory + data_name + '.tkn', 'rb'))
    else:
        all_docs = []
        with open(encode_directory + data_name + '.json') as dfile:
            for idx, line in enumerate(dfile):
                user_info = json.loads(line)
                for doc in user_info['docs']:
                    all_docs.append(doc['text'])

        tok = Tokenizer(num_words=15001)  # 15000 known + 1 unknown tokens
        tok.fit_on_texts(all_docs)
        pickle.dump(tok, open(encode_directory + data_name + '.tkn', 'wb'))
        del all_docs

    params = {
        'batch_size': 512,
        'vocab_size': tok.num_words,
        'user_size': -1,  # +1 for unknown
        'emb_dim': 300,
        'emb_path': '/data/models/BioWordVec_PubMed_MIMICIII_d200.vec.bin',
        'word_emb_path': '../resources/embedding/{}/word_emb.npy'.format(data_name),
        'user_emb_path': '../resources/embedding/{}/user_emb.npy'.format(data_name),
        'word_emb_train': False,
        'user_emb_train': True,
        'user_task_weight': 1,
        'epochs': 10,
        'optimizer': 'sgd',
        'lr': 5e-5,
        'negative_sample': 20,
        'max_len': 512,
    }

    # load user encoder, which convert users into indices
    if os.path.exists(odirectory + 'train_corpus.pkl') and \
            os.path.exists(encode_directory + 'user_encoder.json'):
        user_corpus = pickle.load(open(odirectory + 'train_corpus.pkl', 'rb'))
        user_encoder = json.load(open(encode_directory + 'user_encoder.json'))
    else:
        user_corpus = dict()
        user_encoder = dict()
        with open(encode_directory + data_name + '.json') as dfile:
            for idx, line in enumerate(dfile):
                user_info = json.loads(line)
                user_info['uid'] = str(user_info['uid'])
                if user_info['uid'] not in user_encoder:
                    user_encoder[user_info['uid']] = len(user_encoder)
                user_info['uid'] = user_encoder[user_info['uid']]

                user_corpus[user_info['uid']] = []

                for doc in user_info['docs']:
                    user_corpus[user_info['uid']].append(
                        tok.texts_to_sequences([doc['text']])[0]  # [:params['max_len']]
                    )
        pickle.dump(user_corpus, open(odirectory + 'train_corpus.pkl', 'wb'))
        json.dump(user_encoder, open(encode_directory + 'user_encoder.json', 'w'))

    # update user information
    params['user_size'] = len(user_encoder) + 1
    if not os.path.exists(encode_directory + 'user_encoder.json'):
        json.dump(user_encoder, open(encode_directory + 'user_encoder.json', 'w'))

    if not os.path.exists('../resources/embedding/{}/'.format(data_name)):
        os.mkdir('../resources/embedding/{}/'.format(data_name))

    # build datasets
    user_words, labels = user_doc_builder(
        user_corpus, tok, negative_samples=params['negative_sample'], output_dir=odirectory
    )

    # build embedding model
    if not os.path.exists(params['word_emb_path']):
        build_emb_layer(
            tok, params['emb_path'], params['word_emb_path']
        )

    ud_model = build_model(params)
    print()
    print(params)

    for epoch in tqdm(range(params['epochs'])):
        loss = 0

        train_iter = user_doc_generator(
            user_words, labels, params['batch_size']
        )
        total_steps = len(labels) // params['batch_size']

        for step, train_batch in tqdm(enumerate(train_iter), total=total_steps):
            '''user info, uw: user-word'''
            ud_pairs, ud_labels = train_batch
            ud_pairs = [np.array(x, dtype=np.int32) for x in zip(*ud_pairs)]
            ud_labels = np.array(ud_labels, dtype=np.int32)

            '''Train'''
            loss += ud_model.train_on_batch(ud_pairs, ud_labels)

            loss_avg = loss / (step + 1)
            if step % 100 == 0:
                print('Epoch: {}, Step: {}'.format(epoch, step))
                print('\tLoss: {}.'.format(loss_avg))
                # print('Remaining Steps: ', round((step + 1) / total_steps, 2))
                print('-------------------------------------------------')

    # save the model
    ud_model.save(odirectory + 'ud_model.h5')
    # save the user embedding
    np.save(odirectory + 'user.npy', ud_model.get_layer(name='user_emb').get_weights()[0])


if __name__ == '__main__':
    dname = sys.argv[1]

    encode_dir = '../data/processed_data/'
    encode_dir = encode_dir + dname + '/'

    odir = '../resources/embedding/{}/'.format(dname)
    if not os.path.exists(odir):
        os.mkdir(odir)
    odir = odir + 'user2vec/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    main(data_name=dname, encode_directory=encode_dir, odirectory=odir)
