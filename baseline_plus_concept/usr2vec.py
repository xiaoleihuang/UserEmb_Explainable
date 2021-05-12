import json
import os
import pickle
import sys
import itertools

from tqdm import tqdm
import keras
import numpy as np
from keras_preprocessing.text import Tokenizer
import gensim
from gensim.models.doc2vec import Doc2Vec
# from keras.preprocessing.sequence import pad_sequences
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for cpu usage
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from baseline_utils import user_word_sampler, data_loader


def user_doc_concept_builder(
        user_docs, all_docs, word_tkn, concept_tkn, user_encoder, negative_samples=1, output_dir=''):
    """This function was re-implemented from the Silvio Amir
    https://github.com/samiroid/usr2vec/tree/master/code

        user_docs (dict): user docs
        sequence (list): a sequence of word indices
        emb_dim (int): document size
    """
    uids = list(user_docs.keys())
    # np.random.shuffle(uids)
    user_doc_concepts = []
    user_doc_labels = []

    for uidx, uid in tqdm(enumerate(uids)):
        user_concepts = list(itertools.chain.from_iterable(user_docs[uid]['concepts']))
        user_concepts_set = set(user_concepts)
        concept_negative_space = [
            concept_tkn[concept] for concept in concept_tkn if concept not in user_concepts_set]
        user_concepts = [concept_tkn[concept] for concept in user_concepts if concept in concept_tkn]

        for doc_id in user_docs[uid]['docs']:
            doc = all_docs[doc_id]
            # doc = word_tkn.texts_to_sequences([doc])[0]
            doc_couples, doc_labels = user_word_sampler(
                uid=user_encoder[uid], sequence=doc,
                tokenizer=word_tkn, negative_samples=negative_samples
            )

            for idx in range(len(doc_labels)):
                if doc_labels[idx] == 1:
                    doc_couples[idx].append(np.random.choice(user_concepts))
                else:  # negative samples
                    doc_couples[idx].append(np.random.choice(concept_negative_space))
            user_doc_concepts.extend(doc_couples)
            user_doc_labels.extend(doc_labels)

    user_doc_concepts = np.asarray(user_doc_concepts, dtype=object)
    user_doc_labels = np.asarray(user_doc_labels)

    tmp = {
        'user_doc_concepts': user_doc_concepts,
        'user_doc_labels': user_doc_labels,
    }
    with open(output_dir + 'usr2vec_user_docs.pkl', 'wb') as wfile:
        pickle.dump(tmp, wfile)
    del tmp

    return user_doc_concepts, user_doc_labels


def user_doc_generator(user_words_concepts, doc_labels, batch_size):
    # shuffle the dataset
    rand_indices = list(range(len(doc_labels)))
    np.random.shuffle(rand_indices)
    user_words_concepts = user_words_concepts[rand_indices]
    doc_labels = doc_labels[rand_indices]

    steps = len(doc_labels) // batch_size
    if len(doc_labels) % batch_size != 0:
        steps += 1

    for idx in range(steps):
        yield user_words_concepts[batch_size * idx: batch_size * (idx + 1)], \
              doc_labels[batch_size * idx: batch_size * (idx + 1)]


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
            'max_len': 512,
        }

    # User Input
    user_input = keras.layers.Input((1,), name='user_input', dtype='int32')
    word_input = keras.layers.Input((1,), name='word_input', dtype='int32')
    concept_input = keras.layers.Input((1,), name='concept_input', dtype='int32')

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

    if os.path.exists(params['concept_emb_path']):
        weights = np.load(params['concept_emb_path'])
        concept_emb = keras.layers.Embedding(
            weights.shape[0], weights.shape[1],
            weights=[weights],
            trainable=True, name='concept_emb',
            input_length=1,
        )
    else:
        concept_emb = keras.layers.Embedding(
            params['concept_size'], params['emb_dim'],
            trainable=True, name='concept_emb',
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
    concept_rep = concept_emb(concept_input)

    if word_emb.output_dim != params['emb_dim']:
        word_rep = keras.layers.Dense(
            params['emb_dim'], activation='sigmoid', name='word_transform'
        )(word_rep)
        concept_rep = keras.layers.Dense(
            params['emb_dim'], activation='sigmoid', name='concept_transform'
        )(concept_rep)

    user_word_dot = keras.layers.dot(
        [user_rep, word_rep], axes=-1
    )
    user_word_dot = keras.layers.Reshape((1,))(user_word_dot)
    user_pred = keras.layers.Dense(
        1, activation='sigmoid', name='user_pred'
    )(user_word_dot)

    user_concept_dot = keras.layers.dot(
        [user_rep, concept_rep], axes=-1
    )
    user_concept_dot = keras.layers.Reshape((1,))(user_concept_dot)
    concept_pred = keras.layers.Dense(
        1, activation='sigmoid', name='concept_pred'
    )(user_concept_dot)

    '''Compose model'''
    if params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(lr=params['lr'])
    else:
        optimizer = keras.optimizers.SGD(
            lr=params['lr'], decay=1e-6, momentum=0.9, nesterov=True)

    # user word model
    ud_model = keras.models.Model(
        inputs=[user_input, word_input, concept_input],
        outputs=[user_pred, concept_pred]
    )
    ud_model.compile(
        loss='binary_crossentropy', optimizer=optimizer,
        loss_weights={
            'concept_pred': params['concept_task_weight'],
            'user_pred': params['word_task_weight']
        }
    )
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
        emb_model = np.zeros((emb_len, len(line.strip().split())-1))  # first one is a word not vector.
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
        'batch_size': 512,  # to accelerate training speed
        'vocab_size': tok.num_words,
        'user_size': -1,  # +1 for unknown
        'emb_dim': 300,
        'dp_rate': .2,
        'data_path': '../resources/embedding/{}/caue_gru/user_docs_concepts.pkl'.format(data_name),
        'emb_path': '/data/models/BioWordVec_PubMed_MIMICIII_d200.vec.bin',
        'concept_dir': encode_directory + 'concepts/',
        'concept_tkn': encode_directory + 'concept_tkn.pkl',
        'word_emb_path': '../resources/embedding/{}/word_emb.npy'.format(data_name),
        'user_emb_path': '../resources/embedding/{}/user_emb.npy'.format(data_name),
        'concept_emb_path': '../resources/embedding/{}/caue_gru/{}_concept_emb.npy'.format(data_name, data_name),
        'word_emb_train': False,
        'user_emb_train': True,
        'word_task_weight': 1,
        'concept_task_weight': .3,
        'epochs': 10,
        'optimizer': 'adam',
        'lr': 3e-5,
        'negative_sample': 3,
        'max_len': 512,
    }

    # load user encoder, which convert users into indices
    concept_tkn = pickle.load(open(params['concept_tkn'], 'rb'))
    # params['concept_size'] = len(concept_tkn)
    user_encoder = json.load(open(encode_directory + 'user_encoder.json'))
    # update user information
    params['user_size'] = len(user_encoder) + 1

    if not os.path.exists(odirectory + 'usr2vec_user_docs.pkl'):
        user_docs, all_docs = data_loader(params['data_path'])  # omit all documents
        print('Tokenizing Documents...')
        all_docs = tok.texts_to_sequences(all_docs)
        # build datasets
        user_words_concepts, doc_labels = user_doc_concept_builder(
            user_docs, all_docs, tok, concept_tkn, user_encoder,
            negative_samples=params['negative_sample'], output_dir=odirectory
        )
    else:
        tmp = pickle.load(open(odirectory + 'usr2vec_user_docs.pkl', 'rb'))
        user_words_concepts = tmp['user_doc_concepts']
        doc_labels = tmp['user_doc_labels']

    if not os.path.exists('../resources/embedding/{}/'.format(data_name)):
        os.mkdir('../resources/embedding/{}/'.format(data_name))

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
            user_words_concepts, doc_labels, params['batch_size']
        )
        total_steps = len(doc_labels) // params['batch_size']

        for step, train_batch in tqdm(enumerate(train_iter), total=total_steps):
            '''user info, uw: user-word'''
            ud_pairs, ud_labels = train_batch
            ud_pairs = [np.array(x, dtype=np.int32) for x in zip(*ud_pairs)]
            ud_labels = np.array(ud_labels, dtype=np.int32)

            '''Train'''
            train_loss = ud_model.train_on_batch(
                x=ud_pairs,
                # share labels
                y={
                    'user_pred': ud_labels,
                    'concept_pred': ud_labels
                }
            )
            loss += train_loss[0]

            loss_avg = loss / (step + 1)
            if (step+1) % 100 == 0:
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
    odir = odir + 'user2vec_concept/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    main(data_name=dname, encode_directory=encode_dir, odirectory=odir)
