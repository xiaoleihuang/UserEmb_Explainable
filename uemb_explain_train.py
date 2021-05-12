import argparse
import os
import json
import pickle
import sys
import datetime
import itertools

from keras_preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
import numpy as np
from tqdm import tqdm
import gensim
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from uemb_explain_model import build_gru_model, CAUEgru, CAUEBert


# because some documents can be extremely long
def split_docs(doc, max_len=512):
    """ let's split the document if its length is more than 512 token
    :return:
    """
    if type(doc) == str:
        doc = doc.split()
    max_len -= 2  # for the two special tokens, start and end

    docs = []
    steps = len(doc) // max_len
    if len(doc) % max_len != 0:
        steps += 1

    for idx in range(steps):
        if idx == steps - 1 and len(doc[idx: max_len * (idx + 1)]) < max_len:
            docs.append(' '.join(doc[max_len * -1:]))  # to fill the last piece with full length
        else:
            docs.append(' '.join(doc[max_len * idx: max_len * (idx + 1)]))

    return docs


# because concepts are not well processed.
def concept_preprocessor(concepts):
    # separate each concept by comma, because each concept
    # filter word less or equal than 3 characters
    tkn = RegexpTokenizer(r'[\w-]+')
    results = []

    for concept in concepts:
        items = concept.lower().replace('(', ' ').replace(')', ' ').split(',')
        if len(items) > 3:
            continue

        # only keep tokens at least three characters
        items = [item for item in tkn.tokenize(concept) if len(item) > 2 and not item.isnumeric()]
        if len(items) == 0:
            continue

        # use phrases
        results.append(' '.join(items))
        # use words
        # results.extend(items)
    return results


def build_concept_weights(params):
    emb_path = params['emb_path']  # pretrained word embedding path
    concept_tkn = pickle.load(open(params['concept_tkn_path'], 'rb'))

    # derived by pretrained embeddings
    # support three types, bin/txt/npy
    emb_len = len(concept_tkn)
    vector_size = -1
    if emb_path.endswith('.bin'):
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            emb_path, binary=True
        )
        vector_size = w2v_model.vector_size

    elif emb_path.endswith('.txt'):
        w2v_model = {}

        with open(emb_path) as dfile:
            for line in dfile:
                line = line.strip()
                if len(line) < 10:
                    continue
                line = line.split()
                word = line[0]
                vectors = np.asarray(line[1:], dtype='float32')
                if vector_size == -1:
                    vector_size = len(vectors)
                w2v_model[word] = vectors
    else:
        raise ValueError('Current other formats are not supported!')

    emb_model = np.zeros((emb_len, vector_size))
    for idx, concept in enumerate(list(concept_tkn.keys())):
        tokens = concept.split()
        vector = [w2v_model[token] for token in tokens if token in w2v_model]
        if len(vector) > 1:
            emb_model[idx] = np.mean(vector, axis=0)
        elif len(vector) == 1:
            emb_model[idx] = vector[0]
            del concept_tkn[concept]

    # save the extracted embedding weights
    np.save(params['concept_emb_path'], emb_model)
    # if len(concept_tkn) != len(emb_model):
    #     pickle.dump(concept_tkn, open(params['concept_tkn_path'], 'wb'))


def build_emb_weights(tokenizer, emb_path, save_path):
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


def data_builder(**kwargs):
    output_dir = kwargs['odir']
    if os.path.exists(output_dir + 'user_docs_concepts.pkl'):
        loaded_data = pickle.load(open(output_dir + 'user_docs_concepts.pkl', 'rb'))
        return loaded_data[0], loaded_data[1]  # user_corpus, all_docs
    else:
        user_corpus = dict()
        all_docs = []
        concepts_info = dict()  # will store concept count and index pair.

        user_stats_path = kwargs['user_stats_path']  # to encode users into indices
        user_encoder = json.load(open(user_stats_path))
        concept_files = os.listdir(kwargs['concept_dir'])

        # load dataset
        with open(kwargs['data_dir'] + '{}.json'.format(kwargs['dname'])) as dfile:
            for line in tqdm(dfile):
                user_entity = json.loads(line)

                if user_entity['uid'] not in user_corpus:
                    user_corpus[user_entity['uid']] = {
                        'uidx': user_encoder[user_entity['uid']],
                        'docs': [],
                        'concepts': [],
                    }

                uid = user_entity['uid'].split('-')[0]
                for doc_entity in user_entity['docs']:
                    snippets = split_docs(
                        doc_entity['text'], max_len=kwargs['max_len'])
                    for snippet in snippets:
                        all_docs.append(snippet)
                        # record the snippet index
                        user_corpus[user_entity['uid']]['docs'].append(len(all_docs) - 1)
                        user_corpus[user_entity['uid']]['concepts'].append([])

                    did = doc_entity['doc_id']
                    concept_fname = '{}_{}.pkl'.format(uid, did)

                    if concept_fname in concept_files:
                        concepts = pickle.load(open(kwargs['concept_dir'] + concept_fname, 'rb'))
                        # filter out low confident medical concepts
                        concepts = [item['preferred_name'].lower() for item in concepts if float(item['score']) > 3.6]
                        # concepts = [item['preferred_name'] for item in concepts]
                        concepts = concept_preprocessor(concepts)
                        for concept in concepts:
                            if concept not in concepts_info:
                                concepts_info[concept] = 0
                            concepts_info[concept] += 1

                        user_corpus[user_entity['uid']]['concepts'].append(concepts)
                    else:
                        user_corpus[user_entity['uid']]['concepts'].append([])

        # sort the concepts info, by its count
        concept_tkn = list(sorted(
            concepts_info.items(),
            key=lambda item: item[1],
            reverse=True
        ))[:kwargs['vocab_size']]
        # map concepts to indices
        concept_tkn = dict(zip(
            [item[0] for item in concept_tkn],
            range(len(concept_tkn))
        ))
        with open(kwargs['concept_tkn_path'], 'wb') as wfile:
            pickle.dump(concept_tkn, wfile)
        build_concept_weights(kwargs)

        if not os.path.exists(kwargs['word_tkn_path']):
            # 15000 known + 1 unknown tokens
            # default value for this work is 15000
            keras_tkn = Tokenizer(num_words=kwargs['vocab_size'] + 1)
            keras_tkn.fit_on_texts(all_docs)
            pickle.dump(keras_tkn, open(kwargs['word_tkn_path'], 'wb'))
        else:
            keras_tkn = pickle.load(open(kwargs['word_tkn_path'], 'rb'))
        build_emb_weights(
            keras_tkn, kwargs['emb_path'], kwargs['word_emb_path'])

        with open(output_dir + 'user_docs_concepts.pkl', 'wb') as wfile:
            pickle.dump([user_corpus, all_docs], wfile)
        return user_corpus, all_docs


def user_doc_builder(user_docs, all_docs, params):
    max_len = params['max_len']
    concept_tkn = pickle.load(open(params['concept_tkn_path'], 'rb'))
    params['concept_size'] = len(concept_tkn)
    user_encoder = json.load(open(params['user_stats_path']))

    if params['method'] == 'caue_gru':
        tokenizer = pickle.load(open(params['word_tkn_path'], 'rb'))
    else:
        tokenizer = BertTokenizer.from_pretrained(params['bert_name'])

    if params['method'] == 'caue_gru':
        # GRU tokenizer
        all_docs = pad_sequences(
            tokenizer.texts_to_sequences(all_docs),
            maxlen=params['max_len']
        )
        if not params['use_keras']:
            all_docs = torch.tensor(all_docs)
    else:
        # BERT tokenizer
        all_docs = [tokenizer.encode_plus(
            doc_item, padding='max_length', max_length=max_len,
            return_tensors='pt', return_token_type_ids=False,
            truncation=True,
        )['input_ids'][0] for doc_item in all_docs]

    process = tqdm(list(user_docs.keys()))
    uids_docs = []
    docs = []
    uids_concepts = []
    concepts = []
    ud_labels = []
    uc_labels = []

    # loop through each user
    for uid in process:
        sample_doc_space = [idx for idx in range(len(all_docs)) if idx not in user_docs[uid]['docs']]
        user_concepts = set(list(itertools.chain.from_iterable(user_docs[uid]['concepts'])))
        sample_concept_space = [key for key in concept_tkn if key not in user_concepts]

        for step, doc_idx in enumerate(user_docs[uid]['docs']):
            # documents
            docs.append(all_docs[doc_idx])
            ud_labels.append(1)
            uids_docs.append(user_encoder[uid])

            # concepts
            if len(user_docs[uid]['concepts'][step]) == 0:
                continue

            user_docs[uid]['concepts'][step] = [
                concept for concept in user_docs[uid]['concepts'][step] if concept in concept_tkn]
            if len(user_docs[uid]['concepts'][step]) > params['concept_sample_size']:
                select_concepts = np.random.choice(
                    user_docs[uid]['concepts'][step],
                    size=params['concept_sample_size'], replace=False,
                )
            else:
                select_concepts = user_docs[uid]['concepts'][step]

            concepts.extend(select_concepts)
            uc_labels.extend([1] * len(select_concepts))
            uids_concepts.extend([user_encoder[uid]] * len(select_concepts))

            # generate negative samples for concepts
            sample_concepts = np.random.choice(
                sample_concept_space, size=params['negative_sample'] * len(select_concepts), replace=False
            )
            concepts.extend(sample_concepts)
            uc_labels.extend([0] * len(sample_concepts))
            uids_concepts.extend([user_encoder[uid]] * len(sample_concepts))

        sample_docs = np.random.choice(
            sample_doc_space, size=params['negative_sample'] * len(user_docs[uid]['docs']), replace=False
        )
        docs.extend([all_docs[doc_idx] for doc_idx in sample_docs])
        ud_labels.extend([0] * len(sample_docs))
        uids_docs.extend([user_encoder[uid]] * len(sample_docs))

    # encode the concepts into indices
    if params['method'] == 'caue_gru':
        concepts = [
            concept_tkn[concept] for concept in concepts if concept in concept_tkn
        ]
    else:
        concepts = [
            tokenizer.encode_plus(
                concept, padding='max_length', max_length=5,
                return_tensors='pt', return_token_type_ids=False,
                truncation=True,
            )['input_ids'][0] for concept in concepts
        ]

    # if use torch version, we have to convert them into tensors
    if not params['use_keras']:
        uids_docs = torch.tensor(uids_docs)
        # docs = torch.tensor(docs, dtype=torch.long)
        docs = torch.stack(docs)
        ud_labels = torch.tensor(ud_labels, dtype=torch.float)

        if params['method'] == 'caue_gru':
            concepts = torch.tensor(concepts)
        else:
            concepts = torch.stack(concepts)
        uids_concepts = torch.tensor(uids_concepts)
        uc_labels = torch.tensor(uc_labels, dtype=torch.float)

    return uids_docs, docs, ud_labels, uids_concepts, concepts, uc_labels


def user_doc_generator(uids_docs, docs, ud_labels, uids_concepts, concepts, uc_labels, params):
    concept_batch_size = int(params['batch_size'] * (len(concepts) / len(docs)))
    # shuffle the dataset
    if not params['use_keras']:
        rand_indices_doc = torch.randperm(ud_labels.shape[0])
        rand_indices_concept = torch.randperm(uc_labels.shape[0])
    else:
        rand_indices_doc = np.asarray(list(range(len(ud_labels))))
        rand_indices_concept = np.asarray(list(range(len(uc_labels))))
        np.random.shuffle(rand_indices_doc)
        np.random.shuffle(rand_indices_concept)

        uids_docs = np.asarray(uids_docs)
        docs = np.asarray(docs)
        ud_labels = np.asarray(ud_labels)
        uids_concepts = np.asarray(uids_concepts)
        concepts = np.asarray(concepts)
        uc_labels = np.asarray(uc_labels)

    uids_docs = uids_docs[rand_indices_doc]
    docs = docs[rand_indices_doc]
    ud_labels = ud_labels[rand_indices_doc]
    uids_concepts = uids_concepts[rand_indices_concept]
    concepts = concepts[rand_indices_concept]
    uc_labels = uc_labels[rand_indices_concept]

    steps = len(ud_labels) // params['batch_size']
    if len(ud_labels) % params['batch_size'] != 0:
        steps += 1

    for idx in range(steps):
        yield uids_docs[params['batch_size'] * idx: params['batch_size'] * (idx + 1)], \
              docs[params['batch_size'] * idx: params['batch_size'] * (idx + 1)], \
              ud_labels[params['batch_size'] * idx: params['batch_size'] * (idx + 1)], \
              uids_concepts[concept_batch_size * idx: concept_batch_size * (idx + 1)], \
              concepts[concept_batch_size * idx: concept_batch_size * (idx + 1)], \
              uc_labels[concept_batch_size * idx: concept_batch_size * (idx + 1)]


def main(params):
    log_dir = params['odir'] + 'log/'
    writer = SummaryWriter(log_dir=log_dir)
    record_name = datetime.datetime.now().strftime('%H:%M:%S %m-%d-%Y')
    device = torch.device(params['device'])
    params['user_size'] = len(
        json.load(open(params['user_stats_path']))
    )

    print('Loading Dataset...')
    user_corpus, all_docs = data_builder(**params)
    print('Building Dataset...')
    uids_docs, docs, ud_labels, uids_concepts, concepts, uc_labels = user_doc_builder(user_corpus, all_docs, params)
    print(params)

    print('Building models...')
    if params['method'] == 'caue_gru':
        if params['use_keras']:
            caue_model = build_gru_model(params)
        else:
            caue_model = CAUEgru(params)
    else:
        caue_model = CAUEBert(params)

    # training settings
    if params['method'] == 'caue_gru' and params['use_keras']:  # Keras Implementation
        optimizer = None
        scheduler = None
        criterion = None
    else:  # Pytorch Implementation
        criterion = nn.BCEWithLogitsLoss().to(device)
        if params['method'] == 'caue_gru':
            optimizer = torch.optim.RMSprop(caue_model.parameters(), lr=params['lr'])
            # optimizer = torch.optim.Adam(caue_model.parameters(), lr=params['lr'])
            scheduler = None  # no needs to adjust lr for the rmsprop
        else:
            optimize_parameters = [
                {'params': [p for n, p in caue_model.named_parameters() if ('bert_model' not in n) and p.requires_grad],
                 'weight_decay_rate': params['decay_rate']},
                {'params': [p for n, p in caue_model.named_parameters() if 'bert_model' in n or (not p.requires_grad)],
                 'weight_decay_rate': 0.0}
            ]
            optimizer = AdamW(optimize_parameters, lr=params['lr'])
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=params['warm_steps'],
                num_training_steps=(len(ud_labels) // params['batch_size'] + 1)*params['epochs']
            )
        if 'cuda' in params['device']:
            caue_model.cuda()

    print('Starting to train...')
    for epoch in range(params['epochs']):
        print('Epoch: {} '.format(epoch))
        train_loss = 0
        if not params['use_keras']:
            caue_model.train()

        train_iter = user_doc_generator(
            uids_docs=uids_docs, docs=docs, ud_labels=ud_labels,
            uids_concepts=uids_concepts, concepts=concepts, uc_labels=uc_labels,
            params=params
        )

        for step, train_batch in enumerate(tqdm(train_iter)):
            '''Train'''
            uids_docs_batch, docs_batch, ud_labels_batch, uids_concepts_batch, concepts_batch, uc_labels_batch = \
                train_batch
            # training for the keras
            if params['use_keras'] and params['method'] == 'caue_gru':
                loss_doc = caue_model.train_on_batch(
                    x={
                        'user_doc_input': uids_docs_batch,
                        'doc_input': docs_batch,
                        'user_concept_input': uids_concepts_batch[:len(ud_labels_batch)],
                        'concept_input': concepts_batch[:len(ud_labels_batch)]
                    },
                    y={
                        'user_doc_pred': ud_labels_batch,
                        'user_concept_pred': uc_labels_batch[:len(ud_labels_batch)],
                    },
                )
                train_loss += loss_doc[0]

                # if len(uids_concepts_batch) != len(ud_labels_batch):
                #     step_size = len(ud_labels_batch) // len(uc_labels_batch)
                #
                #     for batch_step in range(step_size):
                #         loss_doc = caue_model.train_on_batch(
                #             x={
                #                 'user_doc_input': uids_docs_batch,
                #                 'doc_input': docs_batch,
                #                 'user_concept_input': uids_concepts_batch[
                #                     batch_step * len(ud_labels_batch): (batch_step+1) * len(ud_labels_batch)],
                #                 'concept_input': concepts_batch[
                #                     batch_step * len(ud_labels_batch): (batch_step+1) * len(ud_labels_batch)]
                #             },
                #             y={
                #                 'user_doc_pred': ud_labels_batch,
                #                 'user_concept_pred': uc_labels_batch[
                #                     batch_step * len(ud_labels_batch): (batch_step+1) * len(ud_labels_batch)],
                #             },
                #         )
                #         print(loss_doc)
                #         train_loss += loss_doc[0]
                # else:
                #     loss_doc = caue_model.train_on_batch(
                #         x={
                #             'user_doc_input': uids_docs_batch,
                #             'doc_input': docs_batch,
                #             'user_concept_input': uids_concepts_batch,
                #             'concept_input': concepts_batch
                #         },
                #         y={
                #             'user_doc_pred': ud_labels_batch,
                #             'user_concept_pred': uc_labels_batch,
                #         },
                #     )
                #     train_loss += loss_doc[0]
            else:
                uids_docs_batch = uids_docs_batch.to(device)
                docs_batch = docs_batch.to(device).long()
                ud_labels_batch = ud_labels_batch.to(device)
                uids_concepts_batch = uids_concepts_batch.to(device)
                concepts_batch = concepts_batch.to(device).long()
                uc_labels_batch = uc_labels_batch.to(device)

                if torch.any(torch.isnan(uids_docs_batch)) or torch.any(torch.isinf(uids_docs_batch)):
                    print('invalid input detected at iteration ', step)
                    continue
                if torch.any(torch.isnan(docs_batch)) or torch.any(torch.isinf(docs_batch)):
                    print('invalid input detected at iteration ', step)
                    continue
                if torch.any(torch.isnan(ud_labels_batch)) or torch.any(torch.isinf(ud_labels_batch)):
                    print('invalid input detected at iteration ', step)
                    continue
                if torch.any(torch.isnan(uids_concepts_batch)) or torch.any(torch.isinf(uids_concepts_batch)):
                    print('invalid input detected at iteration ', step)
                    continue
                if torch.any(torch.isnan(concepts_batch)) or torch.any(torch.isinf(concepts_batch)):
                    print('invalid input detected at iteration ', step)
                    continue
                if torch.any(torch.isnan(uc_labels_batch)) or torch.any(torch.isinf(uc_labels_batch)):
                    print('invalid input detected at iteration ', step)
                    continue

                optimizer.zero_grad()
                output_doc, output_concept = caue_model(**{
                    'input_uids4doc': uids_docs_batch,
                    'input_doc_ids': docs_batch,
                    'input_uids4concept': uids_concepts_batch,
                    'input_concept_ids': concepts_batch
                })
                loss_doc = criterion(output_doc, ud_labels_batch)
                if params['use_concept']:
                    loss_concept = criterion(output_concept, uc_labels_batch)

                    # print('Doc Prediction Loss: ', loss_doc.item())
                    # print('Concept Prediction Loss: ', loss_concept.item())
                    loss = loss_doc * params['doc_task_weight'] + loss_concept * params['concept_task_weight'] * \
                        (len(ud_labels_batch) / len(uc_labels_batch))
                else:
                    loss = loss_doc * params['doc_task_weight']
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(caue_model.parameters(), 0.5)
                optimizer.step()

                if scheduler:  # this only applies for the BERT model
                    scheduler.step()

            train_loss_avg = train_loss / (step + 1)
            writer.add_scalar(
                'Loss/train - {}'.format(record_name),
                train_loss_avg,
                step + (len(uids_docs_batch) // params['batch_size']) * epoch
            )
            if (step+1) % 100 == 0:
                print('Epoch: {}, Step: {}'.format(epoch, step))
                print('\t Loss: {}.'.format(train_loss_avg))
                print('-------------------------------------------------')

        # save the user embedding and the model
        if params['use_keras'] and params['method'] == 'caue_gru':
            caue_model.save(params['odir'] + '{}.model'.format(params['method']))
            np.save(
                params['odir'] + 'user_{}.npy'.format(epoch),
                caue_model.get_layer(name='user_emb').get_weights()[0]
            )
        else:
            torch.save(caue_model, params['odir'] + '{}.pth'.format(params['method']))
            np.save(
                params['odir'] + 'user_{}.npy'.format(epoch),
                caue_model.uemb.weight.cpu().detach().numpy()
            )
    writer.close()


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Parameters
    ----------
    v

    Returns
    -------

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--method', type=str, help='caue_gru or caue_bert')
    parser.add_argument('--dname', type=str, help='The data\'s name')
    parser.add_argument('--use_concept', type=str2bool, help='If use concept as additional features')
    parser.add_argument('--use_keras', type=bool, help='If use keras implementation for the GRU method', default=False)
    parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
    parser.add_argument('--ng_num', type=int, help='Number of negative samples', default=3)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--max_len', type=int, help='Max length', default=512)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimensions', default=300)
    parser.add_argument('--device', type=str, help='cpu or cuda')
    args = parser.parse_args()

    if args.method not in ['caue_gru', 'caue_bert']:
        print('Method {} is not supported.'.format(args.method))
        sys.exit()

    data_dir = './data/processed_data/{}/'.format(args.dname)
    odir = './resources/embedding/{}/'.format(args.dname)
    if not os.path.exists(odir):
        os.mkdir(odir)

    if args.use_concept:
        odir = odir + '{}/'.format(args.method)
    else:
        odir = odir + '{}_no/'.format(args.method)
    if not os.path.exists(odir):
        os.mkdir(odir)

    if args.device == 'cpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for cpu usage
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parameters = {
        'method': args.method,
        'batch_size': 16,
        'user_size': -1,
        'dname': args.dname,
        'data_dir': data_dir,
        'odir': odir,
        'user_stats_path': data_dir + 'user_encoder.json',
        'concept_dir': data_dir + 'concepts/',
        'emb_path': '/data/models/BioWordVec_PubMed_MIMICIII_d200.vec.bin',
        'word_emb_train': False,
        'word_emb_path': odir + 'word_emb.npy'.format(args.dname),
        'user_emb_path': odir + 'user_emb.npy'.format(args.dname),
        'user_emb_train': True,
        'concept_emb_path': odir + '{}_concept_emb.npy'.format(args.dname),
        'doc_task_weight': 1,
        'concept_task_weight': .03 if args.use_concept else 0,
        'epochs': 15,
        'optimizer': 'adam',
        'lr': args.lr,
        'negative_sample': args.ng_num,
        'max_len': args.max_len,
        # 'emilyalsentzer/Bio_ClinicalBERT'
        # 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
        # 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12'
        # '/data/models/mimiciii_roberta_10e_128b'
        'bert_name': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        'decay_rate': .9,
        'warm_steps': 33,
        'bidirectional': True,
        'emb_dim': args.emb_dim,
        'dp_rate': .2,
        'device': args.device,
        'vocab_size': 15000,
        'concept_tkn_path': data_dir + 'concept_tkn.pkl',
        'word_tkn_path': data_dir + 'word_tkn.pkl',
        'concept_sample_size': 33,  # to sample the number per document for training, prevent too many
        'use_concept': args.use_concept,
        'use_keras': args.use_keras,
    }
    main(parameters)
