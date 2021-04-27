import argparse
import os
import json
import pickle
import sys

from nltk.tokenize import RegexpTokenizer
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm
import gensim
from keras.preprocessing.sequence import pad_sequences
# load data
# load concepts
# import model
# train model


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
        if idx == steps-1 and len(doc[idx: max_len * (idx + 1)]) < max_len:
            docs.append(' '.join(doc[max_len * -1:]))  # to fill the last piece with full length
        else:
            docs.append(' '.join(doc[max_len * idx: max_len * (idx + 1)]))

    return docs


# because concepts are not well processed.
def concept_preprocessor(concepts):
    # separate each concept by comma, because each concept
    # filter word less or equal than 3 characters
    tkn = RegexpTokenizer(r'[\w-]+')
    pass


def build_concept_tokenizer(concepts):
    # this will help build concept embeddings

    pass


def data_loader(**kwargs):
    user_stats_path = kwargs['user_stats_path']  # to encode users into indices
    concept_dir = kwargs['concept_dir']
    data_path = kwargs['data_path']
    concept_tkn_path = kwargs['concept_tkn_path']
    word_tkn_path = kwargs['word_tkn_path']
    bert_tkn = BertTokenizer.from_pretrained(kwargs['bert_name'])

    # load dataset
    with open(data_path) as dfile:
        for line in dfile:
            user_entity = json.loads(line)

    # load concepts

    pass


def main(params):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--method', type=str, help='GRU or BERT')
    parser.add_argument('--dname', type=str, help='The data\'s name')
    parser.add_argument('--lr', type=float, help='Learning rate', default=.0001)
    parser.add_argument('--ng_num', type=int, help='Number of negative samples', default=1)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('--max_len', type=int, help='Max length', default=512)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimensions', default=300)
    parser.add_argument('--device', type=str, help='cpu or cuda')
    args = parser.parse_args()

    if args.method not in ['gru2user', 'bert2user']:
        print('Method {} is not supported.'.format(args.method))
        sys.exit()

    data_dir = './data/processed_data/{}/'.format(args.dname)
    odir = './resources/embedding/{}/'.format(args.dname)
    if not os.path.exists(odir):
        os.mkdir(odir)
    odir = odir + '{}/'.format(args.method)
    if not os.path.exists(odir):
        os.mkdir(odir)

    if args.device == 'cpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for cpu usage
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    parameters = {
        'batch_size': 32,
        'user_size': -1,
        'word_emb_path': './resources/embedding/{}/word_emb.npy'.format(args.dname),
        'user_emb_path': './resources/embedding/{}/user_emb.npy'.format(args.dname),
        'user_task_weight': 1,
        'concept_task_weight': 1,
        'epochs': 15,
        'optimizer': 'adam',
        'lr': args.lr,
        'negative_sample': args.ng_num,
        'max_len': args.max_len,
        # 'emilyalsentzer/Bio_ClinicalBERT'
        # 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
        # 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12'
        'bert_name': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        'decay_rate': .9,
        'warm_steps': 33,
        'emb_dim': args.emb_dim,
        'dp_rate': .2,
        'dname': args.dname,
        'encode_dir': data_dir,
        'odir': odir,
        'device': args.device
    }
    pass
