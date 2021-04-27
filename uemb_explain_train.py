import argparse
import os
import json
import pickle

from nltk.tokenize import RegexpTokenizer
from transformers import BertTokenizer
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


def build_concept_encoder(concepts):
    # this will help build concept embeddings

    pass


def data_loader(**kwargs):
    user_stats_path = kwargs['user_stats_path']  # to encode users into indices
    concept_dir = kwargs['concept_dir']
    data_path = kwargs['data_path']
    concept_tkn_path = kwargs['concept_tkn_path']
    word_tkn_path = kwargs['word_tkn_path']
    bert_tkn = BertTokenizer.from_pretrained(kwargs['bert_name'])

    # load concepts
    pass


def main():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--dname', type=str, help='The data\'s name')
    parser.add_argument('--lr', type=float, help='Learning rate', default=.0001)
    parser.add_argument('--ng_num', type=int, help='Number of negative samples', default=1)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('--max_len', type=int, help='Max length', default=512)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimensions', default=300)
    args = parser.parse_args()


    pass
