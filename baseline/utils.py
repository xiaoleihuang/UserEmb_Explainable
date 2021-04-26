""" user word generator by negative sampling (only 1 negative sample)"""

import numpy as np
import random
import os
import json


def sample_decay(count, decay=2):
    """Calculate decay number for sampling user and product
    """
    return 1 / (1 + decay * count)


def user_word_sampler(uid, sequence, vocab_size, filter_words=None, negative_samples=1):
    """This function was partially adopted from
    https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/sequence.py#L151

        uid (int): a user id index
        sequence (list): a sequence of word indices
        vocab_size (int): word vocabulary size
    """
    couples = []
    labels = []

    for wid in sequence:
        couples.append([uid, wid])
        labels.append(1)

    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)

        for idx in range(num_negative_samples):
            wid = random.randint(1, vocab_size - 1)

            # ensure user did not use the word
            if filter_words:
                while wid in filter_words and len(filter_words) <= vocab_size:
                    wid = random.randint(1, vocab_size - 1)

            couples.append([uid, wid])
            labels.append(0)

    # shuffle
    seed = random.randint(0, int(10e6))
    random.seed(seed)
    random.shuffle(couples)
    random.seed(seed)
    random.shuffle(labels)
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
    pass
    
