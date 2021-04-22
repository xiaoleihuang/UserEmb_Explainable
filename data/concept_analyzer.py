import os
import json
from collections import Counter
from multiprocessing import Pool

import numpy as np


def read_user(doc_names):

    with open()
    pass

def concept_stats(corpus_dir, odir, task_name):
    # number of medical documents
    flist = os.listdir(corpus_dir)
    user_flist = dict()
    for fname in flist:
        uid = fname.split('_')[0]
        if uid not in user_flist:
            user_flist[uid] = []
        user_flist[uid].append(fname)

    # concept info
    concepts = dict()  # count on user level
    general_info = dict()  # general information

    general_info['num_concept'] = 0
    general_info['num_unique_concept'] = 0
    general_info['num_unique_concept_type'] = 0
    general_info['concept_type_stats'] = []
    general_info['concept_token_stats'] = []



    # TODO
    for concept in doc_entity['concepts']:
        concepts['num_concept'] += 1
        concepts['concept_token_stats'].append(concept['preferred_name'])
        concepts['concept_type_stats'].extend(concept['semtypes'])
    
    general_info['num_unique_concept'] = len(set(general_info['concept_token_stats']))
    general_info['num_unique_concept_type'] = len(set(general_info['concept_type_stats']))
    general_info['concept_token_stats'] = Counter(general_info['concept_token_stats']).most_common()
    general_info['concept_type_stats'] = Counter(general_info['concept_type_stats']).most_common()
    json.dump(general_info, open(opath, 'w'), indent=4)


if __name__ == '__main__':
    dlist = ['diabetes', ]  # 'mimic-iii'
    odir = '../resources/analyze/'
    indir = './processed_data/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    # generate data stats for each dataset
    for dname in dlist:
        data_path = indir + dname + '/{}.json'.format(dname)
        output_dir = odir + '{}/'.format(dname)
        concept_stats(data_path, output_dir, dname)
