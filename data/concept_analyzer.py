import os
import json
from multiprocessing import Pool

import numpy as np

def concept_stats(corpus_dir, opath):
    # concept info
    concepts = dict()
    concepts['num_concept'] = 0
    concepts['num_unique_concept'] = 0
    concepts['num_unique_concept_type'] = 0
    concepts['concept_type_stats'] = []
    concepts['concept_token_stats'] = []
    
    # TODO
    for concept in doc_entity['concepts']:
        concepts['num_concept'] += 1
        concepts['concept_token_stats'].append(concept['preferred_name'])
        concepts['concept_type_stats'].extend(concept['semtypes'])
    
    concepts['num_unique_concept'] = len(set(concepts['concept_token_stats']))
    concepts['num_unique_concept_type'] = len(set(concepts['concept_type_stats']))
    concepts['concept_token_stats'] = Counter(concepts['concept_token_stats']).most_common()
    concepts['concept_type_stats'] = Counter(concepts['concept_type_stats']).most_common()
    json.dump(concepts, open(opath, 'w'), indent=4)


if __name__ == '__main__':
    dlist = ['diabetes', ]  # 'mimic-iii'
    odir = '../resources/analyze/'
    indir = './processed_data/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    # generate data stats for each dataset
    for dname in dlist:
        data_path = indir + dname + '/{}.json'.format(dname)
        output_path = odir + '{}_stats.json'.format(dname)
        concept_stats(data_path, output_path)
        
