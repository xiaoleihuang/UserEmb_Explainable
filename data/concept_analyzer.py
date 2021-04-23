import os
import json
from collections import Counter
import pickle

import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def concept_stats(corpus_dir, odir, task_name):
    if not os.path.exists(odir):
        os.mkdir(odir)

    general_info = dict()  # general information
    general_info['num_concept'] = 0
    general_info['num_unique_concept'] = set()
    general_info['num_unique_concept_type'] = set()
    general_info['concept_type_stats'] = []
    general_info['concept_token_stats'] = []

    # this will help encode user concepts during training process
    user_concepts = dict()
    score_filter = True  # to control if we use selected entities

    # number of medical documents
    flist = os.listdir(corpus_dir)

    for fname in flist:
        uid = fname.split('_')[0]
        if uid not in user_concepts:
            # record semantic types and individual tokens
            user_concepts[uid] = {
                'semtypes': [],
                'entities': [],
            }
        concepts = pickle.load(open(corpus_dir + fname, 'rb'))
        # control scores of the named entities
        for concept in concepts:
            if score_filter and concept['score'] < 3.6:
                continue

            user_concepts[uid]['semtypes'].append(concept['semtypes'])
            user_concepts[uid]['entities'].append(concept['preferred_name'].lower())
            general_info['num_concept'] += 1
            general_info['num_unique_concept'].add(concept['preferred_name'].lower())
            general_info['num_unique_concept_type'].add(concept['semtypes'])
            general_info['concept_type_stats'].append(concept['semtypes'])
            general_info['concept_token_stats'].append(concept['preferred_name'].lower())
    
    general_info['num_unique_concept'] = len(general_info['num_unique_concept'])
    general_info['num_unique_concept_type'] = len(general_info['num_unique_concept_type'])
    general_info['concept_token_stats'] = Counter(general_info['concept_token_stats']).most_common()
    general_info['concept_type_stats'] = Counter(general_info['concept_type_stats']).most_common()
    json.dump(general_info, open(odir + 'concept_{}_stats.json'.format(task_name), 'w'), indent=4)
    if score_filter:
        json.dump(user_concepts, open(odir + 'concept_{}_user_filtered.json'.format(task_name), 'w'))
    else:
        json.dump(user_concepts, open(odir + 'concept_{}_user.json'.format(task_name), 'w'))


# this function is to answer the importance of medical concepts from a qualitative perspective
# by the idea of patients with similar disease diagnosis should share similar medical concepts;
# and patients with different disease diagnosis should have different medical concepts;
# therefore this can highlight that incorporating the medical concepts can better understand differences
# between groups of patients. Therefore, this can indicate that incorporating concepts can help better
# learn user/patient presentations.
# From another level, a linear regression model can check the relationship between concept similarity of two patients
# and disease label similarity of two patients
def qual_concepts_sim(corpus_dir, odir, task_name):

    pass


# this function is to explore similar issue, if the medical concepts with users/patients are important;
# by the idea of building disease classification models only use medical entities as features.
# Therefore, we can compare performances of using entities only and using whole documents
# on patient disease classification task;
# Features will be all uni-grams
def quant_concepts_sim(**kwargs):
    corpus_path = kwargs['corpus_path']
    concept_path = kwargs['concept_path']

    vectorizer = TfidfVectorizer(max_features=2000)
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1)
    num_label = 10  # only experiment with top 10 labels


    pass


if __name__ == '__main__':
    dlist = ['diabetes', ]  # 'mimic-iii'
    output_dir = '../resources/analyze/'
    indir = './processed_data/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # generate data stats for each dataset
    for dname in dlist:
        data_path = indir + dname + '/{}.json'.format(dname)
        concept_stats(data_path, output_dir + '{}/'.format(dname), dname)
