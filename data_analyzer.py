"""
Script to analyze data
"""
import json
import os
from collections import Counter
import numpy as np


def data_stats(corpus_path, opath):
    results = dict()
    # document info
    results['num_doc'] = 0
    results['num_tokens'] = 0
    results['num_vocab'] = set()
    results['token_per_doc'] = 0
    results['max_token_per_doc'] = 0
    results['min_token_per_doc'] = float('inf')
    results['median_token_per_doc'] = []

    # user info
    results['num_user'] = 0
    results['token_per_user'] = 0
    results['ratio_male'] = 0
    results['ratio_female'] = 0
    results['average_age'] = []

    # tag info
    results['num_unique_tag'] = 0
    results['num_tag'] = 0
    results['tag_stats'] = []

    # concept info
    concepts = dict()
    concepts['num_concept'] = 0
    concepts['num_unique_concept'] = 0
    concepts['num_unique_concept_type'] = 0
    concepts['concept_type_stats'] = []
    concepts['concept_token_stats'] = []

    with open(corpus_path) as dfile:
        for line in dfile:
            if len(line) < 10:
                continue
            results['num_user'] += 1
            user = json.loads(line)
            results['num_doc'] += len(user['docs'])

            results['num_tag'] += len(user['tags'])
            results['tag_stats'].extend(user['tags'])

            if user['gender'] == 'F':
                results['ratio_female'] += 1
            else:
                results['ratio_male'] += 1
            if user['age'] != 'x':
                user_age = float(user['age'])
                if 10 < user_age < 120:
                    results['average_age'].append(user_age)

            for doc_entity in user['docs']:
                tokens = doc_entity['text'].split()
                results['num_tokens'] += len(tokens)
                results['num_vocab'].update(tokens)
                results['median_token_per_doc'].append(len(tokens))
                if len(tokens) > results['max_token_per_doc']:
                    results['max_token_per_doc'] = len(tokens)
                if len(tokens) < results['min_token_per_doc']:
                    results['min_token_per_doc'] = len(tokens)

                for concept in doc_entity['concepts']:
                    concepts['num_concept'] += 1
                    concepts['concept_token_stats'].append(concept['preferred_name'])
                    concepts['concept_type_stats'].append(concept['semtypes'])

    concepts['num_unique_concept'] = len(set(concepts['concept_token_stats']))
    concepts['num_unique_concept_type'] = len(set(concepts['concept_type_stats']))
    concepts['concept_token_stats'] = Counter(concepts['concept_token_stats']).most_common()
    concepts['concept_type_stats'] = Counter(concepts['concept_type_stats']).most_common()
    json.dump(concepts, open(os.path.splitext(opath)[0] + '_concept.json', 'w'), indent=4)

    results['ratio_male'] /= results['num_user']
    results['ratio_male'] = round(results['ratio_male'], 2)
    results['ratio_female'] /= results['num_user']
    results['ratio_female'] = round(results['ratio_female'], 2)
    results['num_vocab'] = len(results['num_vocab'])

    results['token_per_user'] = results['num_tokens'] / results['num_user']
    results['token_per_user'] = round(results['token_per_user'], 2)

    results['token_per_doc'] = results['num_tokens'] / results['num_doc']
    results['token_per_doc'] = round(results['token_per_doc'], 2)
    results['median_token_per_doc'] = np.median(results['median_token_per_doc'])

    results['average_age'] = sorted(results['average_age'], reverse=True)
    results['age_stats'] = [
        results['average_age'][-1], np.percentile(results['average_age'], 25),
        np.percentile(results['average_age'], 50), np.percentile(results['average_age'], 75),
        results['average_age'][0]
    ]
    results['age_stats'] = [round(item, 2) for item in results['age_stats']]
    results['average_age'] = np.mean(results['average_age'])
    results['average_age'] = round(results['average_age'], 2)

    results['tag_stats'] = Counter(results['tag_stats']).most_common()
    results['num_unique_tag'] = len(results['tag_stats'])

    json.dump(results, open(opath, 'w'), indent=4)


if __name__ == '__main__':
    dlist = ['diabetes', 'mimic-iii']
    odir = './resources/analyze/'
    indir = './data/processed_data/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    # generate data stats for each dataset
    for dname in dlist:
        data_path = indir + dname + '/{}.json'.format(dname)
        output_path = odir + '{}_stats.json'.format(dname)
        data_stats(data_path, output_path)
