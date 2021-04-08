# this script is to build datasets
# icd code group was adapted from:
# https://github.com/YerevaNN/mimic3-benchmarks/blob/master/mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml

import json
import os
import heapq
from collections import Counter
import re
from dateutil.parser import parse
import xmltodict
import pickle
import yaml

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from pymetamap import MetaMap


def sigmoid(value):
    return 1 / (1 + np.exp(-1 * value))


''' Like the word2vec, sampling words depend on its frequency
    Therefore, we need to find a method to rank:
        1. movie = f(vote/max_vote) + f(review_score/10)
        2. restaurant = f(review_count/max_count) + f(star/5)
        3. amazon product = f(review_count/max_count) + f(avg_score/5)
        4. user = f(num_review/max_review)
'''


def rank_bid(review_count, score, max_count, base):
    """calculate business popularity
        review_count: the number of reviews
        score: average review score
        max_count: max review counts in this category, to normalize the review count
        base: the score base, 10 for movie, 5 for yelp and amazon
    """
    return sigmoid(review_count / max_count) + sigmoid(score / base)


def preprocess(doc, stopwords=None, min_len=None):
    """Split, tokenize documents
        stopwords (set)
    """
    # lowercase
    if stopwords is None:
        stopwords = set()

    # replace url
    doc = re.sub(r"https?:\S+", "url", doc)

    doc = doc.replace('\n', ' ')
    doc = doc.replace('\t', ' ')

    # replace anonymized entities, numeric id
    doc = re.sub(r"\[\*\*Numeric.*?\*\*\]", "num_id", doc)
    # replace and extract anonymized entities, known lastname
    # doc = re.sub(r"\[\*\*Known lastname.*?(\d+)\*\*\]", r"lastname\1", doc)
    doc = re.sub(r"\[\*\*Known lastname.*?(\d+)\*\*\]", "lastname", doc)
    doc = re.sub(r"\[\*\*Last Name.*?\*\*\]", "lastname", doc)

    # replace anonymized entities, firstname
    doc = re.sub(r"\[\*\*Known firstname.*?(\d+)\*\*\]", "firstname", doc)
    doc = re.sub(r"\[\*\*First Name.*?\*\*\]", "firstname", doc)

    # remove anonymized entities, MD
    doc = re.sub(r"\[\*\*MD.*?\*\*\]", "", doc)
    # replace anonymized entities, numeric id
    doc = re.sub(r"\[\*\*Hospital.*?\*\*\]", "hospital", doc)

    # replace date
    doc = re.sub(r"(\d+)+(\-)+(\d+)+(\-)+(\d+)", "date", doc)
    # remove all serialization eg 1. 1) or 1.1
    doc = re.sub(r"(\d+)+(\.|\))+(\d+)", "", doc)
    doc = re.sub(r"(\d+)+(\.|\))", "", doc)

    doc = re.sub(r"\b(\w+)( \1\b)+", r"\1", doc)  # removing consecutive duplicate words
    doc = re.sub(r"(\b)(d|[dD])\.?(r|[rR])\.?(\b)", " ", doc)  # remove Dr abbreviation
    doc = re.sub(r"([^A-Za-z0-9\s](\s)){2,}", " ", doc)  # remove consecutive punctuations

    # ellipsis normalization
    doc = re.sub(r'\.+', '.', doc)
    doc = re.sub(r'!+', '!', doc)
    doc = re.sub(r'\*+', ' ', doc)
    doc = re.sub(r'_+', ' ', doc)
    doc = re.sub(r',+', ',', doc)

    doc = doc.lower()
    doc = [item.strip() for item in word_tokenize(doc)
           if len(item.strip()) > 1 and item not in stopwords
           ]  # tokenize

    if min_len and len(doc) < min_len:
        return 'x'
    else:
        return ' '.join(doc)


def process_concepts(entities):

    # TODO
    pass


def format_time(time):
    tdate = parse(time)
    return tdate.strftime('%m-%d-%Y')


def simple_gender_clf(all_tokens):
    """ Classify if the patient is a male or female by feminine and masculine words

    :param all_tokens: a list of tokens
    :return:
    """
    feminine_words = {'her', 'she', 'hers', 'lady', 'woman'}
    masculine_words = {'him', 'he', 'man', 'gentleman', 'hers'}

    counts = Counter(all_tokens)
    feminine_count = sum([counts[token] for token in feminine_words])
    masculine_count = sum([counts[token] for token in masculine_words])

    if feminine_count > masculine_count:
        return 'F'
    else:
        return 'M'


def build_tokenizer(dname, indir, odir):
    """

    :param dname: data name
    :param indir: input data directory
    :param odir:
    :return:
    """
    if not os.path.exists(odir):
        os.mkdir(odir)
    odir = os.path.join(odir, dname)
    if not os.path.exists(odir):
        os.mkdir(odir)
    opath = odir + '/' + dname + '.tkn'

    if os.path.exists(opath):
        return pickle.load(open(opath, 'rb'))
    else:
        tok = Tokenizer(num_words=20001)  # 20000 known + 1 unknown tokens
        corpus = []

        with open(os.path.join(indir, dname) + '/' + dname + '.json') as dfile:
            for line in dfile:
                user = json.loads(line)
                for doc_entity in user['docs']:
                    corpus.append(doc_entity['text'])
        tok.fit_on_texts(corpus)
        with open(opath, 'wb') as wfile:
            pickle.dump(tok, wfile)
        return tok


def process_amazon(indir, odir):
    """ Extract the amazon health data according to our needs

    :param indir:
    :param odir:
    :return:
    """
    fname = 'Health_and_Personal_Care_5.json'
    meta_fname = 'meta_' + fname

    '''find the top 4 categories'''
    # load bid2genre
    #    bid2genre = dict()
    #    with open(indir + meta_fname) as dfile:
    #        for line in dfile:
    #            entity = json.loads(line)
    #            if 'asin' not in entity:
    #                continue

    #            if 'categories' not in entity or len(entity['categories']) == 0 \
    #                or len(entity['categories'][0]) == 0:
    #                continue

    #            for genre in entity['categories'][0]:
    #                if genre not in bid2genre:
    #                    bid2genre[genre.strip()] = set()
    #                bid2genre[genre.strip()].add(entity['asin'])

    if not os.path.exists(odir):
        os.mkdir(odir)

    # define genre names, genre value must match file name
    products_info = dict()
    products_filters = {
        'Vitamins & Dietary Supplements': set(),
        'Sexual Wellness': set(),
        'Shaving & Hair Removal': set(),
        'Sports Nutrition': set()
    }
    users_info = dict()
    top_tokens = Counter()

    '''Collect bids to filters'''
    with open(indir + meta_fname) as dfile:
        for line in dfile:
            entity = json.loads(line)
            if 'asin' not in entity:
                continue
            if 'categories' not in entity or len(entity['categories']) == 0 \
                    or len(entity['categories'][0]) == 0:
                continue

            for genre in entity['categories'][0]:
                if genre in products_filters:
                    products_filters[genre].add(entity['asin'])

    # sample
    #    min_val = min([len(item) for item in products_filters])
    #    for genre in products_filters:
    #        if len(products_filters[genre]) == min_val:
    #            continue
    #        products_filters[genre] = set(np.random.choice(list(products_filters[genre]), size=min_val))

    '''Collect Product and User information'''
    print('Collecting product and user information in each genre...')
    print('Working on: ', indir + fname)
    with open(indir + fname) as dfile:
        for line in dfile:
            entity = json.loads(line.strip())

            # check uid, bid, review text
            if len(entity['reviewText']) < 10:
                continue
            if len(entity['reviewerID']) < 3:
                continue
            if len(entity['asin']) < 3:
                continue

            genres = []
            for genre in products_filters:
                if entity['asin'] in products_filters[genre]:
                    genres.append(genre)
            if len(genres) == 0:
                continue

            entity['reviewText'] = preprocess(entity['reviewText'])
            if not entity['reviewText']:
                continue

            # count tokens
            for token in entity['reviewText'].split():
                if not token.isalpha():
                    continue
                top_tokens[token] += 1

            # User info
            if entity['reviewerID'] not in users_info:
                users_info[entity['reviewerID']] = dict()
                users_info[entity['reviewerID']]['review_count'] = 0
                users_info[entity['reviewerID']]['words'] = set()
                users_info[entity['reviewerID']]['bids'] = set()
            users_info[entity['reviewerID']]['review_count'] += 1

            # Product info
            if entity['asin'] not in products_info:
                products_info[entity['asin']] = dict()
                products_info[entity['asin']]['star'] = entity['overall']
                products_info[entity['asin']]['review_count'] = 1
                products_info[entity['asin']]['words'] = set()
                products_info[entity['asin']]['uids'] = set()
                products_info[entity['asin']]['genre'] = genres
            else:
                products_info[entity['asin']]['star'] = \
                    products_info[entity['asin']]['star'] * \
                    products_info[entity['asin']]['review_count'] + \
                    entity['overall']
                products_info[entity['asin']]['review_count'] += 1
                products_info[entity['asin']]['star'] /= \
                    products_info[entity['asin']]['review_count']

    top_tokens = dict(
        top_tokens.most_common(15)  # top 10 frequentist tokens (letter only)
    )

    '''Filter out the product and user less than the require number'''
    print('Filter user and sample products')
    for uid in list(users_info):
        if users_info[uid]['review_count'] < 5:
            del users_info[uid]

    max_count = dict()
    for bid in list(products_info):
        if products_info[bid]['review_count'] < 10:
            del products_info[bid]
            continue

        for genre in products_info[bid]['genre']:
            if genre not in max_count:
                max_count[genre] = 0

            if products_info[bid]['review_count'] > max_count[genre]:
                max_count[genre] = products_info[bid]['review_count']

    for bid in products_info:
        products_info[bid]['popularity'] = rank_bid(
            products_info[bid]['review_count'],
            products_info[bid]['star'],
            max_count[genre], 5.0
        )

    '''Build the review'''
    print('Building review data...')
    # review file
    rfile = open(odir + 'amazon_health.tsv', 'w')
    columns = ['rid', 'bid', 'uid', 'text', 'date', 'genre', 'label']
    rfile.write('\t'.join(columns) + '\n')

    with open(indir + fname) as dfile:
        for line in dfile:
            entity = json.loads(line)

            '''Filters'''
            # user control
            if entity['reviewerID'] not in users_info:
                continue

            # filter out the categories
            if entity['asin'] not in products_info:
                continue

            # filter out text less than 10 tokens
            entity['reviewText'] = preprocess(entity['reviewText'])
            if not entity['reviewText']:
                continue

            '''Data collection'''
            # encode labels
            if entity['overall'] > 3:
                entity['overall'] = 2
            elif entity['overall'] < 3:
                entity['overall'] = 0
            else:
                entity['overall'] = 1

            # collect review data
            line = '\t'.join([
                entity['reviewerID'] + '#' + str(entity['unixReviewTime']), entity['asin'],
                entity['reviewerID'], entity['reviewText'], format_time(entity['reviewTime']),
                ','.join(products_info[entity['asin']]['genre']), str(entity['overall'])
            ])
            rfile.write(line + '\n')

            # collect words for both products and users
            for token in entity['reviewText'].split():
                if not token.isalpha():
                    continue

                if token in top_tokens:
                    continue

                products_info[entity['asin']]['words'].add(token)
                users_info[entity['reviewerID']]['words'].add(token)

            # collect purchasing behaviors
            products_info[entity['asin']]['uids'].add(entity['reviewerID'])
            users_info[entity['reviewerID']]['bids'].add(entity['asin'])

    rfile.flush()
    rfile.close()

    '''save user and product information'''
    print('Saving user information...')
    user_idx = list()
    product_idx = list()
    with open(odir + 'users.json', 'w') as wfile:
        for uid in users_info:
            if len(users_info[uid]['words']) == 0:
                continue

            users_info[uid]['uid'] = uid
            users_info[uid]['words'] = list(users_info[uid]['words'])
            users_info[uid]['bids'] = list(users_info[uid]['bids'])

            wfile.write(json.dumps(users_info[uid]) + '\n')
            heapq.heappush(user_idx, (users_info[uid]['review_count'], uid))
    user_idx_encoder = dict()  # a dictionary for user idx mapping
    init_idx = len(user_idx)  # 0 is the reserved idx for unknown
    while init_idx > 0:
        item = heapq.heappop(user_idx)
        user_idx_encoder[item[1]] = init_idx
        init_idx -= 1
    with open(odir + 'user_idx.json', 'w') as wfile:
        wfile.write(json.dumps(user_idx_encoder))

    print('Saving product information...')
    with open(odir + 'products.json', 'w') as wfile:
        for bid in products_info:
            if len(products_info[bid]['words']) == 0:
                continue

            products_info[bid]['bid'] = bid
            products_info[bid]['words'] = list(products_info[bid]['words'])
            products_info[bid]['uids'] = list(products_info[bid]['uids'])
            wfile.write(json.dumps(products_info[bid]) + '\n')
            heapq.heappush(product_idx, (products_info[bid]['popularity'], bid))
    product_idx_encoder = dict()  # a dictionary for product idx mapping
    init_idx = len(product_idx)  # 0 is the reserved idx for unknown
    while init_idx > 0:
        item = heapq.heappop(product_idx)
        product_idx_encoder[item[1]] = init_idx
        init_idx -= 1
    with open(odir + 'product_idx.json', 'w') as wfile:
        wfile.write(json.dumps(product_idx_encoder))


def process_diabetes(indir, odir):
    """ Extract the diabetes data according to our needs

    :param indir:
    :param odir:
    :return:
    """
    file_list = [fname for fname in os.listdir(indir) if fname != '.DS_Store']
    opath = os.path.join(odir, 'diabetes.json')
    wfile = open(opath, 'w')

    # extract age information
    user_age = dict()
    with open('./resources/diabetes_age.csv') as dfile:
        cols = dfile.readline().strip().split()
        user_idx = cols.index('uid')
        age_idx = cols.index('age')

        for line in dfile:
            line = line.strip().split()
            if len(line) != len(cols):
                continue
            user_age[line[user_idx]] = line[age_idx] if line[age_idx] != '-1' else 'x'

    for fname in file_list:
        print('Working on: ', fname)
        result = dict()
        result['uid'] = fname.split('.')[0]
        all_user_tokens = []
        fpath = os.path.join(indir, fname)

        # convert the xml to json dictionary
        dfile = xmltodict.parse(open(fpath).read())

        # read the tags of the patient
        result['tags'] = []
        result['tags_set'] = set()
        for tag in dfile['PatientMatching']['TAGS']:
            # filter out some tags that have low Kappa-agreement < 50%
            if tag in ['ENGLISH', 'MAKES-DECISIONS', 'ADVANCED-CAD']:
                continue

            if dfile['PatientMatching']['TAGS'][tag]['@met'] == 'met':
                result['tags'].append(tag.lower())
                result['tags_set'].add(tag.lower())
        result['tags_set'] = list(result['tags_set'])

        # parse every diagnosis report
        result['docs'] = []
        separator = '*' * 100  # each report was separated by 100 stars
        did = 0  # doc id

        for snippet in dfile['PatientMatching']['TEXT'].split(separator):
            snippet = snippet.strip()
            if len(snippet) < 10:
                continue
            result['docs'].append(dict())

            # extract date
            try:
                snippet = re.split('^Record [d|D]ate: ([^\s]+)*', snippet)[1:]
                result['docs'][-1]['date'] = snippet[0]
            except IOError or OSError:
                print(snippet)
            snippet = snippet[1]

            # remove the footnote
            snippet = snippet.split('___________________________________')[0].strip('-').strip()
            snippet = snippet.split('\n')
            collection = []
            for idx in range(len(snippet)):
                line = ' '.join([token.strip() for token in snippet[idx].split() if len(token.strip()) > 0])
                if len(line) < 5:
                    continue
                if idx < 5:
                    if len(line) < 30:
                        continue
                if 'VISIT DATE' in snippet[idx]:
                    continue
                line = preprocess(line)
                collection.append(line)
            result['docs'][-1]['text'] = ' '.join(collection)

            # filter out empty records
            if len(result['docs'][-1]['text'].split()) < 10:
                result['docs'].pop(-1)
                continue

            all_user_tokens.extend(result['docs'][-1]['text'].split())
            result['docs'][-1]['tags'] = result['tags']
            result['docs'][-1]['doc_id'] = str(did)
            result['docs'][-1]['concepts'] = list()  # TODO, concepts from metamap
            did += 1

        # filter out empty patients
        if len(result['docs']) < 1:
            continue

        # classify the gender by token counts
        result['gender'] = simple_gender_clf(all_user_tokens)
        result['age'] = user_age[result['uid']]
        wfile.write(json.dumps(result) + '\n')


def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code


def ethnicity_encode_mimic(ethnicity):
    """This code is adopted from:
    https://github.com/YerevaNN/mimic3-benchmarks/blob/master/mimic3benchmark/preprocessing.py

    :param ethnicity:
    :return:
    """
    e_map = {'ASIAN': 1,
             'BLACK': 2,
             'CARIBBEAN ISLAND': 2,
             'HISPANIC': 3,
             'SOUTH AMERICAN': 3,
             'WHITE': 4,
             'MIDDLE EASTERN': 4,
             'PORTUGUESE': 4,
             'AMERICAN INDIAN': 0,
             'NATIVE HAWAIIAN': 0,
             'UNABLE TO OBTAIN': 0,
             'PATIENT DECLINED TO ANSWER': 0,
             'UNKNOWN': 0,
             'OTHER': 0,
             '': 0}
    ethnicity = ethnicity.replace(' OR ', '/').split(' - ')[0].split('/')[0]
    return e_map[ethnicity] if ethnicity in e_map else e_map['OTHER']


def process_mimic(indir, odir):
    """

    :param indir:
    :param odir:
    :return:
    """
    mm = MetaMap.get_instance('/data/xiaolei/public_mm/bin/metamap')
    results = dict()
    # progressively load the note event
    notes = pd.read_csv(
        indir + 'NOTEEVENTS.csv', dtype=str,
        usecols=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY', 'TEXT']
    )
    notes.CHARTDATE = pd.to_datetime(notes.CHARTDATE)

    # filter out none discharge summary
    # similar to https://github.com/jamesmullenbach/caml-mimic/blob/master/notebooks/dataproc_mimic_III.ipynb
    # notes = notes[notes.CATEGORY == 'Discharge summary']  # extract all types of documents
    notes.SUBJECT_ID = notes.SUBJECT_ID.apply(lambda x: x.strip())
    # filter out patients who have less than two notes
    patient_counts = Counter(notes.SUBJECT_ID)
    patient_counts = set([item[0] for item in patient_counts.items()])
    notes = notes[notes.SUBJECT_ID.isin(patient_counts)]
    # filter out patients whoever generates less than 3 notes per stay
    counts = Counter(notes.HADM_ID)
    counts = dict([item for item in counts.items() if item[1] > 2])
    notes = notes[notes.HADM_ID.isin(counts)]
    notes.fillna('x', inplace=True)
    # preprocess the note documents, filter out documents less than 50 tokens
    notes.TEXT = notes.TEXT.apply(lambda x: preprocess(x, min_len=50))
    notes = notes[notes.TEXT != 'x']

    # load patient table
    patient_set = set(notes.SUBJECT_ID)
    patients = pd.read_csv(
        indir + 'PATIENTS.csv', dtype=str,
        usecols=['SUBJECT_ID', 'GENDER', 'DOB']
    )
    patients.SUBJECT_ID = patients.SUBJECT_ID.apply(lambda x: x.strip())
    patients = patients[patients.SUBJECT_ID.isin(patient_set)]
    patients.DOB = pd.to_datetime(patients.DOB)
    patients.fillna('x', inplace=True)
    # convert to a dictionary for fast search, the first value is id, the 2nd value is the timestamp
    patients = dict((z[0], list(z[1:])) for z in zip(patients.SUBJECT_ID, patients.GENDER, patients.DOB))

    # get admission table, aim for the ethnicity information
    admits = pd.read_csv(
        indir + 'ADMISSIONS.csv', dtype=str,
        usecols=['SUBJECT_ID', 'ETHNICITY']
    )
    admits.ETHNICITY = admits.ETHNICITY.fillna('OTHER')  # replace N/A values
    # admits = admits.dropna(subset=['SUBJECT_ID', 'ETHNICITY'])
    admits.ETHNICITY = admits.ETHNICITY.apply(lambda x: ethnicity_encode_mimic(x.strip()))
    admits.fillna('x', inplace=True)
    admits = dict(zip(admits.SUBJECT_ID, admits.ETHNICITY))

    # load icd codes
    icd_encoder = dict()
    dfile = yaml.load(open('./resources/hcup_ccs_2015_definitions.yaml'), Loader=yaml.FullLoader)
    for tmp_key in dfile:
        for tmp_code in dfile[tmp_key]['codes']:
            icd_encoder[tmp_code] = tmp_key

    dfcodes = dict()
    hadm_set = set(notes.HADM_ID)
    with open(indir + 'DIAGNOSES_ICD.csv') as dfile:
        cols = [col.replace('"', '').strip() for col in dfile.readline().strip().split(',')]
        icd_idx = cols.index('ICD9_CODE')
        subj_idx = cols.index('SUBJECT_ID')
        hadm_idx = cols.index('HADM_ID')

        for line in dfile:
            line = [item.replace('"', '').strip() for item in line.strip().split(',')]
            if len(line) != len(cols):
                continue
            if line[subj_idx] not in patient_set:
                continue
            if line[hadm_idx] not in hadm_set:
                continue
            code_id = '{0}-{1}'.format(line[subj_idx], line[hadm_idx])
            if code_id not in dfcodes:
                dfcodes[code_id] = list()
            dfcodes[code_id].append(icd_encoder.get(line[icd_idx].strip()))

    # loop through each note
    for index, row in notes.iterrows():
        uid = row['SUBJECT_ID'] + '-' + row['HADM_ID']
        if uid not in results:
            u_age = (row['CHARTDATE'].to_pydatetime() - patients[row['SUBJECT_ID']][1].to_pydatetime()
                     ).total_seconds() / 3600 / 24 / 365
            if u_age < 18:  # filter out patients younger than 18
                continue

            results[uid] = {
                'uid': uid,
                # calculate from the current stay and patient's DOB
                'age': u_age,
                'gender': patients[row['SUBJECT_ID']][0],  # first value is the gender
                'ethnicity': admits[row['SUBJECT_ID']],  # ethnicity
                'tags_set': set(),  # convert to list in the end, unique tags
                'tags': list(),  # collect all patient tags
                'docs': list(),  # collect all patient notes
            }

        # filter out empty records
        if len(row['TEXT'].split()) < 10:
            continue

        concepts = mm.extract_concepts([row['TEXT']],  word_sense_disambiguation=True)
        results[uid]['docs'].append({
            'doc_id': row['ROW_ID'],
            'date': row['CHARTDATE'].strftime('%Y-%m-%d'),
            'text': row['TEXT'],
            'tags': dfcodes[uid],
            'concepts': [],  # TODO
        })
        results[uid]['tags_set'].update(dfcodes[uid])
        results[uid]['tags'].extend(dfcodes[uid])

    opath = os.path.join(odir, 'mimic-iii.json')
    with open(opath, 'w') as wfile:
        for uid in results:
            # filter out empty records
            if len(results[uid]['docs']) < 1:
                continue
            results[uid]['tags_set'] = list(results[uid]['tags_set'])
            wfile.write(json.dumps(results[uid]) + '\n')


if __name__ == '__main__':
    # flist = ['amazon', 'diabetes', 'mimic']
    output_dir = './data/processed_data/'

    # amazon health dataset
    # amazon_indir = './data/raw_data/amazon/'
    # if not os.path.exists(output_dir + 'amazon/'):
    #     os.mkdir(output_dir + 'amazon/')
    # process_amazon(amazon_indir, output_dir + 'amazon/')

    # diabetes
    diabetes_indir = './data/raw_data/diabetes/all/'
    if not os.path.exists(output_dir + 'diabetes/'):
        os.mkdir(output_dir + 'diabetes/')
    process_diabetes(diabetes_indir, output_dir + 'diabetes/')

    # mimic-iii
    mimic_indir = '/data/xiaolei/physionet.org/files/mimiciii/1.4/'
    if not os.path.exists(output_dir + 'mimic-iii/'):
        os.mkdir(output_dir + 'mimic-iii/')
    process_mimic(mimic_indir, output_dir + 'mimic-iii/')
