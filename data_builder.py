# this script is to build datasets
# icd code group was adapted from:
# https://github.com/YerevaNN/mimic3-benchmarks/blob/master/mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml

import json
import multiprocessing
import os
import heapq
from collections import Counter
import re
from dateutil.parser import parse
import xmltodict
import pickle
import yaml
from multiprocessing import Pool
import sys
import subprocess
import tempfile
import logging

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from pymetamap import Corpus, CorpusLite
from pymetamap import MetaMap
# from pymetamap import MetaMapLite
from keras.preprocessing.text import Tokenizer
from pandarallel import pandarallel
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(process)s %(levelname)s %(message)s',
    filename='../resources/concepts.log',
    filemode='a'
)


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


# reduce line length
def partition(temp_l, n):
    for i in range(0, len(temp_l), n):
        yield ' '.join(temp_l[i:i + n])


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
    doc = re.sub(r'\.+', '', doc)
    doc = re.sub(r'!+', '!', doc)
    doc = re.sub(r'\*+', ' ', doc)
    doc = re.sub(r'_+', ' ', doc)
    doc = re.sub(r',+', ',', doc)

    # special character
    doc = doc.replace('w/', '')

    doc = doc.lower()
    doc = [item.strip() for item in word_tokenize(doc)
           if len(item.strip()) > 1 and item not in stopwords
           ]  # tokenize

    if min_len and len(doc) < min_len:
        return 'x'
    else:
        return ' '.join(doc)


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


# code adapt from https://github.com/AnthonyMRios/pymetamap/blob/master/pymetamap/SubprocessBackendLite.py
def metamaplite_concepts(sentences=None, ids=None,
                         restrict_to_sts=None, restrict_to_sources=None):
    """ extract_concepts takes a list of sentences and ids(optional)
        then returns a list of Concept objects extracted via
        MetaMapLite.
        Supported Options:
            Restrict to Semantic Types --restrict_to_sts
            Restrict to Sources --restrict_to_sources
        For information about the available options visit
        http://metamap.nlm.nih.gov/.
        Note: If an error is encountered the process will be closed
              and whatever was processed, if anything, will be
              returned along with the error found.
    """
    metamap_home = os.environ['METAMAP_HOME']
    if not sentences:
        raise ValueError("You must either pass a list of sentences.")

    input_file = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix='.mmi')

    if sentences is not None:
        if ids is not None:
            for identifier, sentence in zip(ids, sentences):
                input_file.write('{0!r}|{1}\n'.format(identifier, sentence).encode('utf8'))
        else:
            for sentence in sentences:
                input_file.write('{0!r}\n'.format(sentence).encode('utf8'))
        input_file.flush()
        input_file.close()

    command = ["bash", os.path.join(metamap_home, "metamaplite.sh")]
    if restrict_to_sts:
        if isinstance(restrict_to_sts, str):
            restrict_to_sts = [restrict_to_sts]
        if len(restrict_to_sts) > 0:
            command.append('--restrict_to_sts={}'.format(str(','.join(restrict_to_sts))))
            # command.append(str(','.join(restrict_to_sts)))

    if restrict_to_sources:
        if isinstance(restrict_to_sources, str):
            restrict_to_sources = [restrict_to_sources]
        if len(restrict_to_sources) > 0:
            command.append('--restrict_to_sources')
            command.append(str(','.join(restrict_to_sources)))

    if ids is not None:
        command.append('--inputformat=sldiwi')

    command.append(input_file.name)
    command.append('--overwrite')
    command.append('--indexdir={}data/ivf/2020AA/USAbase'.format(metamap_home))
    command.append('--modelsdir={}data/models/'.format(metamap_home))
    command.append('--specialtermsfile={}data/specialterms.txt'.format(metamap_home))
    # command.append(output_file.name)

    output_file_name, file_extension = os.path.splitext(input_file.name)
    output_file_name += "." + "mmi"

    output_file_name, file_extension = os.path.splitext(input_file.name)
    output_file_name += "." + "mmi"

    # output = str(output_file.read())
    metamap_process = subprocess.Popen(command, stdout=subprocess.PIPE)
    while metamap_process.poll() is None:
        stdout = str(metamap_process.stdout.readline())
        if 'ERROR' in stdout:
            metamap_process.terminate()
            print(stdout.rstrip())

    # print("input file name: {0}".format(input_file.name))
    output_file_name, file_extension = os.path.splitext(input_file.name)
    output_file_name += "." + "mmi"
    # print("output_file_name: {0}".format(output_file_name))
    with open(output_file_name) as fd:
        output = fd.read()
    # output = str(output_file.read())
    # print("output: {0}".format(output))
    concepts = CorpusLite.load(output.splitlines())
    return concepts


# code adapt from https://github.com/AnthonyMRios/pymetamap/blob/master/pymetamap/SubprocessBackend.py
def metamap_concepts(sentences=None,
                     ids=None,
                     composite_phrase=4,
                     file_format='sldi',
                     allow_acronym_variants=False,
                     word_sense_disambiguation=False,
                     allow_large_n=False,
                     strict_model=False,
                     relaxed_model=False,
                     allow_overmatches=False,
                     allow_concept_gaps=False,
                     term_processing=False,
                     no_derivational_variants=False,
                     derivational_variants=False,
                     ignore_word_order=False,
                     unique_acronym_variants=False,
                     prefer_multiple_concepts=False,
                     ignore_stop_phrases=False,
                     compute_all_mappings=False,
                     prune=-1,
                     mm_data_version=False,
                     exclude_sources=None,
                     restrict_to_sources=None,
                     restrict_to_sts=None,
                     exclude_sts=None,
                     no_nums=None):
    """ extract_concepts takes a list of sentences and ids(optional)
        then returns a list of Concept objects extracted via
        MetaMap.
        Supported Options:
            Composite Phrase -Q
            Word Sense Disambiguation -y
            use strict model -A
            use relaxed model -C
            allow large N -l
            allow overmatches -o
            allow concept gaps -g
            term processing -z
            No Derivational Variants -d
            All Derivational Variants -D
            Ignore Word Order -i
            Allow Acronym Variants -a
            Unique Acronym Variants -u
            Prefer Multiple Concepts -Y
            Ignore Stop Phrases -K
            Compute All Mappings -b
            MM Data Version -V
            Exclude Sources -e
            Restrict to Sources -R
            Restrict to Semantic Types -J
            Exclude Semantic Types -k
            Suppress Numerical Concepts --no_nums
        For information about the available options visit
        http://metamap.nlm.nih.gov/.
        Note: If an error is encountered the process will be closed
              and whatever was processed, if anything, will be
              returned along with the error found.
    """
    if no_nums is None:
        no_nums = []
    if exclude_sts is None:
        exclude_sts = []
    if restrict_to_sts is None:
        restrict_to_sts = []
    if restrict_to_sources is None:
        restrict_to_sources = []
    if allow_acronym_variants and unique_acronym_variants:
        raise ValueError("You can't use both allow_acronym_variants and unique_acronym_variants.")
    if not sentences:
        raise ValueError("You must either pass a list of sentences.")
    if file_format not in ['sldi', 'sldiID']:
        raise ValueError("file_format must be either sldi or sldiID")

    command = list()
    command.append('metamap')
    command.append('-N')
    command.append('-Q')
    command.append(str(composite_phrase))
    if mm_data_version is not False:
        if mm_data_version not in ['Base', 'USAbase', 'NLM']:
            raise ValueError("mm_data_version must be Base, USAbase, or NLM.")
        command.append('-V')
        command.append(str(mm_data_version))
    if word_sense_disambiguation:
        command.append('-y')
    if strict_model:
        command.append('-A')
    if prune != -1:
        command.append('--prune')
        command.append(str(prune))
    if relaxed_model:
        command.append('-C')
    if allow_large_n:
        command.append('-l')
    if allow_overmatches:
        command.append('-o')
    if allow_concept_gaps:
        command.append('-g')
    if term_processing:
        command.append('-z')
    if no_derivational_variants:
        command.append('-d')
    if derivational_variants:
        command.append('-D')
    if ignore_word_order:
        command.append('-i')
    if allow_acronym_variants:
        command.append('-a')
    if unique_acronym_variants:
        command.append('-u')
    if prefer_multiple_concepts:
        command.append('-Y')
    if ignore_stop_phrases:
        command.append('-K')
    if compute_all_mappings:
        command.append('-b')
    if exclude_sources and len(exclude_sources) > 0:
        command.append('-e')
        command.append(str(','.join(exclude_sources)))
    if restrict_to_sources and len(restrict_to_sources) > 0:
        command.append('-R')
        command.append(str(','.join(restrict_to_sources)))
    if restrict_to_sts and len(restrict_to_sts) > 0:
        command.append('-J')
        command.append(str(','.join(restrict_to_sts)))
    if exclude_sts and len(exclude_sts) > 0:
        command.append('-k')
        command.append(str(','.join(exclude_sts)))
    if no_nums and len(no_nums) > 0:
        command.append('--no_nums')
        command.append(str(','.join(no_nums)))
    if ids is not None or (file_format == 'sldiID' and sentences is None):
        command.append('--sldiID')
    else:
        command.append('--sldi')

    command.append('--silent')

    output = None
    if sentences is not None:
        input_text = None
        if ids is not None:
            for identifier, sentence in zip(ids, sentences):
                if input_text is None:
                    input_text = '{0!r}|{1!r}\n'.format(identifier, sentence).encode('utf8')
                else:
                    input_text += '{0!r}|{1!r}\n'.format(identifier, sentence).encode('utf8')
        else:
            for sentence in sentences:
                if input_text is None:
                    input_text = '{0!r}\n'.format(sentence).encode('utf8')
                else:
                    input_text += '{0!r}\n'.format(sentence).encode('utf8')

        input_command = list()
        input_command.append('echo')
        input_command.append('-e')
        input_command.append(input_text)

        input_process = subprocess.Popen(input_command, stdout=subprocess.PIPE)
        metamap_process = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=input_process.stdout)

        output, error = metamap_process.communicate()
        if sys.version_info[0] > 2:
            if isinstance(output, bytes):
                output = output.decode()

        if metamap_process.returncode == 0:
            output = output.split('\n')
            info_line = 0
            for idx in range(len(output)):
                if len(output[idx].split('|')) > 1 and output[idx].split('|')[1] in ['MMI', 'AA', 'UA']:
                    info_line = idx
                    break
            output = [item for item in output[info_line:] if len(item) > 3]
            output = Corpus.load(output)
        else:
            return None
    return output


def process_concepts(entities):
    results = []
    for entity in entities:
        item = dict()
        try:
            item['semtypes'] = entity.semtypes.lstrip('[').rstrip(']').split(',')
        except AttributeError:
            continue
        # define filter list
        if 'qnco' in item['semtypes']:  # filter out Quantitative Concept: per year
            continue
        if 'tmco' in item['semtypes']:  # filter out Temporal Concept: day
            continue
        if 'hlca' in item['semtypes']:  # Health Care Activity: Hospital admission; Tapering - action;
            continue
        if 'idcn' in item['semtypes']:  # Idea or Concept: Presentation
            continue
        if 'hcro' in item['semtypes']:  # Health Care Related Organization: Accident and Emergency department
            continue
        if 'clna' in item['semtypes']:  # Clinical Attribute: History of present illness
            continue
        if 'ftcn' in item['semtypes']:  # Functional Concept: Negation, Extraocular; Due to
            continue
        if 'qlco' in item['semtypes']:  # Qualitative Concept: Started
            continue
        if 'fndg' in item['semtypes']:  # Finding: Present
            continue
        if 'acty' in item['semtypes']:  # Activity: Obscure; Assessed; Departure - action
            continue
        if 'mnob' in item['semtypes']:  # Manufactured Object: Machine; Beds
            continue
        if 'plnt' in item['semtypes']:  # Plant
            continue
        if 'podg' in item['semtypes']:  # Patient or Disabled Group: Patients
            continue
        if 'popg' in item['semtypes']:  # Population Group
            continue
        if 'prog' in item['semtypes']:  # Professional or Occupational Group: Physicians
            continue
        if 'pros' in item['semtypes']:  # Professional Society
            continue
        if 'elii' in item['semtypes']:  # Element, Ion, or Isotope: lead
            continue
        if 'anim' in item['semtypes']:  # Animal: Show
            continue
        if 'inpr' in item['semtypes']:  # Intellectual Product: Code; Telephone Number
            continue
        if 'orgf' in item['semtypes']:  # Organism Function: Movement; Expiration, function; Inspiration function
            continue
        if 'npop' in item['semtypes']:  # Natural Phenomenon or Process: Saturated
            continue
        if 'lang' in item['semtypes']:  # Language: Herero language
            continue
        if 'spco' in item['semtypes']:  # Spatial Concept: Scattered; Round shape
            continue
        if 'bpoc' in item['semtypes']:  # Body Part, Organ, or Organ Component: Eminence
            continue
        if 'phsf' in item['semtypes']:  # Physiologic Function: Respiration
            continue
        if 'clas' in item['semtypes']:  # Classification: Trial Phase
            continue
        if 'food' in item['semtypes']:  # Food: Food
            continue
        if 'orga' in item['semtypes']:  # Organism Attribute: Body Temperature
            continue
        if 'cnce' in item['semtypes']:  # Conceptual Entity: System Alert
            continue
        if 'ocdi' in item['semtypes']:  # Occupation or Discipline: Science of Chemistry
            continue
        if 'resa' in item['semtypes']:  # Research Activity: Diagnosis Study
            continue
        if 'lbpr' in item['semtypes']:  # Laboratory Procedure: International Normalized Ratio
            continue

        # item['index'] = len(results)
        # item['mm'] = entity.mm
        item['score'] = entity.score
        item['preferred_name'] = entity.preferred_name
        item['cui'] = entity.cui
        # item['trigger'] = entity.trigger.lstrip('[').rstrip(']').split(',')
        # item['pos_info'] = entity.pos_info
        results.append(item)

    return results


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


def process_diabetes_thread(finfo):
    fname, opath, user_age, indir = finfo
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

    for snippet in tqdm(dfile['PatientMatching']['TEXT'].split(separator)):
        result['docs'].append(dict())
        snippet = snippet.strip()

        if len(snippet) < 10:
            continue
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
            all_user_tokens.extend(line.split())
            collection.append(line)

        doc_text = ' '.join(collection)
        # filter out empty records
        if len(doc_text.split()) < 10:
            result['docs'].pop(-1)
            continue
        result['docs'][-1]['text'] = doc_text

        concepts_collection = []
        # parameters documentation: https://metamap.nlm.nih.gov/Docs/MM_2016_Usage.pdf
        # https://metamap.nlm.nih.gov/Docs/README_javaapi.shtml
        step_size = 5
        steps = len(collection) // step_size

        # reduce line length
        if len(collection) % step_size != 0:
            steps += 1
        for step in range(steps):
            try:
                # concepts, error = mm.extract_concepts(
                concepts = metamap_concepts(
                    collection[step * step_size: (step + 1) * step_size], word_sense_disambiguation=True,
                    unique_acronym_variants=True, ignore_stop_phrases=True, no_derivational_variants=True,
                    no_nums=['all'], exclude_sts=[
                        'bpoc', 'spco', 'lang', 'npop', 'orgf', 'qnco', 'tmco', 'hlca', 'idcn', 'hcro', 'clna',
                        'ftcn', 'qlco', 'fndg', 'acty', 'mnob', 'plnt', 'podg', 'popg', 'prog', 'pros', 'elii',
                        'anim', 'inpr'
                    ],
                )
                if concepts:
                    concepts_collection.extend(process_concepts(concepts))
            except IndexError:
                try:
                    tmp_collection = collection[step * step_size: (step + 1) * step_size]
                    partition_size = 100
                    collection_len = len(tmp_collection)
                    for line_idx in range(len(tmp_collection)):
                        if len(tmp_collection[line_idx].split()) > partition_size:
                            tmp_collection.extend(
                                list(partition(tmp_collection[line_idx].split(), partition_size)))
                        else:
                            tmp_collection.append(tmp_collection[line_idx])
                    tmp_collection = tmp_collection[collection_len:]

                    tmp_steps = len(tmp_collection) // step_size
                    if len(tmp_collection) % step_size != 0:
                        tmp_steps += 1

                    for tmp_step in range(tmp_steps):
                        # concepts, error = mm.extract_concepts(
                        concepts = metamap_concepts(
                            tmp_collection[tmp_step * step_size: (tmp_step + 1) * step_size],
                            word_sense_disambiguation=True,
                            unique_acronym_variants=True, ignore_stop_phrases=True, no_derivational_variants=True,
                            no_nums=['all'], exclude_sts=[
                                'bpoc', 'spco', 'lang', 'npop', 'orgf', 'qnco', 'tmco', 'hlca', 'idcn', 'hcro',
                                'clna', 'ftcn', 'qlco', 'fndg', 'acty', 'mnob', 'plnt', 'podg', 'popg', 'prog',
                                'pros', 'elii', 'anim', 'inpr'
                            ], prune=33,
                        )
                        if concepts:
                            concepts_collection.extend(process_concepts(concepts))
                except IndexError:
                    continue

        if len(concepts_collection) > 0:
            with open(
                    os.environ['CONCEPT_ODIR'] + '{}_{}.pkl'.format(result['uid'], did), 'wb'
            ) as cfile:
                pickle.dump(concepts_collection, cfile)

        result['docs'][-1]['tags'] = result['tags']
        result['docs'][-1]['doc_id'] = str(did)
        did += 1

    # filter out empty patients
    if len(result['docs']) < 1:
        return

    # classify the gender by token counts
    result['gender'] = simple_gender_clf(all_user_tokens)
    result['age'] = user_age[result['uid']]
    if len(result['docs'][-1]) == 0:
        result['docs'].pop(-1)
    with open(opath, 'a') as wfile:
        wfile.write(json.dumps(result) + '\n')


def process_diabetes(indir, odir):
    """ Extract the diabetes data according to our needs

    :param indir:
    :param odir:
    :return:
    """
    opath = os.path.join(odir, 'diabetes.json')
    wfile = open(opath, 'w')
    wfile.close()
    if not os.path.exists(os.environ['CONCEPT_ODIR']):
        os.mkdir(os.environ['CONCEPT_ODIR'])

    # load tokenizers and concept extractor
    # mm = MetaMap.get_instance('/data/xiaolei/public_mm/bin/metamap')

    # extract age information
    user_age = dict()
    with open('resources/diabetes_age.csv') as dfile:
        cols = dfile.readline().strip().split()
        user_idx = cols.index('uid')
        age_idx = cols.index('age')

        for line in dfile:
            line = line.strip().split()
            if len(line) != len(cols):
                continue
            user_age[line[user_idx]] = line[age_idx] if line[age_idx] != '-1' else 'x'
    file_list = [(fname, opath, user_age, indir) for fname in os.listdir(indir) if fname != '.DS_Store']
    pool = Pool(os.cpu_count())
    pool.map(process_diabetes_thread, file_list)


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


def get_concept_thread(input_text):
    # parameters documentation: https://metamap.nlm.nih.gov/Docs/MM_2016_Usage.pdf
    # https://metamap.nlm.nih.gov/Docs/README_javaapi.shtml
    row_id, input_text, uid = input_text
    if os.path.exists(os.environ['CONCEPT_ODIR'] + '{}.pkl'.format(row_id)):
        return
    collection = sent_tokenize(input_text)
    step_size = 5
    steps = len(collection) // step_size
    if len(collection) % step_size != 0:
        steps += 1

    concepts_collection = []
    mm = MetaMap.get_instance(os.environ['METAMAP_HOME'])
    # mm = MetaMapLite.get_instance('/data/xiaolei/public_mm_lite/')

    for step in tqdm(range(steps)):
        step_collection = collection[step * step_size: (step + 1) * step_size]
        try:
            # concepts = metamap_concepts(
            concepts, error = mm.extract_concepts(
                sentences=step_collection,
                word_sense_disambiguation=True,
                unique_acronym_variants=True,
                ignore_stop_phrases=True,
                no_derivational_variants=True,
                no_nums=['all'],
                exclude_sts=[
                    'bpoc', 'spco', 'lang', 'npop', 'orgf', 'qnco', 'tmco', 'hlca', 'idcn', 'hcro', 'clna',
                    'ftcn', 'qlco', 'fndg', 'acty', 'mnob', 'plnt', 'podg', 'popg', 'prog', 'pros', 'elii',
                    'anim', 'inpr', 'food'
                ],
                # for metamap lite only
                # restrict_to_sts=[],
            )
            if concepts:
                concepts_collection.extend(process_concepts(concepts))
        except IndexError:
            pass
            # try:
            #     tmp_collection = step_collection
            #     partition_size = 60
            #     collection_len = len(tmp_collection)
            #     for line_idx in range(len(tmp_collection)):
            #         if len(tmp_collection[line_idx].split()) > partition_size:
            #             tmp_collection.extend(list(partition(tmp_collection[line_idx].split(), partition_size)))
            #         else:
            #             tmp_collection.append(tmp_collection[line_idx])
            #     tmp_collection = tmp_collection[collection_len:]
            #
            #     tmp_steps = len(tmp_collection) // step_size
            #     if len(tmp_collection) % step_size != 0:
            #         tmp_steps += 1
            #
            #     for tmp_step in range(tmp_steps):
            #         # concepts = metamap_concepts(
            #         concepts, error = mm.extract_concepts(
            #             sentences=tmp_collection[tmp_step * step_size: (tmp_step + 1) * step_size],
            #             word_sense_disambiguation=True,
            #             unique_acronym_variants=True,
            #             ignore_stop_phrases=True, no_derivational_variants=True,
            #             no_nums=['all'], exclude_sts=[
            #                 'bpoc', 'spco', 'lang', 'npop', 'orgf', 'qnco', 'tmco', 'hlca', 'idcn', 'hcro', 'clna',
            #                 'ftcn', 'qlco', 'fndg', 'acty', 'mnob', 'plnt', 'podg', 'popg', 'prog', 'pros', 'elii',
            #                 'anim', 'inpr'
            #             ], prune=33,
            #             # for metamap lite only
            #             # restrict_to_sts=[],
            #         )
            #         if concepts:
            #             concepts_collection.extend(process_concepts(concepts))
            # except IndexError:
            #     pass

    with multiprocessing.Lock():
        logging.debug("Finished concept extraction with ROW_ID %s." % row_id)

    if len(concepts_collection) > 0:
        with open(os.environ['CONCEPT_ODIR'] + '{}_{}.pkl'.format(uid, row_id), 'wb') as wfile:
            pickle.dump(concepts_collection, wfile)


def extract_concepts_sequential(notes_df):
    # get list of row_id and text pairs
    texts = list(notes_df.TEXT.iteritems())
    uids = list(notes_df.SUBJECT_ID.iteritems())
    # filter out blank lines
    texts = [
        [item[0].strip(), item[1], uids[idx][1].strip()] for idx, item in enumerate(texts)
        if len(item) > 1 and len(item[1].strip()) > 2
    ]

    print('Extracting Concepts ...')
    for idx in tqdm(range(len(texts))):
        get_concept_thread(texts[idx])


def extract_concepts_parallel(notes_df):
    # get list of row_id and text pairs
    texts = list(notes_df.TEXT.iteritems())
    uids = list(notes_df.SUBJECT_ID.iteritems())
    # filter out blank lines
    texts = [
        [item[0], item[1], uids[idx][1]] for idx, item in enumerate(texts)
        if len(item) > 1 and len(item[1].strip()) > 2
    ]
    num_thread = os.cpu_count()
    pool = Pool(num_thread)

    print('Extracting Concepts ...')
    pool.map(get_concept_thread, texts)
    pool.close()


def process_mimic(indir, odir):
    """

    :param indir:
    :param odir:
    :return:
    """
    # progressively load the note event
    notes = pd.read_csv(
        indir + 'NOTEEVENTS.csv', dtype=str, index_col='ROW_ID',
        usecols=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY', 'TEXT']
    )
    notes.CHARTDATE = pd.to_datetime(notes.CHARTDATE)
    notes.SUBJECT_ID = notes.SUBJECT_ID.apply(lambda x: x.strip())

    # load patient table
    print('Getting patient and admission information...')
    patient_set = set(notes.SUBJECT_ID)
    patients = pd.read_csv(
        indir + 'PATIENTS.csv', dtype=str,
        usecols=['SUBJECT_ID', 'GENDER', 'DOB']
    )
    patients.SUBJECT_ID = patients.SUBJECT_ID.apply(lambda x: x.strip())
    patients = patients[patients.SUBJECT_ID.isin(patient_set)]
    patients.DOB = pd.to_datetime(patients.DOB)
    patients.fillna('x', inplace=True)
    patients = patients[patients.DOB != 'x']
    # convert to a dictionary for fast search, the first value is id, the 2nd value is the timestamp
    patients = dict((z[0], list(z[1:])) for z in zip(patients.SUBJECT_ID, patients.GENDER, patients.DOB))

    # get age of patients at the time of admission
    for index, row in notes.iterrows():
        if row['SUBJECT_ID'] not in patients:
            continue

        u_age = (
            row['CHARTDATE'].to_pydatetime() - patients[row['SUBJECT_ID']][1].to_pydatetime()
        ).total_seconds() / 3600 / 24 / 365

        patients[row['SUBJECT_ID']].append(u_age)

        if u_age < 18:
            del patients[row['SUBJECT_ID']]

    print('Filtering and Preprocessing Notes...')
    # only limit to the discharge summary
    # similar to https://github.com/jamesmullenbach/caml-mimic/blob/master/notebooks/dataproc_mimic_III.ipynb
    notes = notes[notes.CATEGORY == 'Discharge summary']
    # filter out notes by the patients age
    notes = notes[notes.SUBJECT_ID.isin(set(list(patients.keys())))]
    # remove some types of documents
    # notes = notes[notes.CATEGORY.isin([
    #     'Nursing/other', 'Radiology', 'Nursing', 'ECG', 'Physician ',
    #     'Discharge summary', 'Echo', 'Respiratory ',
    # ])]
    # filter out patients who have less than two notes
    # patient_counts = Counter(notes.SUBJECT_ID)
    # patient_counts = set([item[0] for item in patient_counts.items() if item[-1] >= 2])
    # notes = notes[notes.SUBJECT_ID.isin(patient_counts)]
    # filter out patients whoever generates less than 2 notes per stay
    # counts = Counter(notes.HADM_ID)
    # counts = dict([item for item in counts.items() if item[1] >= 2])
    # notes = notes[notes.HADM_ID.isin(counts)]

    # process the patients again
    patient_set = set(notes.SUBJECT_ID)
    for pid in list(patients.keys()):
        if pid not in patient_set:
            del patients[pid]
    print('We have number of patients: ', len(patients))

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
    print('Converting ICD codes...')
    icd_encoder = dict()
    dfile = yaml.load(open('../resources/hcup_ccs_2015_definitions.yaml'), Loader=yaml.FullLoader)
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

    # extract concepts from the notes
    notes_concepts_dir = os.environ['CONCEPT_ODIR']
    if not os.path.exists(notes_concepts_dir):
        os.mkdir(notes_concepts_dir)
    # extract_concepts_sequential(notes, notes_concepts_path)
    extract_concepts_parallel(notes)

    # preprocess the note documents, filter out documents less than 50 tokens
    # notes.TEXT = notes.TEXT.apply(lambda x: preprocess(x, min_len=50))
    # run parallel, consumes lots of memories for 15 GB / 48 workers.
    pandarallel.initialize()
    notes.TEXT = notes.TEXT.parallel_apply(lambda x: preprocess(x, min_len=50))
    notes.fillna('x', inplace=True)
    notes = notes[notes.TEXT != 'x']
    print('We have number of documents: ', len(notes))

    # loop through each note
    print('Processing each row...')
    results = dict()
    for index, row in notes.iterrows():
        uid = row['SUBJECT_ID'] + '-' + row['HADM_ID']
        if uid not in results:
            results[uid] = {
                'uid': uid,
                # calculate from the current stay and patient's DOB
                'age': patients[row['SUBJECT_ID']][2],  # third value is the age
                'gender': patients[row['SUBJECT_ID']][0],  # first value is the gender
                'ethnicity': admits[row['SUBJECT_ID']],  # ethnicity
                'tags_set': set(),  # convert to list in the end, unique tags
                'tags': list(),  # collect all patient tags
                'docs': list(),  # collect all patient notes
            }

        results[uid]['docs'].append({
            'doc_id': index,
            'date': row['CHARTDATE'].strftime('%Y-%m-%d'),
            'text': row['TEXT'],
            'tags': dfcodes[uid],
        })
        results[uid]['tags_set'].update(dfcodes[uid])
        results[uid]['tags'].extend(dfcodes[uid])

    opath = os.path.join(odir, 'mimic-iii.json')
    with open(opath, 'w') as wfile:
        for uid in results:
            # filter out empty records
            if len(results[uid]['docs']) == 0:
                continue
            results[uid]['tags_set'] = list(results[uid]['tags_set'])
            wfile.write(json.dumps(results[uid]) + '\n')


if __name__ == '__main__':
    # flist = ['amazon', 'diabetes', 'mimic']
    output_dir = 'data/processed_data/'
    os.environ['METAMAP_HOME'] = '/data/xiaolei/public_mm_lite/'

    # amazon health dataset
    # amazon_indir = './data/raw_data/amazon/'
    # if not os.path.exists(output_dir + 'amazon/'):
    #     os.mkdir(output_dir + 'amazon/')
    # process_amazon(amazon_indir, output_dir + 'amazon/')

    # diabetes
    diabetes_indir = './raw_data/diabetes/all/'
    os.environ['CONCEPT_ODIR'] = './processed_data/{}/concepts/'.format('diabetes')
    if not os.path.exists(output_dir + 'diabetes/'):
        os.mkdir(output_dir + 'diabetes/')
    process_diabetes(diabetes_indir, output_dir + 'diabetes/')

    # mimic-iii
    mimic_indir = '/data/xiaolei/physionet.org/files/mimiciii/1.4/'
    os.environ['CONCEPT_ODIR'] = './processed_data/{}/concepts/'.format('mimic-iii')
    if not os.path.exists(output_dir + 'mimic-iii/'):
        os.mkdir(output_dir + 'mimic-iii/')
    process_mimic(mimic_indir, output_dir + 'mimic-iii/')
