import os
import json
from collections import Counter
import pickle
import itertools

import statsmodels.api as sm
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def concept_stats(concept_dir, odir, task_name):
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
    flist = os.listdir(concept_dir)

    for fname in flist:
        uid = fname.split('_')[0]
        if uid not in user_concepts:
            # record semantic types and individual tokens
            user_concepts[uid] = {
                'semtypes': [],
                'entities': [],
            }
        concepts = pickle.load(open(concept_dir + fname, 'rb'))
        # control scores of the named entities
        for concept in concepts:
            if score_filter and float(concept['score']) < 3.6:
                continue

            user_concepts[uid]['semtypes'].append(concept['semtypes'])
            user_concepts[uid]['entities'].append(concept['preferred_name'].lower())
            general_info['num_concept'] += 1
            general_info['num_unique_concept'].update(concept['preferred_name'].lower())
            general_info['num_unique_concept_type'].update(concept['semtypes'])
            general_info['concept_type_stats'].extend(concept['semtypes'])
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
def qual_concepts_sim(**kwargs):
    corpus_path = kwargs['corpus_path']
    concept_dir = kwargs['concept_dir']
    data_stats_path = kwargs['data_stats_path']
    odir = kwargs['output_dir']
    task_name = kwargs['task_name']

    concept_files = set(os.listdir(concept_dir))
    data_stats = json.load(open(data_stats_path))
    labels = [item[0] for item in data_stats['tag_stats']]

    # get the user_docs
    if os.path.exists(odir + 'user_docs_{}.pkl'.format(task_name)):
        user_docs = pickle.load(open(odir + 'user_docs_{}.pkl'.format(task_name), 'rb'))
    else:
        user_docs = {}
        with open(corpus_path) as dfile:
            for line in dfile:
                user_entry = json.loads(line)
                if user_entry['uid'] not in user_docs:
                    user_docs[user_entry['uid']] = {
                        'doc': '',
                        'entity': [],
                        'label': [0] * len(labels),
                    }
                label_set = set(user_entry['tags'])

                # get label
                for idx, tag in enumerate(labels):
                    if tag in label_set:
                        user_docs[user_entry['uid']]['label'][idx] = 1
                if sum(user_docs[user_entry['uid']]['label']) in [0, 10]:
                    del user_docs[user_entry['uid']]
                    continue

                # get concepts and docs
                uid = user_entry['uid'].split('-')[0]
                for doc_entry in user_entry['docs']:
                    did = doc_entry['doc_id']
                    fname = '{}_{}.pkl'.format(uid, did)
                    if fname in concept_files:
                        concepts = pickle.load(open(concept_dir + fname, 'rb'))
                        concepts = [item['preferred_name'] for item in concepts]
                        user_docs[user_entry['uid']]['entity'].extend(concepts)

                    user_docs[user_entry['uid']]['doc'] += ' ' + doc_entry['text']

        with open(odir + 'user_docs_{}.pkl'.format(task_name), 'wb') as wfile:
            pickle.dump(user_docs, wfile)

    # build document feature encoder
    if os.path.exists(odir + 'vectorizer_doc_{}.pkl'.format(task_name)):
        vect_doc = pickle.load(open(odir + 'vectorizer_doc_{}.pkl'.format(task_name), 'rb'))
    else:
        vect_doc = TfidfVectorizer(
            max_features=10000, tokenizer=dummy_func, preprocessor=dummy_func)
        vect_doc.fit([user_docs[uid]['doc'] for uid in user_docs])
        with open(odir + 'vectorizer_doc_{}.pkl'.format(task_name), 'wb') as wfile:
            pickle.dump(vect_doc, wfile)

    # build concept feature encoder
    if os.path.exists(odir + 'vectorizer_concept_{}.pkl'.format(task_name)):
        vect_concept = pickle.load(open(odir + 'vectorizer_concept_{}.pkl'.format(task_name), 'rb'))
    else:
        vect_concept = TfidfVectorizer(max_features=10000, tokenizer=dummy_func, preprocessor=dummy_func)
        vect_concept.fit([user_docs[uid]['entity'] for uid in user_docs])
        with open(odir + 'vectorizer_concept_{}.pkl'.format(task_name), 'wb') as wfile:
            pickle.dump(vect_concept, wfile)




def dummy_func(doc):
    if type(doc) == str:
        return word_tokenize(doc)
    return doc


class LogisticRegressionKeras:
    def __init__(self, num_class, input_dim):
        self.model = Sequential()
        self.model.add(Dense(
            num_class, activation='sigmoid', kernel_regularizer=L1L2(l1=0, l2=0.001),
            input_dim=input_dim
        ))
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy'
        )

    def fit(self, train_x, train_y, epoch_num=50):
        self.model.fit(
            x=train_x, y=train_y, epochs=epoch_num
        )

    def predict(self, x_test):
        return self.model.predict(x_test)


# this function is to explore similar issue, if the medical concepts with users/patients are important;
# by the idea of building disease classification models only use medical entities as features.
# Therefore, we can compare performances of using entities only and using whole documents
# on patient disease classification task;
# Features will be all uni-grams
def quant_concepts_sim(**kwargs):
    corpus_path = kwargs['corpus_path']
    concept_dir = kwargs['concept_dir']
    data_stats_path = kwargs['data_stats_path']
    odir = kwargs['output_dir']
    task_name = kwargs['task_name']

    concept_files = set(os.listdir(concept_dir))
    data_stats = json.load(open(data_stats_path))
    num_label = 10  # only experiment with top 10 labels
    top_labels = [item[0] for item in data_stats['tag_stats']]
    top_labels = top_labels[:num_label]  # 2nd element is label occurrence
    # num_epochs = 15

    if os.path.exists(odir + 'user_docs_{}.pkl'.format(task_name)):
        user_docs = pickle.load(open(odir + 'user_docs_{}.pkl'.format(task_name), 'rb'))
    else:
        user_docs = {}
        with open(corpus_path) as dfile:
            for line in dfile:
                user_entry = json.loads(line)
                if user_entry['uid'] not in user_docs:
                    user_docs[user_entry['uid']] = {
                        'doc': [],
                        'entity': [],
                        'label': [0] * len(top_labels),
                    }
                label_set = set(user_entry['tags'])

                # get label
                for idx, tag in enumerate(top_labels):
                    if tag in label_set:
                        user_docs[user_entry['uid']]['label'][idx] = 1
                if sum(user_docs[user_entry['uid']]['label']) in [0, len(top_labels)]:
                    del user_docs[user_entry['uid']]
                    continue

                # get concepts and docs
                uid = user_entry['uid'].split('-')[0]
                for doc_entry in user_entry['docs']:
                    did = doc_entry['doc_id']
                    fname = '{}_{}.pkl'.format(uid, did)
                    if fname in concept_files:
                        concepts = pickle.load(open(concept_dir + fname, 'rb'))
                        concepts = [item['preferred_name'] for item in concepts]
                        user_docs[user_entry['uid']]['entity'].extend(concepts)

                    user_docs[user_entry['uid']]['doc'].extend(doc_entry['text'].split())

        with open(odir + 'user_docs_{}.pkl'.format(task_name), 'wb') as wfile:
            pickle.dump(user_docs, wfile)

    uids = list(user_docs.keys())
    # for uid in uids:
    #     if sum(user_docs[uid]['label']) == 10 or 0:
    #         print(user_docs[uid]['label'])
    # import sys
    # sys.exit()
    np.random.shuffle(uids)

    if os.path.exists(odir + 'vectorizer_doc_{}.pkl'.format(task_name)):
        vect_doc = pickle.load(open(odir + 'vectorizer_doc_{}.pkl'.format(task_name), 'rb'))
    else:
        vect_doc = TfidfVectorizer(
            max_features=10000, tokenizer=dummy_func, preprocessor=dummy_func)
        vect_doc.fit([user_docs[uid]['doc'] for uid in user_docs])
        with open(odir + 'vectorizer_doc_{}.pkl'.format(task_name), 'wb') as wfile:
            pickle.dump(vect_doc, wfile)

    if os.path.exists(odir + 'vectorizer_concept_{}.pkl'.format(task_name)):
        vect_concept = pickle.load(open(odir + 'vectorizer_concept_{}.pkl'.format(task_name), 'rb'))
    else:
        vect_concept = TfidfVectorizer(max_features=10000, tokenizer=dummy_func, preprocessor=dummy_func)
        vect_concept.fit([user_docs[uid]['entity'] for uid in user_docs])
        with open(odir + 'vectorizer_concept_{}.pkl'.format(task_name), 'wb') as wfile:
            pickle.dump(vect_concept, wfile)

    kf = KFold(n_splits=5)
    doc_f1 = []
    concept_f1 = []
    for train_idx, test_idx in kf.split(uids):
        train_uids = [uids[item] for item in train_idx]
        test_uids = [uids[item] for item in test_idx]

        x_train = [user_docs[item]['doc'] for item in train_uids]
        x_test = [user_docs[item]['doc'] for item in test_uids]
        y_train = np.asarray([user_docs[item]['label'] for item in train_uids])
        y_test = [user_docs[item]['label'] for item in test_uids]
        y_test = np.asarray(list(itertools.chain.from_iterable(y_test)))

        # train and test logistic regression on documents
        x_train = vect_doc.transform(x_train).toarray()
        x_test = vect_doc.transform(x_test).toarray()
        # lr_model = LogisticRegressionKeras(num_class=num_label, input_dim=vect_doc.max_features)
        # lr_model.fit(x_train, y_train, epoch_num=15)

        # lr_model = MultiOutputClassifier(KNeighborsClassifier(), n_jobs=-1)
        # lr_model = MultiOutputClassifier(DecisionTreeClassifier(), n_jobs=-1)
        lr_model = MultiOutputClassifier(MLPClassifier(
            early_stopping=True, activation='logistic', max_iter=1000), n_jobs=-1)
        lr_model.fit(x_train, y_train)

        y_pred = lr_model.predict(x_test)
        # y_pred = y_pred.round()
        y_pred = np.asarray(list(itertools.chain.from_iterable(y_pred)))
        doc_f1.append(f1_score(y_true=y_test, y_pred=y_pred, average='weighted'))

        # train and test logistic regression on concepts
        x_train = [user_docs[item]['entity'] for item in train_uids]
        x_test = [user_docs[item]['entity'] for item in test_uids]
        x_train = vect_concept.transform(x_train).toarray()
        x_test = vect_concept.transform(x_test).toarray()
        # lr_model = LogisticRegressionKeras(num_class=num_label, input_dim=vect_concept.max_features)
        # lr_model.fit(x_train, y_train, epoch_num=15)

        # lr_model = MultiOutputClassifier(KNeighborsClassifier(), n_jobs=-1)
        # lr_model = MultiOutputClassifier(DecisionTreeClassifier(), n_jobs=-1)
        lr_model = MultiOutputClassifier(MLPClassifier(
            early_stopping=True, activation='logistic', max_iter=1000), n_jobs=-1)
        lr_model.fit(x_train, y_train)

        y_pred = lr_model.predict(x_test)
        # y_pred = y_pred.round()
        y_pred = np.asarray(list(itertools.chain.from_iterable(y_pred)))
        concept_f1.append(f1_score(y_true=y_test, y_pred=y_pred, average='weighted'))

    print('Concept 5-fold F1: ', concept_f1)
    print('Concept 5-fold F1 Average: ', np.mean(concept_f1))
    print('Doc 5-fold F1: ', doc_f1)
    print('Doc 5-fold F1 Average: ', np.mean(doc_f1))
    # feature Jaccard similarities
    concept_keys = set([item.lower() for item in vect_concept.vocabulary_.keys()])
    doc_keys = set([item.lower() for item in vect_doc.vocabulary_.keys()])
    print(
        'Feature Similarities between Doc and Concept: ',
        len(concept_keys.intersection(doc_keys)) / len(doc_keys.union(concept_keys))
    )


if __name__ == '__main__':
    dlist = ['mimic-iii']  # 'diabetes', 'mimic-iii'
    output_dir = '../resources/analyze/'
    indir = './processed_data/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # generate data stats for each dataset
    for dname in dlist:
        data_path = indir + dname + '/{}.json'.format(dname)
        concet_dir = indir + '{}/concepts/'.format(dname)

        concept_odir = output_dir + '{}/'.format(dname)
        if not os.path.exists(concept_odir):
            os.mkdir(concept_odir)
        quant_odir = concept_odir + 'quant/'
        if not os.path.exists(quant_odir):
            os.mkdir(quant_odir)
        qual_odir = concept_odir + 'qual/'
        if not os.path.exists(qual_odir):
            os.mkdir(qual_odir)

        # get stats of concepts
        # concept_stats(concet_dir, concept_odir, dname)
        # quantitative analysis
        quant_concepts_sim(
            corpus_path=data_path,
            concept_dir=concet_dir,
            data_stats_path=output_dir + '{}_stats.json'.format(dname, dname),
            output_dir=quant_odir,
            task_name=dname,
        )

        # # another analysis perspective
        # qual_concepts_sim(
        #     corpus_path=data_path,
        #     concept_dir=concet_dir,
        #     data_stats_path=output_dir + '{}_stats.json'.format(dname, dname),
        #     output_dir=quant_odir,
        #     task_name=dname,
        # )
