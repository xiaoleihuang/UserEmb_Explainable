"""Evaluation scripts
"""
import json
import os
import argparse
import datetime
from collections import Counter

import numpy as np
from scipy.spatial import distance
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import SpectralClustering
import keras
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for cpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def data_loader(params):
    # load the user encoder
    with open(params['data_dir'] + 'user_encoder.json') as dfile:
        user_encoder = json.load(dfile)

    # load the tag stats to filter long-tail tags
    tag_stats = json.load(open(params['stats_path']))['tag_stats']
    if len(tag_stats) > params['top_tags']:
        tag_stats = tag_stats[:params['top_tags']]
    tag_stats = dict(tag_stats)

    # open the data
    tag_encoder = dict()
    user_tags = dict()
    with open(params['data_dir'] + params['dname'] + '.json') as dfile:
        for line in dfile:
            user_info = json.loads(line)
            uid = user_encoder[user_info['uid']]
            for tag in user_info['tags_set']:
                if tag not in tag_encoder:
                    tag_encoder[tag] = len(tag_encoder)
                    # remove the tag if the tag not in filter
                    if tag_stats and tag not in tag_stats:
                        del tag_encoder[tag]

            user_tags[uid] = {
                'tags': Counter(user_info['tags']),
                'tags_set': user_info['tags_set']
            }

    # load the user embeddings, default to load the user.npy
    if os.path.exists(params['emb_dir'] + 'user.npy'):
        uembs = np.load(params['emb_dir'] + 'user.npy')
    elif os.path.exists(params['emb_dir'] + 'user.txt'):
        # this assumes using .txt file
        uembs = [[]] * len(user_tags)
        with open(params['emb_dir'] + 'user.txt') as dfile:
            for line in dfile:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                uid = user_encoder[line[0]]
                uembs[uid] = [float(item) for item in line[1].split()]
    else:
        uembs = np.load(params['emb_dir'] + 'user_9.npy')
    return uembs, user_tags, tag_encoder, user_encoder


def regression(params):
    """ Regression Evaluation Methods

    :param params: parameters
    :return:
    """
    opath = params['odir'] + 'regression-{}.json'.format(params['dname'])
    uembs, user_tags, tag_encoder, user_encoder = data_loader(params)

    # generate true labels by comparing the user tag similarity
    true_labels_tags = []
    true_labels_sets = []
    pred_labels_vals = []
    uids = list(user_tags.keys())
    for idx in range(len(user_tags)):
        for jdx in range(idx, len(user_tags)):
            idx_uid = uids[idx]
            jdx_uid = uids[jdx]

            # for true_label_tags
            idx_tags = [user_tags[idx_uid]['tags'].get(tag, 0) for tag in tag_encoder]
            tag_sum = sum(idx_tags)
            if tag_sum == 0:
                continue
            idx_tags = [item / tag_sum for item in idx_tags]  # normalization

            jdx_tags = [user_tags[jdx_uid]['tags'].get(tag, 0) for tag in tag_encoder]
            tag_sum = sum(jdx_tags)
            if tag_sum == 0:
                continue
            jdx_tags = [item / tag_sum for item in jdx_tags]  # normalization

            if params['sim_method'] == 'cosine':
                true_labels_tags.append(distance.cosine(idx_tags, jdx_tags))
            else:
                true_labels_tags.append(1 / (1 + np.exp(-1 * np.dot(idx_tags, jdx_tags))))

            # for true_labels_sets
            idx_tags = [1 if tag in user_tags[idx_uid]['tags_set'] else 0 for tag in tag_encoder]
            jdx_tags = [1 if tag in user_tags[jdx_uid]['tags_set'] else 0 for tag in tag_encoder]
            if params['sim_method'] == 'cosine':
                true_labels_sets.append(distance.cosine(idx_tags, jdx_tags))
            else:
                true_labels_sets.append(1 / (1 + np.exp(-1 * np.dot(idx_tags, jdx_tags))))

            # for predictions
            if params['sim_method'] == 'cosine':
                pred_labels_vals.append(distance.cosine(uembs[idx_uid], uembs[jdx_uid]))
            else:
                pred_labels_vals.append(1 / (1 + np.exp(-1 * np.dot(uembs[idx_uid], uembs[jdx_uid]))))

    # calculate rmse
    results = dict()
    results['rmse_tags'] = np.sqrt(
        metrics.mean_squared_error(
            y_true=true_labels_tags, y_pred=pred_labels_vals
        )
    )
    results['rmse_tags_set'] = np.sqrt(
        metrics.mean_squared_error(
            y_true=true_labels_sets, y_pred=pred_labels_vals
        )
    )

    # calculate r-square
    results['r2_tags'] = metrics.r2_score(
        y_true=true_labels_tags, y_pred=pred_labels_vals
    )
    results['r2_tags_set'] = metrics.r2_score(
        y_true=true_labels_sets, y_pred=pred_labels_vals
    )

    results = json.dumps(results, indent=4)
    print(results)
    with open(opath, 'a') as wfile:
        wfile.write(json.dumps(params) + '\n')
        wfile.write(results)
        wfile.write('\n\n')


def classification(params):
    """

    :param params: parameters
    :return:
    """
    opath = params['odir'] + 'classification-{}.json'.format(params['dname'])
    uembs, user_tags, tag_encoder, user_encoder = data_loader(params)
    uids = list(user_tags.keys())

    # format into x, y instances
    data = []
    labels = []
    for uid in uids:
        y = [0] * len(tag_encoder)
        for tag in user_tags[uid]['tags_set']:
            if tag not in tag_encoder:
                continue
            y[tag_encoder[tag]] = 1
        labels.append(y)
        # uid has been converted into index in previous steps
        data.append(uembs[uid])
    data = np.asarray(data)
    labels = np.asarray(labels)

    # split into train/test, k-folds cross validation
    kf = KFold(n_splits=5, shuffle=True)
    results = {
        'precision': [],
        'recall': [],
        'f1-score': [],
    }
    for train_idx, test_idx in kf.split(data):
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        lr = keras.models.Sequential()
        lr.add(keras.layers.BatchNormalization())
        lr.add(keras.layers.Dense(
            len(y_train[0]), activation='softmax',
            kernel_regularizer=keras.regularizers.L1L2(l1=0, l2=0.001),
            input_dim=len(x_train[0])
        ))
        lr.compile(optimizer='adam', loss='categorical_crossentropy')
        lr.fit(x_train, y_train, epochs=1000, verbose=False)
        y_preds_probs = lr.predict(x_test)
        y_preds = y_preds_probs.round()

        for item in zip(y_preds, y_test):
            # results['accuracy'].append(metrics.accuracy_score(y_true=item[1], y_pred=item[0]))
            results['precision'].append(metrics.precision_score(y_true=item[1], y_pred=item[0], average='binary'))
            results['recall'].append(metrics.recall_score(y_true=item[1], y_pred=item[0], average='binary'))
            results['f1-score'].append(metrics.f1_score(y_true=item[1], y_pred=item[0], average='binary'))

    # average the scores
    # results['accuracy'] = np.mean(results['accuracy'])
    results['precision'] = np.mean(results['precision'])
    results['recall'] = np.mean(results['recall'])
    results['f1-score'] = np.mean(results['f1-score'])

    results = json.dumps(results, indent=4)
    print(results)
    with open(opath, 'a') as wfile:
        wfile.write(json.dumps(params) + '\n')
        wfile.write(results)
        wfile.write('\n\n')


def retrieval(params):
    # Jaccard similarity measurements
    opath = params['odir'] + 'retrieval-{}.json'.format(params['dname'])
    uembs, user_tags, tag_encoder, user_encoder = data_loader(params)
    uembs = np.asarray(uembs)
    uids = list(user_tags.keys())
    results = {
        'tags_set': [],
        'tags': [],
    }

    select_num = 10
    # random_seed = 33
    # np.random.seed(random_seed)
    # uids = np.random.choice(uids, size=select_num, replace=False)

    for uid in uids:
        similar_users = np.dot(uembs, uembs[uid].T)
        similar_users = np.argsort(similar_users)

        for idx, sim_user in enumerate(similar_users):
            if sim_user not in user_tags or sim_user == uid:
                continue
            if len(user_tags[sim_user]['tags']) < 1:
                continue
            if idx == select_num:
                break

            shared_tags = set([item for item in user_tags[uid]['tags_set'] if item in tag_encoder])
            if len(shared_tags) < 1:
                continue
            union_tags = shared_tags.union([item for item in user_tags[sim_user]['tags_set'] if item in tag_encoder])
            shared_tags = shared_tags.intersection(user_tags[sim_user]['tags_set'])

            results['tags'].append(
                (sum([user_tags[uid]['tags'].get(tag, 0) for tag in shared_tags]) + sum(
                    [user_tags[sim_user]['tags'].get(tag, 0) for tag in shared_tags])) /
                (sum(user_tags[uid]['tags'].values()) + sum(user_tags[sim_user]['tags'].values()))
            )
            results['tags_set'].append(len(shared_tags) / len(union_tags))

    results['tags'] = np.mean(results['tags'])
    results['tags_set'] = np.mean(results['tags_set'])

    results = json.dumps(results, indent=4)
    print(results)
    with open(opath, 'a') as wfile:
        wfile.write(json.dumps(params) + '\n')
        wfile.write(results)
        wfile.write('\n\n')


def mortality_eval(params):
    """
    This function is design for mimic-iii evaluation only
    Returns
    -------

    """
    if params['dname'] != 'mimic-iii':
        print('Only MIMIC-III is supported!')
        return

    # load dataset
    uembs, _, _, user_encoder = data_loader(params)
    # load the mortality labels
    mortality_labels = json.load(open(params['data_dir'] + 'mortality.json'))

    # two types of evaluation: classification and clustering
    data_x = []
    data_y = []
    idx2user = dict()
    for uid in user_encoder.keys():
        if uid not in mortality_labels:
            continue
        data_x.append(uembs[user_encoder[uid]])
        data_y.append(mortality_labels[uid])
        idx2user[len(idx2user)] = uid
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)

    # five folds cross evaluation
    lr = LogisticRegression(class_weight='balanced', n_jobs=-1)
    # split into train/test, k-folds cross validation
    kf = KFold(n_splits=5, shuffle=True)
    results = {
        'precision': [],
        'recall': [],
        'f1-score': [],
        'clustering': 0
    }
    for train_idx, test_idx in tqdm(kf.split(data_y), total=5):
        x_train, x_test = data_x[train_idx], data_x[test_idx]
        y_train, y_test = data_y[train_idx], data_y[test_idx]
        lr.fit(X=x_train, y=y_train)
        predicts = lr.predict(x_test)
        results['precision'].append(metrics.precision_score(y_pred=predicts, y_true=y_test))
        results['recall'].append(metrics.recall_score(y_pred=predicts, y_true=y_test))
        results['f1-score'].append(metrics.f1_score(y_pred=predicts, y_true=y_test, average='weighted'))

    # clustering
    cluster = SpectralClustering(n_clusters=2, n_jobs=-1)
    cluster_labels = cluster.fit_predict(data_x)
    predicts = []
    y_test = []
    for uidx in range(len(cluster_labels)):
        for ujdx in range(uidx + 1, len(cluster_labels)):
            if mortality_labels[idx2user[uidx]] == mortality_labels[idx2user[ujdx]]:
                y_test.append(1)
                if cluster_labels[uidx] == cluster_labels[ujdx]:
                    predicts.append(1)
                else:
                    predicts.append(0)
            else:
                y_test.append(0)
                if cluster_labels[uidx] == cluster_labels[ujdx]:
                    predicts.append(0)
                else:
                    predicts.append(1)
    results['clustering'] = metrics.f1_score(y_pred=predicts, y_true=y_test, average='weighted')
    results = json.dumps(results, indent=4)
    print(results)

    opath = params['odir'] + 'mortality-{}.json'.format(params['dname'])
    with open(opath, 'a') as wfile:
        wfile.write(json.dumps(params) + '\n')
        wfile.write(results)
        wfile.write('\n\n')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dname', type=str, help='data name')
    args.add_argument('--model', type=str, help='user embedding model')
    args.add_argument('--sim_method', type=str, default='cosine')
    args.add_argument('--top_tags', type=int, default=50)
    args = args.parse_args()

    # categories of data names
    parameters = {
        'dname': args.dname,
        'model': args.model,
        'data_dir': './data/processed_data/{}/'.format(args.dname),
        'emb_dir': './resources/embedding/{}/{}/'.format(args.dname, args.model),
        'stats_path': './resources/analyze/{}_stats.json'.format(args.dname),
        'odir': './resources/eval/',
        'eval_time': datetime.datetime.now().strftime('%H:%M:%S %m-%d-%Y'),
        'sim_method': args.sim_method,
        'top_tags': args.top_tags
    }
    if not os.path.exists(parameters['odir']):
        os.mkdir(parameters['odir'])

    # print('Regression Evaluation: ')
    # regression(parameters)
    # print()
    #
    # print('Classification Evaluation: ')
    # classification(parameters)
    # print()
    #
    # print('Retrieval Evaluation: ')
    # retrieval(parameters)
    # print()

    print('MIMIC-III Mortality: ')
    mortality_eval(parameters)
    print()
