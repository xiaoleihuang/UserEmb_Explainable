import json
import os
import subprocess

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def analysis_viz(dpath, data_name='Diabetes'):
    df = pd.read_csv(dpath)
    df = df[df.data == data_name]
    sns.set_theme(style="white")
    ax = sns.barplot(x="feature", y="f1-score", data=df)
    ax.set(ylim=(min(df['f1-score']) * 0.9, max(df['f1-score']) * 1.05))

    for bar in ax.patches:
        ax.annotate(
            format(bar.get_height(), '.3f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', va='center', size=15, xytext=(0, 8), textcoords='offset points', fontsize=16
        )
    ax.set_title(data_name, fontsize=18)
    ax.set_ylabel("F1-score", fontsize=15)
    ax.set_xlabel("Feature Type", fontsize=15)
    # plt.show()
    opath = '../resources/analyze/{}/{}_quant.pdf'.format(data_name.lower(), data_name.lower())
    ax.figure.savefig(opath, format='pdf')
    plt.close()
    subprocess.run("pdfcrop {} {}".format(opath, opath), shell=True, text=True)


def user_viz(data_name, method_name):
    emb_dir = '../resources/embedding/{}/{}/'.format(data_name, method_name)
    data_dir = '../data/processed_data/{}/'.format(data_name)
    stats_path = '../resources/analyze/{}_stats.json'.format(data_name)

    # load user encoder
    with open(data_dir + 'user_encoder.json') as dfile:
        user_encoder = json.load(dfile)

    # load tag stats
    tag_stats = json.load(open(stats_path))['tag_stats']
    if len(tag_stats) > 50:
        tag_stats = tag_stats[:50]
    top_tag = tag_stats[0][0]
    tag_stats = dict(tag_stats)

    tag_encoder = dict()
    user_tags = dict()
    with open(data_dir+data_name + '.json') as dfile:
        for line in dfile:
            user_info = json.loads(line)
            uid = user_encoder[user_info['uid']]

            for tag in user_info['tags_set']:
                if tag not in tag_encoder:
                    tag_encoder[tag] = len(tag_encoder)
                    # remove the tag if the tag not in filter
                    if tag_stats and tag not in tag_stats:
                        del tag_encoder[tag]

            # user_tags[uid] = {
            #     'tags': Counter(user_info['tags']),
            #     'tags_set': user_info['tags_set']
            # }
            user_tags[uid] = 1 if top_tag in user_info['tags_set'] else 0

    # write down the results
    inpath = '../resources/analyze/{}/{}_viz.tsv'.format(data_name, method_name)
    if not os.path.exists(inpath):
        # load user embeddings
        # load the user embeddings, default to load the user.npy
        if os.path.exists(emb_dir + 'user.npy'):
            uembs = np.load(emb_dir + 'user.npy')
        else:
            # this assumes using .txt file
            uembs = [[]] * len(user_tags)
            with open(emb_dir + 'user.txt') as dfile:
                for line in dfile:
                    line = line.strip().split('\t')
                    if len(line) != 2:
                        continue
                    uid = user_encoder[line[0]]
                    uembs[uid] = [float(item) for item in line[1].split()]
            uembs = np.asarray(uembs)

        tsne = TSNE(n_components=2, n_jobs=-1)
        uembs = tsne.fit_transform(uembs)

        with open(inpath, 'w') as wfile:
            wfile.write('x\ty\tlabel\n')
            for udx in user_tags:
                wfile.write('{}\t{}\t{}\n'.format(uembs[udx][0], uembs[udx][1], user_tags[udx]))

    # visualization
    df = pd.read_csv(inpath, sep='\t')
    a4_dims = (12.27, 12.27)
    # cmap = sns.cubehelix_palette(as_cmap=True)
    fig, ax = plt.subplots(figsize=a4_dims)

    opath = '../resources/analyze/{}/{}_viz.pdf'.format(data_name, method_name)
    # points = sns.scatterplot()
    # # fig.colorbar(points)
    #
    # plt.ylabel('X', fontsize=20)
    # plt.xlabel('Y', fontsize=20)
    # plt.title(method, fontsize=20)
    # plt.savefig(opath, format='pdf')
    # plt.show()
    # plt.close()

    viz_plot = sns.scatterplot(data=df, x='x', y='y', hue='label', ax=ax)
    viz_plot.set_ylabel('X', fontsize=20)
    viz_plot.set_xlabel('Y', fontsize=20)
    # plt.setp(ax.get_legend().get_texts(), fontsize=22)
    # plt.setp(ax.get_legend().get_title(), fontsize=22)
    viz_plot.figure.savefig(opath, format='pdf')
    plt.close()


if __name__ == '__main__':
    # quant_path = '../resources/analyze/quant.csv'
    # for dname in ['Diabetes', 'MIMIC-III']:
    #     analysis_viz(quant_path, dname)

    for dname in ['diabetes', 'mimic-iii']:
        for method in ['caue_gru', 'user2vec', 'suisil2user']:
            user_viz(dname, method)
