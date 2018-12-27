# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:03:11 2018

@author: kaush
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

SO_ANSWERS = os.path.join(os.path.dirname(__file__), 'answer.csv')


def load_so_answers():
	return pd.read_csv(SO_ANSWERS, index_col=0)

def show_viz():
    feature_list = ['Readability','Answer_score','Answerer_score','Similarity','Time_diff','Polarity']

    dtm = load_so_answers()
    plt.show()
    plt.figure()
    sns.countplot(x='LABEL',data=dtm)
    plt.figure()
    g = sns.distplot(dtm.ANSWER_SCORE, kde=False)
    g.set_xscale('log')
    dft = dtm.dropna()
    plt.figure()
    sns.distplot(dft.ANSWERER_SCORE, kde=False)
    plt.figure()
    sns.distplot(dft.SIMILARITY, kde=False)
    plt.figure()
    sns.distplot(dft.POLARITY, kde=False)
    plt.figure()
    sns.stripplot(x='LABEL', y='ANSWERER_SCORE', data=dtm, alpha=0.3, jitter=True)
    plt.figure()
    sns.stripplot(x='LABEL', y='ANSWER_SCORE', data=dtm, alpha=0.3, jitter=True)
    plt.figure()
    sns.stripplot(x='LABEL', y='SIMILARITY', data=dtm, alpha=0.3, jitter=True)
    plt.figure()
    sns.stripplot(x='LABEL', y='TIME_DIFF', data=dtm, alpha=0.3, jitter=True)
    plt.figure()
    sns.stripplot(x='LABEL', y='POLARITY', data=dtm, alpha=0.3, jitter=True)
    plt.figure()
    sns.pairplot(dft, hue='LABEL')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(dtm.iloc[:, :-1].corr())
    ax.set_xticklabels(['']+feature_list)
    ax.set_yticklabels(['']+feature_list)
    fig.colorbar(cax)
    plt.show()
def main():
	show_viz()

if __name__ == '__main__':
	main()