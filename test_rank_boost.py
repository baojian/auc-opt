# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from data_preprocess import data_preprocess_yeast
from sklearn.metrics import roc_auc_score


def re_order(r1, fid):
    pass


def update_potential(num_samples):
    for i in range(len(num_samples)):
        pass


def learn_weaker_ranker(init_d):
    pass


def init():
    pass


def learn():
    pass


def rank_boost(num_iter, num_threshold):
    init_d = None
    weak_rank_ht = None
    for t in range(num_iter):
        learn_weaker_ranker(init_d=init_d)
    pass


def main():
    for trial_id in range(200):
        f = open('aucs.txt', mode='a')
        os.system(" java -jar ~/RankLib-2.13/target/RankLib-2.13.jar -silent "
                  "-train /network/rit/lab/ceashpc/bz383376/data/auc-logistic/17_yeast/yeast_tr_%03d.dat "
                  "-test /network/rit/lab/ceashpc/bz383376/data/auc-logistic/17_yeast/yeast_te_%03d.dat "
                  "-ranker 2 -round 1000 -tc 8 -metric2t ERR@8 -metric2T ERR@8 -save model.txt"
                  % (trial_id, trial_id))
        os.system("java -jar ~/RankLib-2.13/target/RankLib-2.13.jar -load model.txt -rank "
                  "/network/rit/lab/ceashpc/bz383376/data/auc-logistic/17_yeast/yeast_te_%03d.dat "
                  "-score scores.txt" % trial_id)
        y_true = []
        with open('/network/rit/lab/ceashpc/bz383376/data/auc-logistic/17_yeast/yeast_te_%03d.dat'
                  % trial_id) as ff:
            for each_line in ff.readlines():
                y_true.append(1 if each_line.split(' ')[0] == '2' else -1)
        y_true = np.asarray(y_true)
        y_score = []
        with open('scores.txt') as ff:
            for each_line in ff.readlines():
                y_score.append(float(each_line.rstrip().split('\t')[-1]))
        f.write('%.4f\n' % roc_auc_score(y_true=y_true, y_score=np.asarray(y_score)))
        f.close()


if __name__ == '__main__':
    main()
