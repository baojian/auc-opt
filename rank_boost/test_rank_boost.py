# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
from data_preprocess import get_data
from sklearn.metrics import roc_auc_score

try:
    from librank_boost_3 import c_rank_boost
except ImportError:
    pass


def rank_boost_debug(x_tr, y_tr, t):
    n, dim = x_tr.shape
    indpos = np.argwhere(y_tr == 1).flatten()
    npos = len(indpos)
    indneg = np.argwhere(y_tr == -1).flatten()
    nneg = len(indneg)
    y_tr = np.concatenate((y_tr[indpos], y_tr[indneg]))
    x_tr = np.concatenate((x_tr[indpos, :], x_tr[indneg, :]))
    print(indpos, indneg)
    indpos = np.argwhere(y_tr == 1).flatten()
    indneg = np.argwhere(y_tr == -1).flatten()
    v = np.concatenate((1 / npos * np.ones(npos), 1 / nneg * np.ones(nneg)))
    s = -y_tr
    rankfeat = np.zeros(t, dtype=int)
    threshold = np.zeros(t)
    alpha = np.zeros(t)
    print(indpos, indneg)
    print('===')
    for k in range(t):
        vpos = np.sum(v[indpos])
        vneg = np.sum(v[indneg])
        print('k: %d vpos: %.2f vneg: %.2f' % (k, vpos, vneg))
        d = v * np.concatenate((vneg * np.ones(npos, dtype=float), vpos * np.ones(nneg)))
        mpi = s * d
        print('d:  %s' % ' '.join(['%.6f' % _ for _ in d]))
        print('s:  %s' % ' '.join(['%.6f' % _ for _ in s]))
        print('pi: %s' % ' '.join(['%.6f' % _ for _ in mpi]))
        rmax = 0
        for i in range(dim):
            fi = x_tr[:, i]
            sortedfi = np.asarray(sorted(fi)[::-1])
            candthreshold = [np.inf]
            candthreshold.extend(list(sortedfi - np.min(np.abs(np.diff(sortedfi))) / 2))
            L = 0
            print('candthreshold: %s' % ' '.join(['%.6f' % _ for _ in candthreshold]))
            print("L: %.6f" % L, end=' ')
            for j in range(1, len(sortedfi) + 1):
                ind = np.argwhere((candthreshold[j] < fi) & (fi <= candthreshold[j - 1]))
                ind = ind.flatten()
                L = L + np.sum(mpi[ind])
                if np.abs(L) > np.abs(rmax):
                    rmax = L
                    rankfeat[k] = i
                    threshold[k] = candthreshold[j]
                print('%.6f ' % L, end=' ')
            print('')
            if np.abs(np.abs(rmax) - 1.) < 0.00001:
                rmax = np.sign(rmax) * 0.99999
        alpha[k] = 0.5 * np.log((1. + rmax) / (1. - rmax))
        print('rmax: %.6f alpha[%d]: %.6f threshold[%d]: %.6f rankfeat[%d]: %d' %
              (rmax, k, alpha[k], k, threshold[k], k, rankfeat[k]))
        v1 = v[indneg] * np.exp(-alpha[k] * (x_tr[indneg, rankfeat[k]] > threshold[k]))
        v0 = v[indpos] * np.exp(alpha[k] * (x_tr[indpos, rankfeat[k]] > threshold[k]))
        if np.sum(v0) == 0 or np.sum(v1) == 0:
            v = np.concatenate((v0, v1))
        else:
            v = np.concatenate((v0 / np.sum(v0), v1 / np.sum(v1)))
        print('v: %s' % ' '.join(['%.6f' % _ for _ in v]))
        print('\n=====\n')
    alpha = -alpha
    print('finish')
    print(alpha)
    print(threshold)
    print(rankfeat)
    return alpha, threshold, rankfeat


def rank_boost(x_tr, y_tr, t):
    start_time = time.time()
    n, dim = x_tr.shape
    indpos = np.argwhere(y_tr == 1).flatten()
    npos = len(indpos)
    indneg = np.argwhere(y_tr == -1).flatten()
    nneg = len(indneg)
    y_tr = np.concatenate((y_tr[indpos], y_tr[indneg]))
    x_tr = np.concatenate((x_tr[indpos, :], x_tr[indneg, :]))
    indpos = np.argwhere(y_tr == 1).flatten()
    indneg = np.argwhere(y_tr == -1).flatten()
    v = np.concatenate((1 / npos * np.ones(npos), 1 / nneg * np.ones(nneg)))
    s = -y_tr
    rankfeat = np.zeros(t, dtype=int)
    threshold = np.zeros(t)
    alpha = np.zeros(t)
    for k in range(t):
        vpos = np.sum(v[indpos])
        vneg = np.sum(v[indneg])
        d = v * np.concatenate((vneg * np.ones(npos, dtype=float), vpos * np.ones(nneg)))
        mpi = s * d
        rmax = 0
        for i in range(dim):
            fi = x_tr[:, i]
            sortedfi = np.asarray(sorted(fi)[::-1])
            candthreshold = [np.inf]
            candthreshold.extend(list(sortedfi - np.min(np.abs(np.diff(sortedfi))) / 2))
            L = 0
            for j in range(1, len(sortedfi) + 1):
                ind = np.argwhere((candthreshold[j] < fi) & (fi <= candthreshold[j - 1]))
                ind = ind.flatten()
                L = L + np.sum(mpi[ind])
                if np.abs(L) > np.abs(rmax):
                    rmax = L
                    rankfeat[k] = i
                    threshold[k] = candthreshold[j]
            if np.abs(np.abs(rmax) - 1.) < 0.00001:
                rmax = np.sign(rmax) * 0.99999
        alpha[k] = 0.5 * np.log((1. + rmax) / (1. - rmax))
        v1 = v[indneg] * np.exp(-alpha[k] * (x_tr[indneg, rankfeat[k]] > threshold[k]))
        v0 = v[indpos] * np.exp(alpha[k] * (x_tr[indpos, rankfeat[k]] > threshold[k]))
        if np.sum(v0) == 0 or np.sum(v1) == 0:
            v = np.concatenate((v0, v1))
        else:
            v = np.concatenate((v0 / np.sum(v0), v1 / np.sum(v1)))
    alpha = -alpha
    return alpha, threshold, rankfeat, time.time() - start_time


def decision_func(x_tr, alpha, threshold, rankfeat):
    y_pred = np.zeros(len(x_tr))
    for i in range(len(alpha)):
        y_pred += alpha[i] * (x_tr[:, rankfeat[i]] > threshold[i])
    return y_pred


def test_2():
    dataset = 'spambase'
    trial_i = 0
    t = 50
    data = get_data(dataset=dataset, num_trials=200)
    tr_index = data['trial_%d_tr_indices' % trial_i]
    te_index = data['trial_%d_te_indices' % trial_i]
    x_tr, y_tr = data['x_tr'][tr_index], data['y_tr'][tr_index]
    x_te, y_te = data['x_tr'][te_index], data['y_tr'][te_index]
    alpha1, threshold1, rankfeat1, run_time1 = rank_boost(np.asarray(x_tr), np.asarray(y_tr), t=t)
    scores1 = decision_func(x_tr, alpha1, threshold1, rankfeat1)
    alpha2, threshold2, rankfeat2, _, run_time2 = c_rank_boost(np.asarray(x_tr, dtype=np.float64),
                                                               np.asarray(y_tr, dtype=np.float64), int(t))
    alpha2, threshold2, rankfeat2 = np.asarray(alpha2), np.asarray(threshold2), np.asarray(rankfeat2, dtype=int)
    scores2 = decision_func(x_tr, alpha2, threshold2, rankfeat2)
    print(np.linalg.norm(scores1 - scores2), run_time2, run_time1 / run_time2)
    print(roc_auc_score(y_true=y_tr, y_score=scores1), end=' ')
    print(roc_auc_score(y_true=y_te, y_score=decision_func(x_te, alpha1, threshold1, rankfeat1)))
    print(roc_auc_score(y_true=y_tr, y_score=scores2), end=' ')
    print(roc_auc_score(y_true=y_te, y_score=decision_func(x_te, alpha2, threshold2, rankfeat2)))


def test_1():
    x_tr = [[2.5, 0.5],
            [3.5, 0.5],
            [4.5, 1.5],
            [0.5, 1.5],
            [0.5, 2.5],
            [1.5, 2.5],
            [2.5, 2.5]]
    x_tr, y_tr = np.asarray(x_tr), np.asarray([1., 1., 1., -1., -1., -1., -1.])
    t = 5
    alpha, threshold, rankfeat, run_time1 = rank_boost(np.asarray(x_tr), np.asarray(y_tr), t)
    scores = decision_func(x_tr, alpha, threshold, rankfeat)
    roc_auc_score(y_true=y_tr, y_score=scores)
    alpha, threshold, rankfeat, _, run_time = c_rank_boost(np.asarray(x_tr, dtype=np.float64),
                                                           np.asarray(y_tr, dtype=np.float64), int(t))
    scores_2 = decision_func(x_tr, np.asarray(alpha), np.asarray(threshold), np.asarray(rankfeat, dtype=int))
    print(np.linalg.norm(scores - scores_2))


def test_3():
    dataset = 'letter_a'
    trial_i = 3
    np.random.seed(trial_i)
    t = 100
    data = get_data(dataset=dataset, num_trials=200)
    tr_index = data['trial_%d_tr_indices' % trial_i]
    te_index = data['trial_%d_te_indices' % trial_i]
    x_tr, y_tr = data['x_tr'][tr_index], data['y_tr'][tr_index]
    x_te, y_te = data['x_tr'][te_index], data['y_tr'][te_index]
    alpha2, threshold2, rankfeat2, _, run_time2 = c_rank_boost(np.asarray(x_tr, dtype=np.float64),
                                                               np.asarray(y_tr, dtype=np.float64), int(t))
    alpha2, threshold2, rankfeat2 = np.asarray(alpha2), np.asarray(threshold2), np.asarray(rankfeat2, dtype=int)
    scores2 = decision_func(x_tr, alpha2, threshold2, rankfeat2)
    tr_auc = roc_auc_score(y_true=y_tr, y_score=scores2)
    te_auc = roc_auc_score(y_true=y_te, y_score=decision_func(x_te, alpha2, threshold2, rankfeat2))
    print('tr_auc: %.6f te_auc: %.6f run_time: %.6f' % (tr_auc, te_auc, run_time2))


def main():
    test_3()


if __name__ == '__main__':
    main()
