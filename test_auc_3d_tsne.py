# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils.extmath import softmax
import multiprocessing
import pickle as pkl

try:
    from libopt_auc_3 import c_opt_auc
except ImportError:
    pass

if os.uname()[1] == 'baojian-ThinkPad-T540p':
    root_path = '/data/auc-logistic/'
elif os.uname()[1] == 'pascal':
    root_path = '/mnt/store2/baojian/data/auc-logistic/'
elif os.uname()[1].endswith('.rit.albany.edu'):
    root_path = '/network/rit/lab/ceashpc/bz383376/data/auc-logistic/'


def run_algo_opt_auc(x_tr, y_tr, x_te, y_te, rand_state, results):
    """
    rand_state : fix the rand_state if there is any.
    """
    np.random.seed(rand_state)
    w, auc, train_time = c_opt_auc(np.asarray(x_tr, dtype=np.float64), np.asarray(y_tr, dtype=np.float64), 2e-16)
    scores = np.dot(x_tr, w)
    auc = roc_auc_score(y_true=y_tr, y_score=scores)
    b_acc, f1, threshold = cal_best_threshold(y_tr=y_tr, scores=scores)
    loss = log_loss(y_true=y_tr, y_pred=softmax(np.c_[-scores, scores]))
    results['opt-auc']['tr'] = {'auc': auc, 'b_acc': b_acc, 'f1': f1, 'loss': loss, 'train_time': train_time}
    results['opt-auc']['w'] = w
    results['opt-auc']['threshold'] = threshold
    print('Opt-AUC on train auc: %.6f acc: %.6f f1: %.6f loss: %.6f train_time: %.2f'
          % (auc, b_acc, f1, results['opt-auc']['tr']['loss'], train_time), flush=True)
    test_time = time.time()
    scores = np.dot(x_te, w)
    y_pred = [1. if _ >= threshold else -1. for _ in scores]
    te_re = {'auc': roc_auc_score(y_true=y_te, y_score=scores),
             'b_acc': balanced_accuracy_score(y_true=y_te, y_pred=y_pred),
             'f1': f1_score(y_true=y_te, y_pred=y_pred),
             'loss': log_loss(y_true=y_te, y_pred=softmax(np.c_[-scores, scores]))}
    results['opt-auc']['te'] = te_re
    print('Opt-AUC on test auc: %.6f acc: %.6f f1: %.6f loss: %.6f test_time: %.2f'
          % (results['opt-auc']['te']['auc'], results['opt-auc']['te']['b_acc'],
             results['opt-auc']['te']['f1'], results['opt-auc']['te']['loss'], time.time() - test_time), flush=True)
    return results
