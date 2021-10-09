# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle as pkl
import multiprocessing
import subprocess
import tempfile
from itertools import product
from functools import reduce

from os.path import join
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from data_preprocess import get_data

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC

try:
    import libopt_auc_3

    try:
        from libopt_auc_3 import c_opt_auc

    except ImportError:
        print('cannot find some function(s) in opt_auc')
        exit(0)
except ImportError:
    pass

if os.uname()[1] == 'baojian-ThinkPad-T540p':
    root_path = '/data/auc-logistic/'
elif os.uname()[1] == 'pascal':
    root_path = '/mnt/store2/baojian/data/auc-logistic/'
elif os.uname()[1].endswith('.rit.albany.edu'):
    root_path = '/network/rit/lab/ceashpc/bz383376/data/auc-logistic/'

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def get_simu_data():
    np.random.seed(0)
    # interesting scale = 3.0,2.0
    x_nega = np.random.normal(loc=0.0, scale=2.0, size=(700, 2))
    x_posi = np.random.normal(loc=1.0, scale=2.0, size=(100, 2))

    scores_nega = np.asarray([1.4 * _[0] + _[1] - 2. for _ in x_nega])
    x_nega = np.asarray(x_nega[np.argwhere(scores_nega < 0).flatten()])
    scores_posi = np.asarray([1.4 * _[0] + _[1] - 1. for _ in x_posi])
    x_posi = np.asarray(x_posi[np.argwhere(scores_posi > 0).flatten()])
    x_outlier = np.random.normal(loc=10.0, scale=2.0, size=(200, 2))
    x_nega = np.concatenate((x_nega, x_outlier))
    x_tr = np.concatenate((x_posi, x_nega))
    y_tr = np.ones(len(x_tr))
    y_tr[len(x_posi):] *= -1.
    perm = np.random.permutation(len(x_tr))
    x_tr = x_tr[perm]
    y_tr = y_tr[perm]
    return x_tr, y_tr


def main():
    x_tr, y_tr = get_simu_data()
    x_min, x_max = x_tr[:, 0].min() - .5, x_tr[:, 0].max() + .5
    y_min, y_max = x_tr[:, 1].min() - .5, x_tr[:, 1].max() + .5
    pp = np.arange(x_min, x_max, 0.1)
    std = StandardScaler().fit(X=x_tr)
    w_opt, auc, train_time = c_opt_auc(np.asarray(std.transform(X=x_tr), dtype=np.float64),
                                       np.asarray(y_tr, dtype=np.float64), 2e-16)
    print('opt-auc', auc)
    list_c, k_fold, best_auc_lr, w_lr, c_lr = np.logspace(-6, 6, 200), 5, -1., None, None
    list_auc_lr = []
    for para_xi in list_c:
        lr = LogisticRegression(
            penalty='l2', dual=False, tol=1e-5, C=para_xi, fit_intercept=True,
            intercept_scaling=1, class_weight=None, random_state=0,
            solver='liblinear', max_iter=10000, multi_class='auto', verbose=0,
            warm_start=False, n_jobs=None, l1_ratio=None)
        lr.fit(X=std.transform(X=x_tr), y=y_tr)
        auc = roc_auc_score(y_true=y_tr, y_score=lr.decision_function(X=std.transform(X=x_tr)))
        list_auc_lr.append(auc)
        if best_auc_lr < auc:
            best_auc_lr = auc
            w_lr = lr.coef_.flatten()
    print('lr', best_auc_lr)
    list_c, k_fold, best_auc_svm, w_svm = np.logspace(-6, 6, 200), 5, -1., None
    list_auc_svm = []
    for para_xi in list_c:
        std = StandardScaler().fit(X=x_tr)
        lin_svm = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-5,
                            C=para_xi, multi_class='ovr', fit_intercept=True,
                            intercept_scaling=1, class_weight=None, verbose=0,
                            random_state=0, max_iter=10000)
        lin_svm.fit(X=std.transform(X=x_tr), y=y_tr)
        auc = roc_auc_score(y_true=y_tr, y_score=lin_svm.decision_function(X=std.transform(X=x_tr)))
        print(para_xi, auc)
        list_auc_svm.append(auc)
        if best_auc_svm < auc:
            best_auc_svm = auc
            w_svm = lin_svm.coef_.flatten()
    plt.plot(list_auc_svm, label='SVM')
    plt.plot(list_auc_lr, label='LR')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.scatter(x_tr[np.argwhere(y_tr > 0), 0], x_tr[np.argwhere(y_tr > 0), 1], c='b', marker='o', s=10)
    ax.scatter(x_tr[np.argwhere(y_tr < 0), 0], x_tr[np.argwhere(y_tr < 0), 1], c='r', s=10)
    ppx = np.arange(x_min, 5, 0.1)
    ax.plot(ppx, [-(w_opt[0] * _) / w_opt[1] for _ in ppx], linestyle='dashed', zorder=-1, label='LO-AUC')
    ax.plot(pp, [-(w_lr[0] * _) / w_lr[1] for _ in pp], linestyle='dashed', zorder=-1, label='LR')
    ppx = np.arange(-2, 1, 0.01)
    ax.plot(ppx, [-(w_svm[0] * _) / w_svm[1] for _ in ppx], linestyle='dashed', zorder=-1, label='SVM')
    print('svm', best_auc_svm)
    print('lr', best_auc_lr)
    plt.legend(fontsize=20)
    plt.show()


def run_model_compare_1():
    import warnings
    np.random.seed(1)
    x_nega = np.random.normal(loc=0.0, scale=2.0, size=(100, 2))
    x_posi = np.random.normal(loc=1.0, scale=2.0, size=(20, 2))
    scores_nega = np.asarray([1.4 * _[0] + _[1] - 2. for _ in x_nega])
    x_nega = np.asarray(x_nega[np.argwhere(scores_nega < 0).flatten()])
    scores_posi = np.asarray([1.4 * _[0] + _[1] - 1. for _ in x_posi])
    x_posi = np.asarray(x_posi[np.argwhere(scores_posi > 0).flatten()])
    x_outlier = np.random.normal(loc=10.0, scale=2.0, size=(20, 2))
    x_nega = np.concatenate((x_nega, x_outlier))
    x_tr = np.concatenate((x_posi, x_nega))
    y_tr = np.ones(len(x_tr))
    y_tr[len(x_posi):] *= -1.
    perm = np.random.permutation(len(x_tr))
    x_tr = x_tr[perm]
    y_tr = y_tr[perm]
    print(np.count_nonzero(y_tr > 0), np.count_nonzero(y_tr < 0))
    h = 0.1
    x_min, x_max = x_tr[:, 0].min() - .5, x_tr[:, 0].max() + .5
    pp = np.arange(x_min, x_max, h)
    std = StandardScaler().fit(X=x_tr)
    w_opt, auc, train_time = c_opt_auc(np.asarray(std.transform(X=x_tr), dtype=np.float64),
                                       np.asarray(y_tr, dtype=np.float64), 2e-16)
    print('opt-auc', auc, w_opt)
    list_lr_auc, list_lr_w, list_lr_intercept = [], [], []
    for para_xi in np.logspace(-10, 10, 200):
        lr = LogisticRegression(
            penalty='l2', dual=False, tol=1e-5, C=para_xi, fit_intercept=True,
            intercept_scaling=1, class_weight=None, random_state=0,
            solver='liblinear', max_iter=100, multi_class='auto', verbose=0,
            warm_start=False, n_jobs=None, l1_ratio=None)
        lr.fit(X=std.transform(X=x_tr), y=y_tr)
        list_lr_auc.append(roc_auc_score(y_true=y_tr, y_score=lr.decision_function(X=std.transform(X=x_tr))))
        list_lr_w.append(lr.coef_.flatten())
        list_lr_intercept.append(lr.intercept_)
    print('lr', max(list_lr_auc), np.argmax(list_lr_auc),
          list_lr_w[int(np.argmax(list_lr_auc))], list_lr_intercept[int(np.argmax(list_lr_auc))])
    list_svm_auc, list_svm_w, list_svm_intercept = [], [], []
    for para_xi in np.logspace(-10, 10, 200):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            std = StandardScaler().fit(X=x_tr)
            lin_svm = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-5,
                                C=para_xi, multi_class='ovr', fit_intercept=True,
                                intercept_scaling=1, class_weight=None, verbose=0,
                                random_state=0, max_iter=100)
            lin_svm.fit(X=std.transform(X=x_tr), y=y_tr)
            list_svm_auc.append(roc_auc_score(
                y_true=y_tr, y_score=lin_svm.decision_function(X=std.transform(X=x_tr))))
            list_svm_w.append(lin_svm.coef_.flatten())
            list_svm_intercept.append(lin_svm.intercept_)
    w_svm = list_svm_w[int(np.argmax(list_svm_auc))]
    w_lr = list_lr_w[int(np.argmax(list_lr_auc))]
    print('svm', max(list_svm_auc), np.argmax(list_svm_auc),
          list_svm_w[int(np.argmax(list_svm_auc))], list_svm_intercept[int(np.argmax(list_svm_auc))])
    plt.plot(list_svm_auc, label='SVM')
    plt.plot(list_lr_auc, label='LR')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.scatter(x_nega[:, 0], x_nega[:, 1], c='b', marker='o', s=10)
    ax.scatter(x_posi[:, 0], x_posi[:, 1], c='r', s=10)
    ppx = np.arange(x_min, 5, h)
    ax.plot(ppx, [-(w_opt[0] * _) / w_opt[1] for _ in ppx], linestyle='dashed', zorder=-1, label='LO-AUC')
    ax.plot(pp, [-(w_lr[0] * _) / w_lr[1] for _ in pp], linestyle='dashed', zorder=-1, label='LR')
    ppx = np.arange(-5., 15., 0.01)
    ax.plot(ppx, [-(w_svm[0] * _) / w_svm[1] for _ in ppx], linestyle='dashed', zorder=-1, label='SVM')
    plt.legend(fontsize=20)
    plt.show()


def cal_best_threshold_tr(y_tr, scores):
    """
    Chosen the threshold for prediction function by
    using balanced accuracy score.
    :param y_tr:
    :param scores:
    :return:
    """
    fpr, tpr, thresholds = roc_curve(y_true=y_tr, y_score=scores)
    y_pred = np.zeros_like(y_tr)
    n_posi, n_nega = len([_ for _ in y_tr if _ > 0]), len([_ for _ in y_tr if _ < 0])
    best_b_acc, best_f1, best_threshold, best_acc, best_am, best_loss = -1., -1., -1.0, -1.0, -1., np.inf
    for fpr_, tpr_, threshold in zip(fpr, tpr, thresholds):
        y_pred[np.argwhere(scores < threshold)] = -1
        y_pred[np.argwhere(scores >= threshold)] = 1
        b_acc = balanced_accuracy_score(y_true=y_tr, y_pred=y_pred)
        if best_b_acc < b_acc:
            best_b_acc = b_acc
            best_f1 = f1_score(y_true=y_tr, y_pred=y_pred)
            best_acc = accuracy_score(y_true=y_tr, y_pred=y_pred)
            confusion_mat = confusion_matrix(y_true=y_tr, y_pred=y_pred)
            best_threshold = threshold
            best_loss = log_loss(y_true=y_tr, y_pred=y_pred)
            best_am = (confusion_mat[0][0] / n_nega + confusion_mat[1][1] / n_posi) / 2.
    auc = roc_auc_score(y_true=y_tr, y_score=scores)
    return auc, best_b_acc, best_f1, best_acc, best_am, best_loss, best_threshold


def get_line(x_min, x_max, y_min, y_max, h, w):
    ppx = np.arange(x_min, x_max, h)
    ppy = [-(w[0] * _) / w[1] for _ in ppx]
    xx, yy = [], []
    for ii, jj in zip(ppx, ppy):
        if x_min <= ii <= x_max and y_min <= jj <= y_max:
            xx.append(ii)
            yy.append(jj)
    return xx, yy


def run_test(num_):
    import warnings
    np.random.seed(1)
    x_nega = np.random.normal(loc=0.0, scale=2.0, size=(700, 2))
    x_posi = np.random.normal(loc=1.0, scale=2.0, size=(100, 2))
    scores_nega = np.asarray([1.4 * _[0] + _[1] - 2. for _ in x_nega])
    x_nega = np.asarray(x_nega[np.argwhere(scores_nega < 0).flatten()])
    scores_posi = np.asarray([1.4 * _[0] + _[1] - 1. for _ in x_posi])
    x_posi = np.asarray(x_posi[np.argwhere(scores_posi > 0).flatten()])
    x_outlier = np.random.normal(loc=10.0, scale=2.0, size=(num_, 2))
    x_nega = np.concatenate((x_nega, x_outlier))
    x_tr = np.concatenate((x_posi, x_nega))
    y_tr = np.ones(len(x_tr))
    y_tr[len(x_posi):] *= -1.
    perm = np.random.permutation(len(x_tr))
    x_tr = x_tr[perm]
    y_tr = y_tr[perm]
    print(np.count_nonzero(y_tr > 0), np.count_nonzero(y_tr < 0))
    x_min, x_max = x_tr[:, 0].min() - .5, x_tr[:, 0].max() + .5
    y_min, y_max = x_tr[:, 1].min() - .5, x_tr[:, 1].max() + .5
    w_opt, auc, train_time = c_opt_auc(np.asarray(x_tr, dtype=np.float64), np.asarray(y_tr, dtype=np.float64), 2e-16)
    auc, best_b_acc, best_f1, best_acc, _, _, _ = cal_best_threshold_tr(y_tr, np.dot(x_tr, w_opt))
    print('opt-auc', auc, best_b_acc, best_f1, w_opt)
    list_lr_auc, list_lr_w, list_lr_intercept, list_lr_acc, list_lr_f1 = [], [], [], [], []
    for para_xi in np.logspace(-10, 10, 100):
        lr = LogisticRegression(
            penalty='l2', dual=False, tol=1e-5, C=para_xi, fit_intercept=True,
            intercept_scaling=1, class_weight=None, random_state=0,
            solver='liblinear', max_iter=100, multi_class='auto', verbose=0,
            warm_start=False, n_jobs=None, l1_ratio=None)
        lr.fit(X=x_tr, y=y_tr)
        list_lr_auc.append(roc_auc_score(y_true=y_tr, y_score=lr.decision_function(X=x_tr)))
        list_lr_w.append(lr.coef_.flatten())
        list_lr_intercept.append(lr.intercept_)
        list_lr_acc.append(balanced_accuracy_score(y_true=y_tr, y_pred=lr.predict(X=x_tr)))
        list_lr_f1.append(f1_score(y_true=y_tr, y_pred=lr.predict(X=x_tr)))
    print('lr', max(list_lr_auc), list_lr_acc[int(np.argmax(list_lr_auc))],
          list_lr_f1[int(np.argmax(list_lr_auc))],
          list_lr_w[int(np.argmax(list_lr_auc))],
          list_lr_intercept[int(np.argmax(list_lr_auc))])
    list_svm_auc, list_svm_w, list_svm_intercept, list_svm_acc, list_svm_f1 = [], [], [], [], []
    for para_xi in np.logspace(-10, 10, 100):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            std = StandardScaler().fit(X=x_tr)
            lin_svm = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-5,
                                C=para_xi, multi_class='ovr', fit_intercept=True,
                                intercept_scaling=1, class_weight=None, verbose=0,
                                random_state=0, max_iter=100)
            lin_svm.fit(X=std.transform(X=x_tr), y=y_tr)
            list_svm_auc.append(roc_auc_score(
                y_true=y_tr, y_score=lin_svm.decision_function(X=std.transform(X=x_tr))))
            list_svm_w.append(lin_svm.coef_.flatten())
            list_svm_intercept.append(lin_svm.intercept_)
            list_svm_acc.append(balanced_accuracy_score(y_true=y_tr, y_pred=lin_svm.predict(X=x_tr)))
            list_svm_f1.append(f1_score(y_true=y_tr, y_pred=lin_svm.predict(X=x_tr)))
    w_svm = list_svm_w[int(np.argmax(list_svm_auc))]
    w_lr = list_lr_w[int(np.argmax(list_lr_auc))]
    print('svm', max(list_svm_auc),
          list_svm_acc[int(np.argmax(list_svm_auc))],
          list_svm_f1[int(np.argmax(list_svm_auc))],
          list_svm_w[int(np.argmax(list_svm_auc))],
          list_svm_intercept[int(np.argmax(list_svm_auc))])
    return x_tr, y_tr, x_min, x_max, y_min, y_max, w_opt, w_lr, w_svm, auc, list_lr_auc, list_svm_auc, x_outlier


def run_model_compare_2():
    fig, ax = plt.subplots(2, 6, figsize=(24, 6))
    for ind, num_ in enumerate([1, 10, 20, 30, 40, 50]):
        x_tr, y_tr, x_min, x_max, y_min, y_max, w_opt, \
        w_lr, w_svm, auc, list_lr_auc, list_svm_auc, x_outlier = run_test(num_)
        ax[0, ind].scatter(x_tr[np.argwhere(y_tr < 0), 0], x_tr[np.argwhere(y_tr < 0), 1], c='b', marker='o', s=10,
                           label=np.count_nonzero(y_tr < 0))
        ax[0, ind].scatter(x_tr[np.argwhere(y_tr > 0), 0], x_tr[np.argwhere(y_tr > 0), 1], c='r', s=10,
                           label=np.count_nonzero(y_tr > 0))
        h = 0.001
        xx, yy = get_line(x_min, x_max, y_min, y_max, h, w_opt)
        ax[0, ind].plot(xx, yy, linestyle='dashed', zorder=-1, label='LO-AUC')
        xx, yy = get_line(x_min, x_max, y_min, y_max, h, w_lr)
        ax[0, ind].plot(xx, yy, linestyle='dashed', zorder=-1, label='LR')
        xx, yy = get_line(x_min, x_max, y_min, y_max, h, w_svm)
        ax[0, ind].plot(xx, yy, linestyle='dashed', zorder=-1, label='SVM')
        ax[0, ind].legend(fontsize=8)
        fpr, tpr, _ = roc_curve(y_true=y_tr, y_score=np.dot(x_tr, w_opt))
        ax[1, ind].plot(fpr, tpr, label='LO-AUC:  %.6f' % auc)
        fpr, tpr, _ = roc_curve(y_true=y_tr, y_score=np.dot(x_tr, w_lr))
        ax[1, ind].plot(fpr, tpr, label='LR:          %.6f' % max(list_lr_auc))
        fpr, tpr, _ = roc_curve(y_true=y_tr, y_score=np.dot(x_tr, w_svm))
        ax[1, ind].plot(fpr, tpr, label='SVM:       %.6f' % max(list_svm_auc))
        ax[1, ind].plot(np.arange(0., 1.01, 0.1), np.arange(0., 1.01, 0.1), c='gray', label='Random: 0.5')
        xx = ax[1, ind].legend(fontsize=8)
        xx._legend_box.align = "right"
        ax[1, 0].set_ylabel('AUC')
        num_outliers = len(x_outlier)
    file_name = 'linear_classifers_100_1.pdf'
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s' % file_name
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()


def run_model_compare_3():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    list_auc_opt = []
    list_auc_lr = []
    list_auc_svm = []
    xx = [1, 10, 20, 30, 40, 50, 80, 100, 120, 140, 150, 160, 180, 200, 220, 240, 260, 280, 300, 350, 400, 450, 500,
          550, 600]
    for ind, num_ in enumerate(xx):
        x_tr, y_tr, x_min, x_max, y_min, y_max, w_opt, \
        w_lr, w_svm, auc, list_lr_auc, list_svm_auc, x_outlier = run_test(num_)
        list_auc_opt.append(auc)
        list_auc_lr.append(max(list_lr_auc))
        list_auc_svm.append(max(list_svm_auc))
    plt.plot([70. / (_ + 500.) for _ in xx][::-1], list_auc_opt, label='LO-AUC')
    plt.plot([70. / (_ + 500.) for _ in xx][::-1], list_auc_lr, label='LR')
    plt.plot([70. / (_ + 500.) for _ in xx][::-1], list_auc_svm, label='SVM')
    plt.ylabel('AUC')
    plt.xlabel('Positive-Ratio')
    plt.legend()
    file_name = 'linear_classifers_100_curve.pdf'
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s' % file_name
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()


if __name__ == '__main__':
    run_model_compare_3()
