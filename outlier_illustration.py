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
    sys.path.append(os.getcwd())
    import libspam_l2

    try:
        from libspam_l2 import c_algo_spam
    except ImportError:
        print('cannot find some function(s) in auc_module')
        exit(0)
    sys.path.append(os.getcwd())
except ImportError:
    pass

try:
    sys.path.append(os.getcwd())
    import libspam_l2

    try:
        from libspauc_l2 import c_algo_spauc
    except ImportError:
        print('cannot find some function(s) in auc_module')
        exit(0)
    sys.path.append(os.getcwd())
except ImportError:
    pass

try:
    import libopt_auc_3

    try:
        from libopt_auc_3 import c_opt_auc

    except ImportError:
        print('cannot find some function(s) in opt_auc')
        exit(0)
except ImportError:
    pass
try:
    import librank_boost_3

    try:
        from librank_boost_3 import c_rank_boost

    except ImportError:
        print('cannot find some function(s) in rank_boost')
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


def cmd_svm_perf(sub_x_tr, sub_y_tr, sub_x_te, sub_y_te, para_xi, kernel):
    f_tr = tempfile.NamedTemporaryFile()
    f_te = tempfile.NamedTemporaryFile()
    for index, item in enumerate(sub_x_tr):
        str_ = [b'%d' % int(sub_y_tr[index])]
        for index_2, val in enumerate(item):
            str_.append(b'%d:%.15f' % (index_2 + 1, val))
        str_ = b' '.join(str_) + b'\n'
        f_tr.write(str_)
    for index, item in enumerate(sub_x_te):
        str_ = [b'%d' % int(sub_y_te[index])]
        for index_2, val in enumerate(item):
            str_.append(b'%d:%.15f' % (index_2 + 1, val))
        str_ = b' '.join(str_) + b'\n'
        f_te.write(str_)
    f_tr.seek(0)
    f_te.seek(0)
    file_model = tempfile.NamedTemporaryFile()
    file_pred = tempfile.NamedTemporaryFile()
    if kernel == 'linear':
        # We set the number of terminate QP sub-problem is 10000.
        learn_cmd = "./svm_perf/svm_perf_learn -v 3 -y 3 -w 9 -t 0 -c %.10f --b 1 -# 10000 %s %s"
        os.system(learn_cmd % (para_xi, f_tr.name, file_model.name))
        file_model.seek(0)
    else:
        learn_cmd = "./svm_perf/svm_perf_learn -v 0 -y 0 -w 3 --i 0 -t 2 -c %.10f --b 0 --k 500 -# 10000 %s %s"
        os.system(learn_cmd % (para_xi, f_tr.name, file_model.name))
        file_model.seek(0)
    pred_cmd = "./svm_perf/svm_perf_classify -v 0 -y 0  %s %s %s"
    os.system(pred_cmd % (f_tr.name, file_model.name, file_pred.name))
    file_pred.seek(0)
    tr_scores = [float(_) for _ in file_pred.readlines()]
    file_model.seek(0)
    f_tr.seek(0)
    file_pred.seek(0)
    os.system(pred_cmd % (f_te.name, file_model.name, file_pred.name))
    file_pred.seek(0)
    te_scores = [float(_) for _ in file_pred.readlines()]
    file_model.seek(0)
    model = file_model.readlines()
    file_pred.close()
    file_model.close()
    f_tr.close()
    f_te.close()
    return tr_scores, te_scores, model


def test_2():
    np.random.seed(1)
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
    h = 0.1
    x_min, x_max = x_tr[:, 0].min() - .5, x_tr[:, 0].max() + .5
    pp = np.arange(x_min, x_max, h)
    std = StandardScaler().fit(X=x_tr)
    w_opt, auc, train_time = c_opt_auc(np.asarray(std.transform(X=x_tr), dtype=np.float64),
                                       np.asarray(y_tr, dtype=np.float64), 2e-16)
    print('opt-auc', auc)
    list_c, k_fold, best_auc, w_lr, c_lr = np.logspace(-6, 10, 100), 5, -1., None, None
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
        if best_auc < auc:
            best_auc = auc
            w_lr = lr.coef_.flatten()
    print('lr', best_auc)
    list_c, k_fold, best_auc, w_svm = np.logspace(-10, 10, 100), 5, -1., None
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
        if best_auc < auc:
            best_auc = auc
            w_svm = lin_svm.coef_.flatten()
    plt.plot(list_auc_svm, label='SVM')
    plt.plot(list_auc_lr, label='LR')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.scatter(x_nega[:, 0], x_nega[:, 1], c='b', marker='o', s=10)
    ax.scatter(x_posi[:, 0], x_posi[:, 1], c='r', s=10)
    ppx = np.arange(x_min, 5, h)
    ax.plot(ppx, [-(w_opt[0] * _) / w_opt[1] for _ in ppx], linestyle='dashed', zorder=-1, label='LO-AUC')
    ax.plot(pp, [-(w_lr[0] * _) / w_lr[1] for _ in pp], linestyle='dashed', zorder=-1, label='LR')
    ppx = np.arange(-.07, 0.04, 0.0001)
    ax.plot(ppx, [-(w_svm[0] * _) / w_svm[1] for _ in ppx], linestyle='dashed', zorder=-1, label='SVM')
    print('svm', best_auc)
    plt.legend(fontsize=20)


def get_data():
    np.random.seed(1)
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
    h = 0.1
    x_tr = StandardScaler().fit_transform(X=x_tr)
    x_min, x_max = x_tr[:, 0].min() - .1, x_tr[:, 0].max() + .1
    y_min, y_max = x_tr[:, 0].min() - .1, x_tr[:, 0].max() + .1
    pp = np.arange(x_min, x_max, h)
    return x_tr, y_tr, pp, x_min, x_max, y_min, y_max, h


def run_outlier_test1():
    x_tr, y_tr, pp, x_min, x_max, h = get_data()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(x_tr[np.argwhere(y_tr < 0), 0], x_tr[np.argwhere(y_tr < 0), 1], c='b', marker='o', s=10)
    ax[0].scatter(x_tr[np.argwhere(y_tr > 0), 0], x_tr[np.argwhere(y_tr > 0), 1], c='r', s=10)
    w_opt, auc, train_time = c_opt_auc(np.asarray(x_tr, dtype=np.float64),
                                       np.asarray(y_tr, dtype=np.float64), 2e-16)
    print('opt-auc', auc)
    list_c, k_fold, best_auc, w_lr, c_lr = np.logspace(-10, 10, 100), 5, -1., None, None
    for para_xi in list_c:
        lr = LogisticRegression(
            penalty='l2', dual=False, tol=1e-5, C=para_xi, fit_intercept=True,
            intercept_scaling=1, class_weight=None, random_state=0,
            solver='liblinear', max_iter=10000, multi_class='auto', verbose=0,
            warm_start=False, n_jobs=None, l1_ratio=None)
        lr.fit(X=x_tr, y=y_tr)
        auc = roc_auc_score(y_true=y_tr, y_score=lr.decision_function(X=x_tr))
        if best_auc < auc:
            best_auc = auc
            w_lr = lr.coef_.flatten()
    print('lr', best_auc)
    list_c, k_fold, best_auc, w_svm = np.logspace(-10, 10, 100), 5, -1., None
    for para_xi in list_c:
        lin_svm = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-5,
                            C=para_xi, multi_class='ovr', fit_intercept=True,
                            intercept_scaling=1, class_weight=None, verbose=0,
                            random_state=0, max_iter=10000)
        lin_svm.fit(X=x_tr, y=y_tr)
        auc = roc_auc_score(y_true=y_tr, y_score=lin_svm.decision_function(X=x_tr))
        if best_auc < auc:
            best_auc = auc
            w_svm = lin_svm.coef_.flatten()
    ppx = np.arange(x_min, x_max, h)
    ax[0].plot(ppx, [-(w_opt[0] * _) / w_opt[1] for _ in ppx], linestyle='dashed', zorder=-1, label='LO-AUC')
    ax[0].plot(pp, [-(w_lr[0] * _) / w_lr[1] for _ in pp], linestyle='dashed', zorder=-1, label='LR')
    ax[0].plot(ppx, [-(w_svm[0] * _) / w_svm[1] for _ in ppx], linestyle='dashed', zorder=-1, label='SVM')
    print('svm', best_auc)
    ax[0].legend(fontsize=20)

    file_name = 'outlier_plot.pdf'
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s' % file_name
    plt.subplots_adjust(wspace=0.01, hspace=0.15)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()
    return x_tr, y_tr


def run_outlier_test2():
    x_tr, y_tr, pp, x_min, x_max, y_min, y_max, h = get_data()
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
    ax1.scatter(x_tr[np.argwhere(y_tr < 0), 0], x_tr[np.argwhere(y_tr < 0), 1],
                c='b', marker='_', s=10, alpha=0.8)
    ax1.scatter(x_tr[np.argwhere(y_tr > 0), 0], x_tr[np.argwhere(y_tr > 0), 1],
                c='r', marker='+', s=15, alpha=0.8)
    w_opt, auc, train_time = c_opt_auc(np.asarray(x_tr, dtype=np.float64),
                                       np.asarray(y_tr, dtype=np.float64), 2e-16)
    print('opt-auc', auc)
    fpr, tpr, _ = roc_curve(y_true=y_tr, y_score=np.dot(x_tr, w_opt))
    ax2.plot(fpr, tpr, label='AUC-opt (%.3f)' % auc, color='tab:red')
    list_c, k_fold, best_auc, w_lr, c_lr = np.logspace(-10, 10, 50), 5, -1., None, None
    for para_xi in list_c:
        lr = LogisticRegression(
            penalty='l2', dual=False, tol=1e-5, C=para_xi, fit_intercept=True,
            intercept_scaling=1, class_weight='balanced', random_state=0,
            solver='liblinear', max_iter=10000, multi_class='auto', verbose=0,
            warm_start=False, n_jobs=None, l1_ratio=None)
        lr.fit(X=x_tr, y=y_tr)
        auc = roc_auc_score(y_true=y_tr, y_score=lr.decision_function(X=x_tr))
        if best_auc < auc:
            best_auc = auc
            w_lr = lr.coef_.flatten()
    fpr, tpr, _ = roc_curve(y_true=y_tr, y_score=np.dot(x_tr, w_lr))
    ax2.plot(fpr, tpr, label='LR (%.3f)' % best_auc, color='tab:blue')
    print('lr', best_auc)
    list_c, k_fold, best_auc, w_svm = np.logspace(-6, 4, 50), 5, -1., None
    for para_xi in list_c:
        tr_scores, te_scores, model = cmd_svm_perf(x_tr, y_tr, x_tr, y_tr, para_xi, 'linear')
        auc = roc_auc_score(y_true=y_tr, y_score=tr_scores)
        print(para_xi, auc)
        if best_auc < auc:
            best_auc = auc
            w_svm = [float(_.split(b':')[1]) for _ in model[-1].split(b' ')[1:3]]
    fpr, tpr, _ = roc_curve(y_true=y_tr, y_score=np.dot(x_tr, w_svm))
    ax2.plot(fpr, tpr, label='SVM-Perf (%.3f)' % best_auc, color='tab:green')
    ax2.plot([0, 1], [0, 1], label='Random', color='lightgray', linestyle=':')
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ppx = np.arange(x_min, x_max, h)
    xx, yy = [], []
    for _ in ppx:
        if y_min <= (-(w_opt[0] * _) / w_opt[1] - 1.) <= y_max:
            xx.append(_)
            yy.append(-(w_opt[0] * _) / w_opt[1] - 1.)
    ax1.plot(xx, yy, linestyle='-', zorder=-1, label='AUC-opt', color='tab:red')
    xx, yy = [], []
    for _ in ppx:
        if y_min <= -(w_svm[0] * _) / w_svm[1] <= y_max:
            xx.append(_)
            yy.append(-(w_svm[0] * _) / w_svm[1])
    ax1.plot(xx, yy, linestyle='--', zorder=-1, label='SVM-Perf', color='tab:green')
    xx, yy = [], []
    for _ in ppx:
        if y_min <= -(w_lr[0] * _) / w_lr[1] <= y_max:
            xx.append(_)
            yy.append(-(w_lr[0] * _) / w_lr[1])
    ax1.plot(xx, yy, linestyle='dotted', zorder=-1, label='LR', color='tab:blue')
    print('svm', best_auc)
    ax1.legend(loc='upper left', frameon=False, fontsize=8)
    ax2.legend(loc='lower right', frameon=False, fontsize=8)
    ax1.tick_params(axis='y', direction='in')
    ax1.tick_params(axis='x', direction='in')
    ax2.tick_params(axis='y', direction='in')
    ax2.tick_params(axis='x', direction='in')
    ax1.set_xticks([])
    ax1.set_yticks([])

    file_name = 'example_plot_data.pdf'
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s' % file_name
    fig1.subplots_adjust(wspace=0.02, hspace=0.02)
    fig1.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()
    file_name = 'example_plot_auc.pdf'
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s' % file_name
    fig2.subplots_adjust(wspace=0.02, hspace=0.02)
    fig2.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()
    return x_tr, y_tr


def main():
    run_outlier_test2()


if __name__ == '__main__':
    main()
