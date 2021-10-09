# -*- coding: utf-8 -*-
import os
import sys
import time
import operator
import warnings
import numpy as np
import pickle as pkl
import matplotlib as mpl
from itertools import product
import matplotlib.pyplot as plt
from test_auc_real import run_algo_opt_auc
from test_auc_real import run_algo_lr
from test_auc_real import run_algo_gb
from test_auc_real import run_algo_rf
from test_auc_real import run_algo_c_svm
from test_auc_real import run_algo_adaboost
from test_auc_real import run_algo_rank_boost
from test_auc_real import run_algo_rbf_svm
from test_auc_real import run_algo_spam_l2
from test_auc_real import run_algo_spauc_l2
from test_auc_real import run_algo_svm_perf
from test_auc_real import decision_func_rank_boost
from test_auc_real import decision_func_svm_perf
from data_preprocess import get_data
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
from sklearn.utils.testing import ignore_warnings
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


def get_z_scores(x_tr, h, results):
    x_min, x_max = x_tr[:, 0].min() - .5, x_tr[:, 0].max() + .5
    y_min, y_max = x_tr[:, 1].min() - .5, x_tr[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    z_scores = dict()
    for ind, method in enumerate(['opt_auc', 'lr', 'c_svm']):
        w = results[method]['w']
        if method == 'lr' or method == 'b_lr':
            w = results[method]['w']
        if method == 'c_svm' or method == 'b_c_svm':
            w = results[method]['w']
        z_values = np.dot(mesh_points, np.asarray(w)).reshape(xx.shape)
        range_z = np.max(z_values) - np.min(z_values)
        z_values = 2. * (z_values - np.min(z_values)) / range_z - 1.
        z_scores[method] = z_values
    return z_scores


def draw_heat_map_fig(results, x_tr, y_tr, x_te, y_te, flag='te', show_bar=False):
    (fig, ax), h = plt.subplots(1, 3, figsize=(14, 4)), 0.05
    fig.patch.set_visible(False)
    from matplotlib import rcParams
    rcParams['axes.titlepad'] = 5
    for i in product(range(3)):
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        if flag == 'te':
            ax[i].scatter(x_tr[:, 0][np.argwhere(y_tr > 0)],
                          x_tr[:, 1][np.argwhere(y_tr > 0)], c='r', s=2, label='posi')
            ax[i].scatter(x_tr[:, 0][np.argwhere(y_tr < 0)],
                          x_tr[:, 1][np.argwhere(y_tr < 0)], c='b', s=2, label='nega')
        else:
            ax[i].scatter(x_te[:, 0][np.argwhere(y_te > 0)],
                          x_te[:, 1][np.argwhere(y_te > 0)], c='r', s=2, label='posi')
            ax[i].scatter(x_te[:, 0][np.argwhere(y_te < 0)],
                          x_te[:, 1][np.argwhere(y_te < 0)], c='b', s=2, label='nega')
    x_min, x_max = x_tr[:, 0].min() - .5, x_tr[:, 0].max() + .5
    y_min, y_max = x_tr[:, 1].min() - .5, x_tr[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z_scores = get_z_scores(x_tr, h, results)
    label_list = ['LR', 'C-SVM', 'OptAUC']
    for ind, method in enumerate(['lr', 'c_svm', 'opt_auc']):
        z_values = z_scores[method]
        print(np.min(z_values), np.max(z_values))
        pp = np.arange(x_min, x_max, h)
        ax[ind].contourf(xx, yy, z_values, cmap=plt.cm.RdBu_r, alpha=1., levels=100, zorder=-1)
        ax[ind].contour(xx, yy, z_values, 100, linewidths=0.5, colors='k', alpha=0.1, zorder=-1)
        ax[ind].set_title('%s' % (label_list[ind]))
        if show_bar:
            plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=mpl.colors.Normalize(
                vmin=np.min(z_values), vmax=np.max(z_values))), ax=ax[0, ind], orientation="horizontal")
        ax[ind].set_xlim(xx.min(), xx.max())
        ax[ind].set_ylim(yy.min(), yy.max())
        ax[ind].set_xticks(())
        ax[ind].set_yticks(())

    file_name = 'heat_map_outlier.pdf'
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s' % file_name
    plt.subplots_adjust(wspace=0.01, hspace=0.15)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()


def test_on_simu2():
    file_name = '/data/auc-logistic/results/real/simu_heat_map.pkl'
    if os.path.exists(file_name):
        return pkl.load(open(file_name, 'rb'))
    dataset, num_trials = 'simu', 1
    data = get_data(dataset=dataset, num_trials=num_trials, split_ratio=0.5)
    trial_id = 0
    results = dict()
    results['opt_auc'] = run_algo_opt_auc((data, trial_id))[trial_id]
    results['lr'] = run_algo_lr((data, trial_id, None))[trial_id]
    results['b_lr'] = run_algo_lr((data, trial_id, 'balanced'))[trial_id]
    results['c_svm'] = run_algo_c_svm((data, trial_id, None))[trial_id]
    results['b_c_svm'] = run_algo_c_svm((data, trial_id, 'balanced'))[trial_id]
    results['gb'] = run_algo_gb((data, trial_id))[trial_id]
    results['rf'] = run_algo_rf((data, trial_id, None))[trial_id]
    results['adaboost'] = run_algo_adaboost((data, trial_id))[trial_id]
    results['rank_boost'] = run_algo_rank_boost((data, trial_id, 'rank_boost'))[trial_id]
    results['rbf_svm'] = run_algo_rbf_svm((data, trial_id, None))[trial_id]
    results['spam'] = run_algo_spam_l2((data, trial_id))[trial_id]
    results['spauc'] = run_algo_spauc_l2((data, trial_id))[trial_id]
    results['svm_perf_lin'] = run_algo_svm_perf((data, trial_id, 'linear'))[trial_id]
    results['svm_perf_rbf'] = run_algo_svm_perf((data, trial_id, 'rbf'))[trial_id]
    pkl.dump(results, open(file_name, 'wb'))
    return results


def run_algo_lr(x_tr, y_tr):
    list_c, k_fold, best_auc, w_lr, c_lr = np.logspace(-10, 10, 100), 5, -1., None, None
    for para_xi in list_c:
        lr = LogisticRegression(
            penalty='l2', dual=False, tol=1e-5, C=para_xi, fit_intercept=True,
            intercept_scaling=1, class_weight=None, random_state=0,
            solver='lbfgs', max_iter=5000, multi_class='auto', verbose=0,
            warm_start=False, n_jobs=None, l1_ratio=None)
        lr.fit(X=x_tr, y=y_tr)
        auc = roc_auc_score(y_true=y_tr, y_score=lr.decision_function(X=x_tr))
        if best_auc < auc:
            best_auc = auc
            w_lr = lr.coef_.flatten()
    return w_lr


def run_algo_c_svm(x_tr, y_tr):
    list_c, k_fold, best_auc, w_svm = np.logspace(-10, 10, 100), 5, -1., None
    for para_xi in list_c:
        std = StandardScaler().fit(X=x_tr)
        lin_svm = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-5,
                            C=para_xi, multi_class='ovr', fit_intercept=True,
                            intercept_scaling=1, class_weight=None, verbose=0,
                            random_state=0, max_iter=5000)
        lin_svm.fit(X=std.transform(X=x_tr), y=y_tr)
        auc = roc_auc_score(y_true=y_tr, y_score=lin_svm.decision_function(X=std.transform(X=x_tr)))
        if best_auc < auc:
            best_auc = auc
            w_svm = lin_svm.coef_.flatten()
    return w_svm


def test_on_simu():
    dataset, num_trials = 'simu', 1
    data = get_data(dataset=dataset, num_trials=num_trials, split_ratio=0.9)
    trial_id = 0
    results = {'opt_auc': dict(), 'lr': dict(), 'c_svm': dict()}
    x_tr, y_tr = data['x_tr'], data['y_tr']
    w_opt, auc, train_time = c_opt_auc(np.asarray(data['x_tr'], dtype=np.float64),
                                       np.asarray(data['y_tr'], dtype=np.float64), 2e-16)
    results['opt_auc']['w'] = w_opt
    results['lr']['w'] = run_algo_lr(x_tr, y_tr)
    results['c_svm']['w'] = run_algo_c_svm(x_tr, y_tr)
    tr_index = data['trial_%d_tr_indices' % trial_id]
    te_index = data['trial_%d_te_indices' % trial_id]
    x_te, y_te = data['x_tr'][te_index], data['y_tr'][te_index]
    x_tr, y_tr = data['x_tr'][tr_index], data['y_tr'][tr_index]
    draw_heat_map_fig(results, x_tr, y_tr, x_te, y_te)
    return results


def draw_results():
    file_name = '/data/auc-logistic/results/real/simu_heat_map.pkl'
    results = pkl.load(open(file_name, 'rb'))
    for method in results:
        print('%15s %.10f %.10f' % (method, results[method][method]['tr']['auc'],
                                    results[method][method]['te']['auc']))
    dataset, num_trials, trial_id = 'fourclass', 1, 0
    data = get_data(dataset=dataset, num_trials=num_trials, split_ratio=0.5)
    tr_index = data['trial_%d_tr_indices' % trial_id]
    te_index = data['trial_%d_te_indices' % trial_id]
    x_te, y_te = data['x_tr'][te_index], data['y_tr'][te_index]
    x_tr, y_tr = data['x_tr'][tr_index], data['y_tr'][tr_index]
    draw_heat_map_fig(results, x_tr, y_tr, x_te, y_te)


def main():
    test_on_simu()


if __name__ == '__main__':
    main()
