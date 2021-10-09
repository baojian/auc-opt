# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import pickle as pkl
import multiprocessing
from os.path import join
from scipy import stats
from itertools import product
import subprocess

from data_preprocess import data_preprocess_australian
from data_preprocess import data_preprocess_diabetes
from data_preprocess import data_preprocess_breast_cancer
from data_preprocess import data_preprocess_splice
from data_preprocess import data_preprocess_ijcnn1
from data_preprocess import data_preprocess_mushrooms
from data_preprocess import data_preprocess_svmguide3
from data_preprocess import data_preprocess_german
from data_preprocess import data_preprocess_fourclass
from data_preprocess import data_preprocess_a9a
from data_preprocess import data_preprocess_w8a
from data_preprocess import data_preprocess_spambase
from data_preprocess import data_preprocess_yeast

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_spam
        from sparse_module import c_algo_spauc
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')
if os.uname()[1] == 'baojian-ThinkPad-T540p':
    root_path = '/network/rit/lab/ceashpc/bz383376/data/auc-logistic/'
elif os.uname()[1] == 'pascal':
    root_path = '/mnt/store2/baojian/data/auc-logistic/'
elif os.uname()[1].endswith('.rit.albany.edu'):
    root_path = '/network/rit/lab/ceashpc/bz383376/data/auc-logistic/'


def get_from_spam_l2(dataset, num_trials):
    if os.path.exists(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, 'spam_l2')):
        results = pkl.load(open(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, 'spam_l2')))
    else:
        results = []
    para_xi_list = np.zeros(num_trials)
    para_l2_list = np.zeros(num_trials)
    for result in results:
        trial_i, (para_xi, para_l2), cv_res, wt, aucs, rts, iters, online_aucs, metrics = result
        para_xi_list[trial_i] = para_xi
        para_l2_list[trial_i] = para_l2
    return para_xi_list, para_l2_list


def generate_data(file_tr, file_te, x_tr, y_tr, x_te, y_te):
    if not os.path.exists(file_tr):
        f = open(file_tr, 'wb')
        for index, item in enumerate(x_tr):
            str_ = [b'%d' % int(y_tr[index])]
            for index_2, val in enumerate(item):
                str_.append(b'%d:%f' % (index_2 + 1, val))
            str_ = b' '.join(str_) + b'\n'
            f.write(str_)
        f.close()
    if not os.path.exists(file_te):
        f = open(file_te, 'wb')
        for index, item in enumerate(x_te):
            str_ = [b'%d' % int(y_te[index])]
            for index_2, val in enumerate(item):
                str_.append(b'%d:%f' % (index_2 + 1, val))
            str_ = b' '.join(str_) + b'\n'
            f.write(str_)
        f.close()


def test_logistic(para):
    data, trial_id = para
    list_c, k_fold = np.logspace(-4, 4, 10), 5
    results = dict()
    start_time = time.time()
    tr_index = data['trial_%d_tr_indices' % trial_id]
    te_index = data['trial_%d_te_indices' % trial_id]
    auc_matrix = np.zeros(shape=(len(list_c), k_fold))
    for (ind_xi, para_xi) in enumerate(list_c):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = data['x_tr'][tr_index[sub_tr_ind]]
            sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            lr = LogisticRegression(penalty='l2', dual=False, tol=1e-6, C=para_xi,
                                    fit_intercept=True, intercept_scaling=1,
                                    solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0)
            lr.fit(X=sub_x_tr, y=sub_y_tr)
            wt = lr.coef_[0]
            auc = 0.0 if np.isnan(wt).any() else roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            auc_matrix[ind_xi][ind] = auc
    max_auc_ind = np.argmax(np.mean(auc_matrix, axis=1))
    best_c = list_c[int(max_auc_ind)]
    lr = LogisticRegression(penalty='l2', dual=False, tol=1e-6, C=best_c,
                            fit_intercept=True, intercept_scaling=1,
                            solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0)
    lr.fit(X=data['x_tr'][tr_index], y=data['y_tr'][tr_index])

    wt, x_te, y_te = lr.coef_[0], data['x_tr'][te_index], data['y_tr'][te_index]
    x_tr, y_tr = data['x_tr'][tr_index], data['y_tr'][tr_index]
    y_tr_scores, y_te_scores = np.dot(x_tr, wt), np.dot(x_te, wt)
    y_pred_tr = lr.predict(X=x_tr)
    auc_tr = roc_auc_score(y_true=y_tr, y_score=y_tr_scores)
    f1_tr = f1_score(y_true=y_tr, y_pred=y_pred_tr)
    acc_tr = accuracy_score(y_true=y_tr, y_pred=y_pred_tr)
    pre_tr = precision_score(y_true=y_tr, y_pred=y_pred_tr)
    rec_tr = recall_score(y_true=y_tr, y_pred=y_pred_tr)
    aver_pre_tr = average_precision_score(y_true=y_tr, y_score=y_tr_scores)

    y_pred_te = lr.predict(X=x_te)
    auc_te = roc_auc_score(y_true=y_te, y_score=y_te_scores)
    f1_te = f1_score(y_true=y_te, y_pred=y_pred_te)
    acc_te = accuracy_score(y_true=y_te, y_pred=y_pred_te)
    pre_te = precision_score(y_true=y_te, y_pred=y_pred_te)
    rec_te = recall_score(y_true=y_te, y_pred=y_pred_te)
    aver_pre_te = average_precision_score(y_true=y_te, y_score=y_te_scores)
    results[trial_id] = {
        'trial_%03d': trial_id,
        'wt': wt,
        'para': [best_c, 'squared_hinge', 'l2', 'max_iter:1000', 'tol:1e-4'],
        'tr': {'y_pred': y_pred_tr, 'auc': auc_tr, 'f1': f1_tr,
               'acc': acc_tr, 'pre': pre_tr, 'rec': rec_tr, 'aver_pre': aver_pre_tr},
        'te': {'y_pred': y_pred_te, 'auc': auc_te, 'f1': f1_te,
               'acc': acc_te, 'pre': pre_te, 'rec': rec_te, 'aver_pre': aver_pre_te}}
    print(trial_id, time.time() - start_time)
    return results


def test_svm_perf(para):
    data, trial_id = para
    list_c, k_fold = np.logspace(-4, 4, 10), 5
    results = dict()
    start_time = time.time()
    tr_index = data['trial_%d_tr_indices' % trial_id]
    te_index = data['trial_%d_te_indices' % trial_id]
    auc_matrix = np.zeros(shape=(len(list_c), k_fold))
    for (ind_xi, para_xi) in enumerate(list_c):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = data['x_tr'][tr_index[sub_tr_ind]]
            sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            file_tr = data['data_path'] + '%s_%03d_%d_sub_train.dat' % (data['name'], trial_id, ind)
            file_te = data['data_path'] + '%s_%03d_%d_sub_test.dat' % (data['name'], trial_id, ind)
            generate_data(file_tr, file_te, sub_x_tr, sub_y_tr, sub_x_te, sub_y_te)
            file_model = data['data_path'] + '%s_%03d_%d_%d_sub_model' % (data['name'], trial_id, ind, ind_xi)
            file_pred = data['data_path'] + '%s_%03d_%d_%d_sub_pred' % (data['name'], trial_id, ind, ind_xi)
            start_time = time.time()
            os.system("./svm_perf/svm_perf_learn -v 0 -w 3 -t 0 -c %f -l 10 --b 1 %s %s" %
                      (para_xi, file_tr, file_model))
            print('finished in %f seconds' % (time.time() - start_time))
            xx = os.popen("./svm_perf/svm_perf_classify %s %s %s" % (file_te, file_model, file_pred)).read()
            xx = xx.lstrip().rstrip().split('\n')[8:]
            print([_.split(':')[0].lstrip().rstrip() for _ in xx])
            print([_.split(':')[1].lstrip().rstrip() for _ in xx])
            with open(file_model, 'rb') as f:
                model = f.readlines()[-1]
                model = model.split(b' ')[1:]
                model = model[:-1]
            wt = np.asarray([float(_.split(b':')[1]) for _ in model])
            # in some cases, the zero column missed.
            if len(wt) == data['p'] - 1:
                wt = np.append(wt, 0.0)
            auc = 0.0 if np.isnan(wt).any() else roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            auc_matrix[ind_xi][ind] = auc
    max_auc_ind = np.argmax(np.mean(auc_matrix, axis=1))
    best_c = list_c[int(max_auc_ind)]
    x_tr = data['x_tr'][tr_index]
    y_tr = data['y_tr'][tr_index]
    x_te = data['x_tr'][te_index]
    y_te = data['y_tr'][te_index]
    file_tr = data['data_path'] + '%s_%03d_train.dat' % (data['name'], trial_id)
    file_te = data['data_path'] + '%s_%03d_test.dat' % (data['name'], trial_id)
    generate_data(file_tr, file_te, x_tr, y_tr, x_te, y_te)
    file_model = data['data_path'] + '%s_%03d_model' % (data['name'], trial_id)
    file_pred = data['data_path'] + '%s_%03d_pred' % (data['name'], trial_id)
    start_time = time.time()
    os.system("./svm_perf/svm_perf_learn -v 0 -w 3 -t 0 -c %f -l 10 --b 1 %s %s" %
              (best_c, file_tr, file_model))
    print('finished in %f seconds' % (time.time() - start_time))
    xx = os.popen("./svm_perf/svm_perf_classify %s %s %s" % (file_te, file_model, file_pred)).read()
    xx = xx.lstrip().rstrip().split('\n')[8:]
    print([_.split(':')[0].lstrip().rstrip() for _ in xx])
    print([_.split(':')[1].lstrip().rstrip() for _ in xx])
    with open(file_model, 'rb') as f:
        model = f.readlines()[-1]
        model = model.split(b' ')[1:]
        model = model[:-1]
    wt = np.asarray([float(_.split(b':')[1]) for _ in model])
    # in some cases, the zero column missed.
    if len(wt) == data['p'] - 1:
        wt = np.append(wt, 0.0)
    y_tr_scores, y_te_scores = np.dot(x_tr, wt), np.dot(x_te, wt)
    y_pred_tr = [1 if _ > 0 else -1 for _ in np.dot(x_tr, wt)]
    auc_tr = roc_auc_score(y_true=y_tr, y_score=y_tr_scores)
    f1_tr = f1_score(y_true=y_tr, y_pred=y_pred_tr)
    acc_tr = accuracy_score(y_true=y_tr, y_pred=y_pred_tr)
    pre_tr = precision_score(y_true=y_tr, y_pred=y_pred_tr)
    rec_tr = recall_score(y_true=y_tr, y_pred=y_pred_tr)
    aver_pre_tr = average_precision_score(y_true=y_tr, y_score=y_tr_scores)

    y_pred_te = [1 if _ > 0 else -1 for _ in np.dot(x_te, wt)]
    auc_te = roc_auc_score(y_true=y_te, y_score=y_te_scores)
    f1_te = f1_score(y_true=y_te, y_pred=y_pred_te)
    acc_te = accuracy_score(y_true=y_te, y_pred=y_pred_te)
    pre_te = precision_score(y_true=y_te, y_pred=y_pred_te)
    rec_te = recall_score(y_true=y_te, y_pred=y_pred_te)
    aver_pre_te = average_precision_score(y_true=y_te, y_score=y_te_scores)
    results[trial_id] = {
        'trial_%03d': trial_id,
        'wt': wt,
        'para': [best_c, 'squared_hinge', 'l2', 'max_iter:1000', 'tol:1e-4'],
        'tr': {'y_pred': y_pred_tr, 'auc': auc_tr, 'f1': f1_tr,
               'acc': acc_tr, 'pre': pre_tr, 'rec': rec_tr, 'aver_pre': aver_pre_tr},
        'te': {'y_pred': y_pred_te, 'auc': auc_te, 'f1': f1_te,
               'acc': acc_te, 'pre': pre_te, 'rec': rec_te, 'aver_pre': aver_pre_te}}
    print('trial_id: %03d' % trial_id, time.time() - start_time)
    return results


def test_rank_boost(para):
    start_time = time.time()
    results = dict()
    data, trial_id = para

    os.system(" java -jar RankLib-2.13.jar -train %s/%s_tr_%03d.dat -ranker 2"
              " -round 100 -tc 5 -save %s/model_%03d.txt" %
              (data['data_path'], data['name'], trial_id, data['data_path'], trial_id))

    os.system("java -jar RankLib-2.13.jar -load %s/model_%03d.txt -rank "
              "%s/%s_te_%03d.dat -score %s/scores_te_%03d.txt" %
              (data['data_path'], trial_id, data['data_path'], data['name'], trial_id, data['data_path'], trial_id))

    os.system("java -jar RankLib-2.13.jar -load %s/model_%03d.txt -rank "
              "%s/%s_tr_%03d.dat -score %s/scores_tr_%03d.txt" %
              (data['data_path'], trial_id, data['data_path'], data['name'], trial_id, data['data_path'], trial_id))
    y_tr_scores = []
    with open('%s/scores_tr_%03d.txt' % (data['data_path'], trial_id)) as ff:
        for each_line in ff.readlines():
            y_tr_scores.append(float(each_line.rstrip().split('\t')[-1]))
    y_te_scores = []
    with open('%s/scores_te_%03d.txt' % (data['data_path'], trial_id)) as ff:
        for each_line in ff.readlines():
            y_te_scores.append(float(each_line.rstrip().split('\t')[-1]))
    tr_index = data['trial_%d_tr_indices' % trial_id]
    te_index = data['trial_%d_te_indices' % trial_id]
    wt = np.random.rand(data['p'])
    x_tr = data['x_tr'][tr_index]
    y_tr = data['y_tr'][tr_index]
    x_te = data['x_tr'][te_index]
    y_te = data['y_tr'][te_index]

    y_pred_tr = [1 if _ > 0 else -1 for _ in np.dot(x_tr, wt)]
    auc_tr = roc_auc_score(y_true=y_tr, y_score=y_tr_scores)
    f1_tr = f1_score(y_true=y_tr, y_pred=y_pred_tr)
    acc_tr = accuracy_score(y_true=y_tr, y_pred=y_pred_tr)
    pre_tr = precision_score(y_true=y_tr, y_pred=y_pred_tr)
    rec_tr = recall_score(y_true=y_tr, y_pred=y_pred_tr)
    aver_pre_tr = average_precision_score(y_true=y_tr, y_score=y_tr_scores)

    y_pred_te = [1 if _ > 0 else -1 for _ in np.dot(x_te, wt)]
    auc_te = roc_auc_score(y_true=y_te, y_score=y_te_scores)
    f1_te = f1_score(y_true=y_te, y_pred=y_pred_te)
    acc_te = accuracy_score(y_true=y_te, y_pred=y_pred_te)
    pre_te = precision_score(y_true=y_te, y_pred=y_pred_te)
    rec_te = recall_score(y_true=y_te, y_pred=y_pred_te)
    aver_pre_te = average_precision_score(y_true=y_te, y_score=y_te_scores)
    results[trial_id] = {
        'trial_%03d': trial_id,
        'wt': wt,
        'para': ['l2', 'max_iter:1000', 'tol:1e-4'],
        'tr': {'y_pred': y_pred_tr, 'auc': auc_tr, 'f1': f1_tr,
               'acc': acc_tr, 'pre': pre_tr, 'rec': rec_tr, 'aver_pre': aver_pre_tr},
        'te': {'y_pred': y_pred_te, 'auc': auc_te, 'f1': f1_te,
               'acc': acc_te, 'pre': pre_te, 'rec': rec_te, 'aver_pre': aver_pre_te}}
    print('trial_id: %03d run_time: %.1f tr_auc: %.4f te_auc: %.4f'
          % (trial_id, time.time() - start_time, auc_tr, auc_te))
    return results


def test_linear_svm(para):
    data, trial_id = para
    list_c, k_fold = np.logspace(-4, 4, 10), 5
    results = dict()
    start_time = time.time()
    tr_index = data['trial_%d_tr_indices' % trial_id]
    te_index = data['trial_%d_te_indices' % trial_id]
    auc_matrix = np.zeros(shape=(len(list_c), k_fold))
    acc_matrix = np.zeros(shape=(len(list_c), k_fold))
    for (ind_xi, para_xi) in enumerate(list_c):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = data['x_tr'][tr_index[sub_tr_ind]]
            sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            lr = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=1e-4,
                           C=para_xi, multi_class='ovr', fit_intercept=True,
                           intercept_scaling=1, random_state=None, max_iter=1000)
            lr.fit(X=sub_x_tr, y=sub_y_tr)
            wt = lr.coef_[0]
            auc = 0.0 if np.isnan(wt).any() else roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            acc = accuracy_score(y_true=sub_y_te, y_pred=lr.predict(X=sub_x_te))
            auc_matrix[ind_xi][ind] = auc
            acc_matrix[ind_xi][ind] = acc
    max_auc_ind = np.argmax(np.mean(auc_matrix, axis=1))
    best_c = list_c[int(max_auc_ind)]
    lr = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=1e-4,
                   C=best_c, multi_class='ovr', fit_intercept=True,
                   intercept_scaling=1, random_state=None, max_iter=1000)
    lr.fit(X=data['x_tr'][tr_index], y=data['y_tr'][tr_index])

    wt, x_te, y_te = lr.coef_[0], data['x_tr'][te_index], data['y_tr'][te_index]
    x_tr, y_tr = data['x_tr'][tr_index], data['y_tr'][tr_index]
    y_tr_scores, y_te_scores = np.dot(x_tr, wt), np.dot(x_te, wt)
    y_pred_tr = lr.predict(X=x_tr)
    auc_tr = roc_auc_score(y_true=y_tr, y_score=y_tr_scores)
    f1_tr = f1_score(y_true=y_tr, y_pred=y_pred_tr)
    acc_tr = accuracy_score(y_true=y_tr, y_pred=y_pred_tr)
    pre_tr = precision_score(y_true=y_tr, y_pred=y_pred_tr)
    rec_tr = recall_score(y_true=y_tr, y_pred=y_pred_tr)
    aver_pre_tr = average_precision_score(y_true=y_tr, y_score=y_tr_scores)

    y_pred_te = lr.predict(X=x_te)
    auc_te = roc_auc_score(y_true=y_te, y_score=y_te_scores)
    f1_te = f1_score(y_true=y_te, y_pred=y_pred_te)
    acc_te = accuracy_score(y_true=y_te, y_pred=y_pred_te)
    pre_te = precision_score(y_true=y_te, y_pred=y_pred_te)
    rec_te = recall_score(y_true=y_te, y_pred=y_pred_te)
    aver_pre_te = average_precision_score(y_true=y_te, y_score=y_te_scores)
    results[trial_id] = {
        'trial_%03d': trial_id,
        'wt': wt,
        'para': [best_c, 'squared_hinge', 'l2', 'max_iter:1000', 'tol:1e-4'],
        'tr': {'y_pred': y_pred_tr, 'auc': auc_tr, 'f1': f1_tr,
               'acc': acc_tr, 'pre': pre_tr, 'rec': rec_tr, 'aver_pre': aver_pre_tr},
        'te': {'y_pred': y_pred_te, 'auc': auc_te, 'f1': f1_te,
               'acc': acc_te, 'pre': pre_te, 'rec': rec_te, 'aver_pre': aver_pre_te}}
    print(trial_id, time.time() - start_time)
    return results


def test_spam_l2(para):
    data, trial_id = para
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_c = np.arange(1., 101., 9)
    list_l2 = np.logspace(-4, 4, 10)
    num_passes, step_len, verbose, record_aucs, stop_eps, k_fold = 100, 1e8, 0, 0, 1e-3, 5
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    results = dict()
    start_time = time.time()
    tr_index = data['trial_%d_tr_indices' % trial_id]
    te_index = data['trial_%d_te_indices' % trial_id]
    index, auc_matrix = 0, dict()
    for (ind_xi, para_xi), (ind_l2, para_l2) in product(enumerate(list_c), enumerate(list_l2)):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = data['x_tr'][tr_index[sub_tr_ind]]
            sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_spam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, 0.0, para_l2)
            wt, aucs, rts, epochs = _
            auc = 0.0 if np.isnan(wt).any() else roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            if (ind_xi, ind_l2) not in auc_matrix:
                auc_matrix[(ind_xi, ind_l2)] = []
            auc_matrix[(ind_xi, ind_l2)].append(auc)
            index += 1
        auc_matrix[(ind_xi, ind_l2)] = np.mean(auc_matrix[(ind_xi, ind_l2)])
    best_ind_xi, best_ind_l2 = max(auc_matrix, key=auc_matrix.get)
    best_c, best_l2 = list_c[best_ind_xi], list_l2[best_ind_l2]
    _ = c_algo_spam(data['x_tr'][tr_index], __, __, __, data['y_tr'][tr_index], 0, data['p'],
                    global_paras, best_c, 0.0, best_l2)
    wt, aucs, rts, epochs = _
    x_te, y_te = data['x_tr'][te_index], data['y_tr'][te_index]
    x_tr, y_tr = data['x_tr'][tr_index], data['y_tr'][tr_index]
    y_tr_scores, y_te_scores = np.dot(x_tr, wt), np.dot(x_te, wt)
    y_pred_tr = (y_tr_scores - np.min(y_tr_scores)) / (np.max(y_tr_scores) - np.min(y_tr_scores))
    y_pred_tr = [-1 if _ < 0.5 else 1 for _ in y_pred_tr]
    auc_tr = roc_auc_score(y_true=y_tr, y_score=y_tr_scores)
    f1_tr = f1_score(y_true=y_tr, y_pred=y_pred_tr)
    acc_tr = accuracy_score(y_true=y_tr, y_pred=y_pred_tr)
    pre_tr = precision_score(y_true=y_tr, y_pred=y_pred_tr)
    rec_tr = recall_score(y_true=y_tr, y_pred=y_pred_tr)
    aver_pre_tr = average_precision_score(y_true=y_tr, y_score=y_tr_scores)

    y_pred_te = (y_te_scores - np.min(y_te_scores)) / (np.max(y_te_scores) - np.min(y_te_scores))
    y_pred_te = [-1 if _ < 0.5 else 1 for _ in y_pred_te]
    auc_te = roc_auc_score(y_true=y_te, y_score=y_te_scores)
    f1_te = f1_score(y_true=y_te, y_pred=y_pred_te)
    acc_te = accuracy_score(y_true=y_te, y_pred=y_pred_te)
    pre_te = precision_score(y_true=y_te, y_pred=y_pred_te)
    rec_te = recall_score(y_true=y_te, y_pred=y_pred_te)
    aver_pre_te = average_precision_score(y_true=y_te, y_score=y_te_scores)

    results[trial_id] = {
        'trial_%03d': trial_id,
        'wt': wt,
        'para': [best_c, 'squared_hinge', 'l2', 'max_iter:100', 'tol:1e-4'],
        'tr': {'y_pred': y_pred_tr, 'auc': auc_tr, 'f1': f1_tr,
               'acc': acc_tr, 'pre': pre_tr, 'rec': rec_tr, 'aver_pre': aver_pre_tr},
        'te': {'y_pred': y_pred_te, 'auc': auc_te, 'f1': f1_te,
               'acc': acc_te, 'pre': pre_te, 'rec': rec_te, 'aver_pre': aver_pre_te}}
    print(trial_id, time.time() - start_time)
    return results


def test_spauc_l2(para):
    data, trial_id = para
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_mu = list(10. ** np.asarray([-7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5]))
    list_l2 = np.logspace(-4, 4, 10)
    num_passes, step_len, verbose, record_aucs, stop_eps, k_fold = 100, 1e8, 0, 0, 1e-3, 5
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    results = dict()
    start_time = time.time()
    tr_index = data['trial_%d_tr_indices' % trial_id]
    te_index = data['trial_%d_te_indices' % trial_id]
    index, auc_matrix = 0, dict()
    for (ind_mu, para_mu), (ind_l2, para_l2) in product(enumerate(list_mu), enumerate(list_l2)):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(tr_index), 1)))):
            sub_x_tr = data['x_tr'][tr_index[sub_tr_ind]]
            sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]
            sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
            sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
            _ = c_algo_spauc(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_mu, 0.0, para_l2)
            wt, aucs, rts, epochs = _
            auc = 0.0 if np.isnan(wt).any() else roc_auc_score(y_true=sub_y_te, y_score=np.dot(sub_x_te, wt))
            if (ind_mu, ind_l2) not in auc_matrix:
                auc_matrix[(ind_mu, ind_l2)] = []
            auc_matrix[(ind_mu, ind_l2)].append(auc)
            index += 1
        auc_matrix[(ind_mu, ind_l2)] = np.mean(auc_matrix[(ind_mu, ind_l2)])
    best_ind_mu, best_ind_l2 = max(auc_matrix, key=auc_matrix.get)
    best_mu, best_l2 = list_mu[best_ind_mu], list_l2[best_ind_l2]
    _ = c_algo_spauc(data['x_tr'][tr_index], __, __, __, data['y_tr'][tr_index], 0, data['p'],
                     global_paras, best_mu, 0.0, best_l2)
    wt, aucs, rts, epochs = _
    x_te, y_te = data['x_tr'][te_index], data['y_tr'][te_index]
    x_tr, y_tr = data['x_tr'][tr_index], data['y_tr'][tr_index]
    y_tr_scores, y_te_scores = np.dot(x_tr, wt), np.dot(x_te, wt)
    y_pred_tr = (y_tr_scores - np.min(y_tr_scores)) / (np.max(y_tr_scores) - np.min(y_tr_scores))
    y_pred_tr = [-1 if _ < 0.5 else 1 for _ in y_pred_tr]
    auc_tr = roc_auc_score(y_true=y_tr, y_score=y_tr_scores)
    f1_tr = f1_score(y_true=y_tr, y_pred=y_pred_tr)
    acc_tr = accuracy_score(y_true=y_tr, y_pred=y_pred_tr)
    pre_tr = precision_score(y_true=y_tr, y_pred=y_pred_tr)
    rec_tr = recall_score(y_true=y_tr, y_pred=y_pred_tr)
    aver_pre_tr = average_precision_score(y_true=y_tr, y_score=y_tr_scores)

    y_pred_te = (y_te_scores - np.min(y_te_scores)) / (np.max(y_te_scores) - np.min(y_te_scores))
    y_pred_te = [-1 if _ < 0.5 else 1 for _ in y_pred_te]
    auc_te = roc_auc_score(y_true=y_te, y_score=y_te_scores)
    f1_te = f1_score(y_true=y_te, y_pred=y_pred_te)
    acc_te = accuracy_score(y_true=y_te, y_pred=y_pred_te)
    pre_te = precision_score(y_true=y_te, y_pred=y_pred_te)
    rec_te = recall_score(y_true=y_te, y_pred=y_pred_te)
    aver_pre_te = average_precision_score(y_true=y_te, y_score=y_te_scores)

    results[trial_id] = {
        'trial_%03d': trial_id,
        'wt': wt,
        'para': [best_mu, 'squared_hinge', 'l2', 'max_iter:100', 'tol:1e-4'],
        'tr': {'y_pred': y_pred_tr, 'auc': auc_tr, 'f1': f1_tr,
               'acc': acc_tr, 'pre': pre_tr, 'rec': rec_tr, 'aver_pre': aver_pre_tr},
        'te': {'y_pred': y_pred_te, 'auc': auc_te, 'f1': f1_te,
               'acc': acc_te, 'pre': pre_te, 'rec': rec_te, 'aver_pre': aver_pre_te}}
    print(trial_id, time.time() - start_time)
    return results


def get_data(dataset, num_trials):
    if dataset == 'australian' or dataset == '01':
        return data_preprocess_australian(num_trials=num_trials)
    elif dataset == 'diabetes' or dataset == '02':
        return data_preprocess_diabetes(num_trials=num_trials)
    elif dataset == 'breast_cancer' or dataset == '03':
        return data_preprocess_breast_cancer(num_trials=num_trials)
    elif dataset == 'splice' or dataset == '04':
        return data_preprocess_splice(num_trials=num_trials)
    elif dataset == 'ijcnn1' or dataset == '05':
        return data_preprocess_ijcnn1(num_trials=num_trials)
    elif dataset == 'mushrooms' or dataset == '06':
        return data_preprocess_mushrooms(num_trials=num_trials)
    elif dataset == 'svmguide3' or dataset == '07':
        return data_preprocess_svmguide3(num_trials=num_trials)
    elif dataset == 'german' or dataset == '08':
        return data_preprocess_german(num_trials=num_trials)
    elif dataset == 'fourclass' or dataset == '09':
        return data_preprocess_fourclass(num_trials=num_trials)
    elif dataset == 'a9a' or dataset == '10':
        return data_preprocess_a9a(num_trials=num_trials)
    elif dataset == 'w8a' or dataset == '11':
        return data_preprocess_w8a(num_trials=num_trials)
    elif dataset == 'spambase' or dataset == '12':
        return data_preprocess_spambase(num_trials=num_trials)
    elif dataset == 'yeast' or dataset == '13':
        return data_preprocess_yeast(num_trials=num_trials)


def get_x_y(arr):
    mean_, std_ = np.mean(arr), np.std(arr)
    lower = mean_ - 6. * std_
    upper = mean_ + 6. * std_
    x_axis = []
    bins = []
    range_a, range_b = np.infty, lower
    if std_ == 0.0:
        std_ = 0.0001
    for i in range(int((upper - lower) / std_) + 1):
        x_axis.append(range_b)
        bins.append(len([_ for _ in arr if range_a < _ <= range_b]) * 1.)
        range_a = range_b
        range_b += std_ * 1.
    return x_axis, bins


def draw_figure_aucs(dataset, num_trials=200):
    import matplotlib.pyplot as plt
    from pylab import rcParams
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r"\usepackage{libertine}"
    plt.rcParams["font.size"] = 14
    rcParams['figure.figsize'] = 8, 3.5
    fig, ax = plt.subplots(1, 2)

    method_list = ['linear_svm', 'logistic', 'svm_perf', 'spam_l2', 'spauc_l2', 'rank_boost']
    title_list = ['LinearSVM', 'Logistic', 'SVM-Perf', 'SPAM-$\displaystyle \ell^2$',
                  'SPAUC-$\displaystyle \ell^2$', 'rank-boost']
    marker_list = ['D', 's', 'P', '>', '*', '<']
    results = dict()
    print('-' * 50 + '%s' % dataset + '-' * 50)
    print(' & '.join(title_list))
    for ind, method in enumerate(method_list):
        results['re_%s' % method] = pkl.load(open(root_path + 're_%s_%s.pkl' % (dataset, method), 'rb'))
    for ind, method in enumerate(method_list):
        re = [results['re_%s' % method][_]['tr']['auc'] for _ in range(num_trials)]
        x_axis, bins = get_x_y(re)
        ax[0].plot(x_axis, bins, marker=marker_list[ind], label=title_list[ind])
        print('%.4f$\pm$%.4f & ' % (float(np.mean(re)), float(np.std(re))), end='')
    print('')
    for ind, method in enumerate(method_list):
        re = [results['re_%s' % method][_]['te']['auc'] for _ in range(num_trials)]
        x_axis, bins = get_x_y(re)
        ax[1].plot(x_axis, bins, marker=marker_list[ind], label=title_list[ind])
        print('%.4f$\pm$%.4f & ' % (float(np.mean(re)), float(np.std(re))), end='')
    print()
    print('---- tr ----')
    sort_mean = []
    for i in range(len(method_list)):
        method_i = method_list[i]
        sort_mean.append(np.mean([results['re_%s' % method_i][_]['tr']['auc'] for _ in range(num_trials)]))
    sort_mean = np.asarray(sort_mean)
    indices = np.argsort(sort_mean)[::-1]
    str_ = [title_list[indices[0]]]
    for i in range(len(method_list) - 1):
        method_i = method_list[indices[i]]
        method_j = method_list[indices[i + 1]]
        re_i = [results['re_%s' % method_i][_]['tr']['auc'] for _ in range(num_trials)]
        re_j = [results['re_%s' % method_j][_]['tr']['auc'] for _ in range(num_trials)]
        stat, p_val = stats.ttest_ind(a=re_i, b=re_j)
        if p_val <= 0.05:
            str_.append('>')
        else:
            str_.append('=')
        str_.append(title_list[indices[i + 1]])
    print(' '.join(str_))
    print('---- te ----')
    sort_mean = []
    for i in range(len(method_list)):
        method_i = method_list[i]
        sort_mean.append(np.mean([results['re_%s' % method_i][_]['te']['auc'] for _ in range(num_trials)]))
    sort_mean = np.asarray(sort_mean)
    indices = np.argsort(sort_mean)[::-1]
    str_ = [title_list[indices[0]]]
    for i in range(len(method_list) - 1):
        method_i = method_list[indices[i]]
        method_j = method_list[indices[i + 1]]
        re_i = [results['re_%s' % method_i][_]['te']['auc'] for _ in range(num_trials)]
        re_j = [results['re_%s' % method_j][_]['te']['auc'] for _ in range(num_trials)]
        stat, p_val = stats.ttest_ind(a=re_i, b=re_j)
        if p_val <= 0.05:
            str_.append('>')
        else:
            str_.append('=')
        str_.append(title_list[indices[i + 1]])
    print(' '.join(str_))
    plt.legend()
    plt.subplots_adjust(wspace=0.25, hspace=0.2)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/aucs_%s.pdf' % dataset
    fig.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def draw_figure_accs(num_trials=200):
    method_list = ['linear_svm', 'logistic', 'svm_perf', 'spam_l2', 'spauc_l2']
    title_list = ['LinearSVM', 'Logistic', 'SVM-Perf', 'SPAM-$\displaystyle \ell^2$',
                  'SPAUC-$\displaystyle \ell^2$']
    results = dict()
    data_sets = ['diabetes', 'breast_cancer', 'splice', 'australian', 'spambase',
                 'german', 'svmguide3', 'ijcnn1', 'w8a']
    print(' & '.join(title_list))
    for dataset in data_sets:
        print(dataset + ' & ', end='')
        for ind, method in enumerate(method_list):
            results['re_%s' % method] = pkl.load(open(root_path + 're_%s_%s.pkl' % (dataset, method), 'rb'))
        for ind, method in enumerate(method_list):
            re = [results['re_%s' % method][_]['tr']['acc'] for _ in range(num_trials)]
            print('%.4f$\pm$%.4f & ' % (float(np.mean(re)), float(np.std(re))), end='')
        print('')
    print('-' * 100)
    for dataset in data_sets:
        print(dataset + ' & ', end='')
        for ind, method in enumerate(method_list):
            results['re_%s' % method] = pkl.load(open(root_path + 're_%s_%s.pkl' % (dataset, method), 'rb'))
        for ind, method in enumerate(method_list):
            re = [results['re_%s' % method][_]['te']['acc'] for _ in range(num_trials)]
            print('%.4f$\pm$%.4f & ' % (float(np.mean(re)), float(np.std(re))), end='')
        print('')
    print('---- tr ----')
    for dataset in data_sets:
        print(dataset + ' & ', end='')
        for ind, method in enumerate(method_list):
            results['re_%s' % method] = pkl.load(open(root_path + 're_%s_%s.pkl' % (dataset, method), 'rb'))
        sort_mean = []
        for i in range(len(method_list)):
            method_i = method_list[i]
            sort_mean.append(np.mean([results['re_%s' % method_i][_]['tr']['acc'] for _ in range(num_trials)]))
        sort_mean = np.asarray(sort_mean)
        indices = np.argsort(sort_mean)[::-1]
        str_ = [title_list[indices[0]]]
        for i in range(len(method_list) - 1):
            method_i = method_list[indices[i]]
            method_j = method_list[indices[i + 1]]
            re_i = [results['re_%s' % method_i][_]['tr']['acc'] for _ in range(num_trials)]
            re_j = [results['re_%s' % method_j][_]['tr']['acc'] for _ in range(num_trials)]
            stat, p_val = stats.ttest_ind(a=re_i, b=re_j)
            if p_val <= 0.05:
                str_.append('>')
            else:
                str_.append('=')
            str_.append(title_list[indices[i + 1]])
        print(' '.join(str_))
    print('---- te ----')
    for dataset in data_sets:
        print(dataset + ' & ', end='')
        for ind, method in enumerate(method_list):
            results['re_%s' % method] = pkl.load(open(root_path + 're_%s_%s.pkl' % (dataset, method), 'rb'))
        sort_mean = []
        for i in range(len(method_list)):
            method_i = method_list[i]
            sort_mean.append(np.mean([results['re_%s' % method_i][_]['te']['acc'] for _ in range(num_trials)]))
        sort_mean = np.asarray(sort_mean)
        indices = np.argsort(sort_mean)[::-1]
        str_ = [title_list[indices[0]]]
        for i in range(len(method_list) - 1):
            method_i = method_list[indices[i]]
            method_j = method_list[indices[i + 1]]
            re_i = [results['re_%s' % method_i][_]['te']['acc'] for _ in range(num_trials)]
            re_j = [results['re_%s' % method_j][_]['te']['acc'] for _ in range(num_trials)]
            stat, p_val = stats.ttest_ind(a=re_i, b=re_j)
            if p_val <= 0.05:
                str_.append('>')
            else:
                str_.append('=')
            str_.append(title_list[indices[i + 1]])
        print(' '.join(str_))


def draw_t_sne():
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = r"\usepackage{libertine}"
    plt.rcParams["font.size"] = 14
    plt.rcParams['figure.figsize'] = 8, 8
    dataset = 'a9a'
    x_tr, y_tr = [], []
    method_list = ['linear_svm', 'logistic', 'svm_perf', 'spam_l2', 'spauc_l2']
    title_list = ['linear-svm', 'logistic', 'svm-perf', 'spam-l2', 'spauc-l2']
    num_trials = 200
    norms = []
    for ind, method in enumerate(method_list):
        results = pkl.load(open(root_path + 're_%s_%s.pkl' % (dataset, method), 'rb'))
        wt = np.asarray([results[item]['wt'] for item in range(num_trials)])
        print(np.mean(wt, axis=0), np.std(wt, axis=0))
        print(np.linalg.norm(np.mean(wt, axis=0)))
        norms.append(np.linalg.norm(np.mean(wt, axis=0)))
        for item in range(num_trials):
            x_tr.append(results[item]['wt'])
            y_tr.append(ind)
    t_sne = TSNE(random_state=123).fit_transform(x_tr)
    fig, ax = plt.subplots(1, 1)
    c = ['r', 'g', 'b', 'y', 'm']
    for xx in range(len(title_list)):
        ax.scatter([t_sne[ind, 0]
                    for ind, y in enumerate(y_tr) if y == xx],
                   [t_sne[ind, 1]
                    for ind, y in enumerate(y_tr) if y == xx],
                   color=c[xx], label='%s : %.4f' % (title_list[xx], norms[xx]))
    ax.set_xlabel("$\displaystyle x_1$")
    ax.set_ylabel("$\displaystyle x_2$")
    ax.legend(fancybox=True, loc='center', framealpha=1.0, frameon=True, borderpad=0.1,
              labelspacing=0.2, handletextpad=0.1, markerfirst=True)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/t_sne-%s.pdf' % dataset
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def main():
    command = sys.argv[1]
    num_trials = 200
    if command == 'run':
        method = sys.argv[2]
        dataset = sys.argv[3]
        num_cpus = int(sys.argv[4])
        data = get_data(dataset=dataset, num_trials=num_trials)
        pool = multiprocessing.Pool(processes=num_cpus)
        para_space = [(data, trial_i) for trial_i in range(num_trials)]
        if method == 'linear_svm':
            results = pool.map(test_linear_svm, para_space)
        elif method == 'spam_l2':
            results = pool.map(test_spam_l2, para_space)
        elif method == 'spauc_l2':
            results = pool.map(test_spauc_l2, para_space)
        elif method == 'logistic':
            results = pool.map(test_logistic, para_space)
        elif method == 'svm_perf':
            results = pool.map(test_svm_perf, para_space)
        elif method == 'rank_boost':
            results = pool.map(test_rank_boost, para_space)
        else:
            results = None
        pool.close()
        pool.join()
        merge_results = dict()
        for re in results:
            for trial_id in re:
                merge_results[trial_id] = re[trial_id]
        pkl.dump(merge_results, open(root_path + 're_%s_%s.pkl' % (dataset, method), 'wb'))
    elif command == 'show_t_sne':
        draw_t_sne()
    elif command == 'show_aucs':
        data_sets = ['diabetes', 'breast_cancer', 'splice', 'australian', 'spambase',
                     'german', 'svmguide3', 'ijcnn1', 'w8a']
        data_sets = ['yeast']
        for dataset in data_sets:
            draw_figure_aucs(dataset=dataset, num_trials=num_trials)
    elif command == 'show_accs':
        draw_figure_accs(num_trials=num_trials)


if __name__ == '__main__':
    main()
