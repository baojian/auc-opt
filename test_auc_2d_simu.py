# -*- coding: utf-8 -*-
import os
import sys
import time
import operator
import warnings
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils.extmath import softmax
from sklearn.preprocessing import StandardScaler
import multiprocessing
import pickle as pkl

from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons

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


def cal_best_threshold(y_tr, scores):
    """
    Chosen the threshold for prediction function by
    using balanced accuracy score.
    :param y_tr:
    :param scores:
    :return:
    """
    fpr, tpr, thresholds = roc_curve(y_true=y_tr, y_score=scores)
    y_pred = np.zeros_like(y_tr)
    best_b_acc, best_f1, best_threshold = -1., -1., -1.0
    for fpr_, tpr_, threshold in zip(fpr, tpr, thresholds):
        y_pred[np.argwhere(scores < threshold)] = -1
        y_pred[np.argwhere(scores >= threshold)] = 1
        b_acc = balanced_accuracy_score(y_true=y_tr, y_pred=y_pred)
        if best_b_acc < b_acc:
            best_b_acc = b_acc
            best_f1 = f1_score(y_true=y_tr, y_pred=y_pred)
            best_threshold = threshold
    return best_b_acc, best_f1, best_threshold


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
             results['opt-auc']['te']['f1'], results['opt-auc']['te']['loss']
             , time.time() - test_time), flush=True)
    return results


def run_algo_lr(x_tr, y_tr, x_te, y_te, rand_state, class_weight, results):
    np.random.seed(rand_state)
    run_time = time.time()
    lr = LogisticRegression(  # without ell_2 regularization.
        penalty='none', dual=False, tol=1e-5, C=1.0, fit_intercept=True,
        intercept_scaling=1, class_weight=class_weight, random_state=rand_state,
        solver='lbfgs', max_iter=10000, multi_class='ovr', verbose=0,
        warm_start=False, n_jobs=1, l1_ratio=None)
    lr.fit(X=x_tr, y=y_tr)
    train_time = time.time() - run_time
    name = 'lr' if class_weight is None else 'wei-lr'
    results[name]['w'] = lr.coef_.flatten()
    results[name]['intercept'] = lr.intercept_
    results[name]['tr']['train_time'] = train_time
    results[name]['tr']['auc'] = roc_auc_score(y_true=y_tr, y_score=lr.decision_function(X=x_tr))
    results[name]['tr']['b_acc'] = balanced_accuracy_score(y_true=y_tr, y_pred=lr.predict(X=x_tr))
    results[name]['tr']['f1'] = f1_score(y_true=y_tr, y_pred=lr.predict(X=x_tr))
    results[name]['tr']['loss'] = log_loss(y_true=y_tr, y_pred=lr.predict_proba(X=x_tr))

    print('LR on train auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f '
          % (results[name]['tr']['auc'], results[name]['tr']['b_acc'],
             results[name]['tr']['f1'], results[name]['tr']['loss'],
             time.time() - run_time), flush=True)
    run_time = time.time()
    results[name]['te']['auc'] = roc_auc_score(y_true=y_te, y_score=lr.decision_function(X=x_te))
    results[name]['te']['b_acc'] = balanced_accuracy_score(y_true=y_te, y_pred=lr.predict(X=x_te))
    results[name]['te']['f1'] = f1_score(y_true=y_te, y_pred=lr.predict(X=x_te))
    results[name]['te']['loss'] = log_loss(y_true=y_te, y_pred=lr.predict_proba(X=x_te))

    print('LR on test auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f '
          % (results[name]['te']['auc'], results[name]['te']['b_acc'],
             results[name]['te']['f1'], results[name]['te']['loss'],
             time.time() - run_time), flush=True)
    return results


def run_algo_svc(x_tr, y_tr, x_te, y_te, rand_state, class_weight, results):
    np.random.seed(rand_state)
    run_time = time.time()
    lin_svm = SVC(
        C=1.0, kernel='linear', degree=3, gamma='scale',
        coef0=0.0, shrinking=True, probability=False,
        tol=1e-5, cache_size=2000, class_weight=class_weight,
        verbose=False, max_iter=-1, decision_function_shape='ovr',
        break_ties=False, random_state=rand_state)
    lin_svm.fit(X=x_tr, y=y_tr)
    name = 'svm' if class_weight is None else 'wei-svm'
    results[name]['w'] = lin_svm.coef_.flatten()
    results[name]['intercept'] = lin_svm.intercept_
    decision = lin_svm.decision_function(X=x_tr)
    results[name]['tr']['train_time'] = time.time() - run_time
    results[name]['tr']['auc'] = roc_auc_score(y_true=y_tr, y_score=decision)
    results[name]['tr']['b_acc'] = balanced_accuracy_score(y_true=y_tr, y_pred=lin_svm.predict(X=x_tr))
    results[name]['tr']['f1'] = f1_score(y_true=y_tr, y_pred=lin_svm.predict(X=x_tr))
    results[name]['tr']['loss'] = log_loss(y_true=y_tr, y_pred=softmax(np.c_[-decision, decision]))
    print('Linear-svm on train auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f'
          % (results[name]['tr']['auc'], results[name]['tr']['b_acc'],
             results[name]['tr']['f1'], results[name]['tr']['loss'],
             time.time() - run_time), flush=True)
    run_time = time.time()
    decision = lin_svm.decision_function(X=x_te)
    results[name]['te']['auc'] = roc_auc_score(y_true=y_te, y_score=decision)
    results[name]['te']['b_acc'] = balanced_accuracy_score(y_true=y_te, y_pred=lin_svm.predict(X=x_te))
    results[name]['te']['f1'] = f1_score(y_true=y_te, y_pred=lin_svm.predict(X=x_te))
    results[name]['te']['loss'] = log_loss(y_true=y_te, y_pred=softmax(np.c_[-decision, decision]))
    print('Linear-svm on train auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f'
          % (results[name]['te']['auc'], results[name]['te']['b_acc'],
             results[name]['te']['f1'], results[name]['te']['loss'],
             time.time() - run_time), flush=True)
    return results


def run_algo_ridge(x_tr, y_tr, x_te, y_te, rand_state, class_weight, results):
    np.random.seed(rand_state)
    run_time = time.time()
    alphas = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    ridge = RidgeClassifierCV(alphas=alphas, fit_intercept=True, normalize=False,
                              scoring=None, cv=None, class_weight=class_weight,
                              store_cv_values=False)
    ridge.fit(X=x_tr, y=y_tr)
    name = 'ridge' if class_weight is None else 'wei-ridge'
    results[name]['w'] = ridge.coef_.flatten()
    results[name]['intercept'] = ridge.intercept_
    decision = ridge.decision_function(X=x_tr)
    results[name]['tr']['train_time'] = time.time() - run_time
    results[name]['tr']['auc'] = roc_auc_score(y_true=y_tr, y_score=ridge.decision_function(X=x_tr))
    results[name]['tr']['b_acc'] = balanced_accuracy_score(y_true=y_tr, y_pred=ridge.predict(X=x_tr))
    results[name]['tr']['f1'] = f1_score(y_true=y_tr, y_pred=ridge.predict(X=x_tr))
    results[name]['tr']['loss'] = log_loss(y_true=y_tr, y_pred=softmax(np.c_[-decision, decision]))

    print('Ridge on train auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f '
          % (results[name]['tr']['auc'], results[name]['tr']['b_acc'],
             results[name]['tr']['f1'], results[name]['tr']['loss'],
             time.time() - run_time), flush=True)
    decision = ridge.decision_function(X=x_te)
    results[name]['te']['auc'] = roc_auc_score(y_true=y_te, y_score=ridge.decision_function(X=x_te))
    results[name]['te']['b_acc'] = balanced_accuracy_score(y_true=y_te, y_pred=ridge.predict(X=x_te))
    results[name]['te']['f1'] = f1_score(y_true=y_te, y_pred=ridge.predict(X=x_te))
    results[name]['te']['loss'] = log_loss(y_true=y_te, y_pred=softmax(np.c_[-decision, decision]))
    print('Ridge on test auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f'
          % (results[name]['te']['auc'], results[name]['te']['b_acc'],
             results[name]['te']['f1'], results[name]['te']['loss'],
             time.time() - run_time), flush=True)
    return results


def run_algo_perceptron(x_tr, y_tr, x_te, y_te, rand_state, class_weight, results):
    np.random.seed(rand_state)
    run_time = time.time()
    percep = Perceptron(
        penalty=None, alpha=0.0, fit_intercept=True,
        max_iter=5000, tol=1e-5, shuffle=True, verbose=0, eta0=1.0,
        n_jobs=1, random_state=rand_state, early_stopping=False,
        n_iter_no_change=5, class_weight=class_weight, warm_start=False)
    percep.fit(X=x_tr, y=y_tr)
    name = 'percep' if class_weight is None else 'wei-percep'
    results[name]['w'] = percep.coef_.flatten()
    results[name]['intercept'] = percep.intercept_
    decision = percep.decision_function(X=x_tr)
    results[name]['tr']['train_time'] = time.time() - run_time
    results[name]['tr']['auc'] = roc_auc_score(y_true=y_tr, y_score=percep.decision_function(X=x_tr))
    results[name]['tr']['b_acc'] = balanced_accuracy_score(y_true=y_tr, y_pred=percep.predict(X=x_tr))
    results[name]['tr']['f1'] = f1_score(y_true=y_tr, y_pred=percep.predict(X=x_tr))
    results[name]['tr']['loss'] = log_loss(y_true=y_tr, y_pred=softmax(np.c_[-decision, decision]))

    print('Perceptron on train auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f '
          % (results[name]['tr']['auc'], results[name]['tr']['b_acc'],
             results[name]['tr']['f1'], results[name]['tr']['loss'],
             time.time() - run_time), flush=True)
    decision = percep.decision_function(X=x_te)
    results[name]['te']['auc'] = roc_auc_score(y_true=y_te, y_score=percep.decision_function(X=x_te))
    results[name]['te']['b_acc'] = balanced_accuracy_score(y_true=y_te, y_pred=percep.predict(X=x_te))
    results[name]['te']['f1'] = f1_score(y_true=y_te, y_pred=percep.predict(X=x_te))
    results[name]['te']['loss'] = log_loss(y_true=y_te, y_pred=softmax(np.c_[-decision, decision]))
    print('Perceptron on test auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f'
          % (results[name]['te']['auc'], results[name]['te']['b_acc'],
             results[name]['te']['f1'], results[name]['te']['loss'],
             time.time() - run_time), flush=True)
    return results


def make_imbalance(x_data, y_data, imbalance_ratio, rand_state):
    np.random.seed(rand_state)
    total_samples = 5000
    posi_indices = np.argwhere(y_data > 0).flatten()
    nega_indices = np.argwhere(y_data < 0).flatten()
    num_posi = int(total_samples * imbalance_ratio)
    num_nega = total_samples - num_posi
    posi_indices = posi_indices[:num_posi]
    nega_indices = nega_indices[:num_nega]
    p, n = len(posi_indices), len(nega_indices)
    x_p_tr, y_p_tr = x_data[posi_indices[:p // 2]], y_data[posi_indices[:p // 2]]
    x_p_te, y_p_te = x_data[posi_indices[p // 2:]], y_data[posi_indices[p // 2:]]
    x_n_tr, y_n_tr = x_data[nega_indices[:n // 2]], y_data[nega_indices[:n // 2]]
    x_n_te, y_n_te = x_data[nega_indices[n // 2:]], y_data[nega_indices[n // 2:]]
    x_tr, y_tr = np.concatenate((x_p_tr, x_n_tr)), np.concatenate((y_p_tr, y_n_tr))
    x_te, y_te = np.concatenate((x_p_te, x_n_te)), np.concatenate((y_p_te, y_n_te))
    pe = np.random.permutation(len(x_tr))
    x_tr, y_tr = x_tr[pe], y_tr[pe]
    pe = np.random.permutation(len(x_te))
    x_te, y_te = x_te[pe], y_te[pe]
    return x_tr, y_tr, x_te, y_te


def simulation_dataset(data_name, noise, rand_state, imbalance_ratio):
    draw_fig = False
    n = 10000
    if data_name == 'blobs':
        x_data, y_data = make_blobs(n_samples=n, n_features=2, centers=2, cluster_std=noise,
                                    center_box=(-1.0, 1.0), shuffle=True, random_state=rand_state)

        y_data[y_data == 0] = -1
        if draw_fig:
            plt.scatter(x_data[np.argwhere(y_data > 0), 0], x_data[np.argwhere(y_data > 0), 1], c='r')
            plt.scatter(x_data[np.argwhere(y_data < 0), 0], x_data[np.argwhere(y_data < 0), 1], c='b')
            plt.show()

        x_data = StandardScaler().fit_transform(x_data)
    elif data_name == 'circles':
        x_data, y_data = make_circles(n_samples=n, shuffle=True, noise=noise, random_state=rand_state, factor=.1)
        y_data[y_data == 0] = -1
        if draw_fig:
            plt.scatter(x_data[np.argwhere(y_data > 0), 0], x_data[np.argwhere(y_data > 0), 1], c='r')
            plt.scatter(x_data[np.argwhere(y_data < 0), 0], x_data[np.argwhere(y_data < 0), 1], c='b')
            plt.show()
        x_data = StandardScaler().fit_transform(x_data)
    elif data_name == 'moons':
        x_data, y_data = make_moons(n_samples=n, shuffle=True, noise=.5, random_state=rand_state)
        y_data[y_data == 0] = -1
        x_data = StandardScaler().fit_transform(x_data)
    else:
        x_data = np.random.normal(loc=0.0, scale=1., size=5000)
        y_data = np.ones(5000)
    x_tr, y_tr, x_te, y_te = make_imbalance(x_data=x_data, y_data=y_data,
                                            imbalance_ratio=imbalance_ratio, rand_state=rand_state)

    return np.asarray(x_tr, dtype=np.float64), np.asarray(y_tr, dtype=np.float64), \
           np.asarray(x_te, dtype=np.float64), np.asarray(y_te, dtype=np.float64)


def run_single_compare(para):
    rand_state, noise, data_name, list_methods = para
    np.random.seed(rand_state)
    all_results = dict()
    for imbalance_ratio in np.arange(0.005, 0.501, 0.005):
        results = {_: {__: dict() for __ in ['tr', 'te']} for _ in list_methods}
        x_tr, y_tr, x_te, y_te = simulation_dataset(data_name, noise, rand_state, imbalance_ratio)
        results = run_algo_opt_auc(x_tr, y_tr, x_te, y_te, rand_state, results)
        results = run_algo_lr(x_tr, y_tr, x_te, y_te, rand_state, None, results)
        results = run_algo_lr(x_tr, y_tr, x_te, y_te, rand_state, 'balanced', results)
        results = run_algo_svc(x_tr, y_tr, x_te, y_te, rand_state, None, results)
        results = run_algo_svc(x_tr, y_tr, x_te, y_te, rand_state, 'balanced', results)
        results = run_algo_ridge(x_tr, y_tr, x_te, y_te, rand_state, None, results)
        results = run_algo_ridge(x_tr, y_tr, x_te, y_te, rand_state, 'balanced', results)
        results = run_algo_perceptron(x_tr, y_tr, x_te, y_te, rand_state, None, results)
        results = run_algo_perceptron(x_tr, y_tr, x_te, y_te, rand_state, 'balanced', results)
        results = {**results, **{'x_tr': x_tr, 'y_tr': y_tr, 'x_te': x_te, 'y_te': y_te,
                                 'data_name': data_name, 'rand_state': rand_state, 'trial_i': rand_state}}
        all_results[imbalance_ratio] = results
    return rand_state, all_results


def test_single():
    run_single_compare((1, .1, 'blobs', ['opt-auc', 'lr', 'wei-lr', 'svm', 'wei-svm',
                                         'ridge', 'wei-ridge', 'percep', 'wei-percep']))


def run_simulation(dataset, noise, num_cpus):
    num_trials = 100
    list_methods = ['opt-auc', 'lr', 'wei-lr', 'svm', 'wei-svm',
                    'ridge', 'wei-ridge', 'percep', 'wei-percep']
    para_space = [(trial_i, noise, dataset, list_methods) for trial_i in range(num_trials)]
    pool = multiprocessing.Pool(processes=num_cpus)
    batch_results = pool.map(run_single_compare, para_space)
    pool.close()
    pool.join()

    for trial_i, results in batch_results:
        pkl.dump(results, open(
            root_path + 'results/simu/results_simu_%s_%.2f_%02d.pkl' % (dataset, noise, trial_i), 'wb'))


def main():
    run_simulation(dataset=sys.argv[1], noise=float(sys.argv[2]), num_cpus=int(sys.argv[3]))


if __name__ == '__main__':
    main()
