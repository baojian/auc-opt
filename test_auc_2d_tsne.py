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
             results['opt-auc']['te']['f1'], results['opt-auc']['te']['loss'], time.time() - test_time), flush=True)
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
             results[name]['tr']['f1'], results[name]['tr']['loss'], time.time() - run_time), flush=True)
    run_time = time.time()
    results[name]['te']['auc'] = roc_auc_score(y_true=y_te, y_score=lr.decision_function(X=x_te))
    results[name]['te']['b_acc'] = balanced_accuracy_score(y_true=y_te, y_pred=lr.predict(X=x_te))
    results[name]['te']['f1'] = f1_score(y_true=y_te, y_pred=lr.predict(X=x_te))
    results[name]['te']['loss'] = log_loss(y_true=y_te, y_pred=lr.predict_proba(X=x_te))

    print('LR on test auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f '
          % (results[name]['te']['auc'], results[name]['te']['b_acc'],
             results[name]['te']['f1'], results[name]['te']['loss'], time.time() - run_time), flush=True)
    return results


def run_algo_svc(x_tr, y_tr, x_te, y_te, rand_state, class_weight, results):
    np.random.seed(rand_state)
    run_time = time.time()
    lin_svm = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=1e-5,
                        C=1. / len(x_tr), multi_class='ovr', fit_intercept=True,
                        intercept_scaling=1, class_weight=class_weight, verbose=0,
                        random_state=rand_state, max_iter=5000)
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
             results[name]['tr']['f1'], results[name]['tr']['loss'], time.time() - run_time), flush=True)
    run_time = time.time()
    decision = lin_svm.decision_function(X=x_te)
    results[name]['te']['auc'] = roc_auc_score(y_true=y_te, y_score=decision)
    results[name]['te']['b_acc'] = balanced_accuracy_score(y_true=y_te, y_pred=lin_svm.predict(X=x_te))
    results[name]['te']['f1'] = f1_score(y_true=y_te, y_pred=lin_svm.predict(X=x_te))
    results[name]['te']['loss'] = log_loss(y_true=y_te, y_pred=softmax(np.c_[-decision, decision]))
    print('Linear-svm on train auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f'
          % (results[name]['te']['auc'], results[name]['te']['b_acc'],
             results[name]['te']['f1'], results[name]['te']['loss'], time.time() - run_time), flush=True)
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
             results[name]['tr']['f1'], results[name]['tr']['loss'], time.time() - run_time), flush=True)
    decision = ridge.decision_function(X=x_te)
    results[name]['te']['auc'] = roc_auc_score(y_true=y_te, y_score=ridge.decision_function(X=x_te))
    results[name]['te']['b_acc'] = balanced_accuracy_score(y_true=y_te, y_pred=ridge.predict(X=x_te))
    results[name]['te']['f1'] = f1_score(y_true=y_te, y_pred=ridge.predict(X=x_te))
    results[name]['te']['loss'] = log_loss(y_true=y_te, y_pred=softmax(np.c_[-decision, decision]))
    print('Ridge on test auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f'
          % (results[name]['te']['auc'], results[name]['te']['b_acc'],
             results[name]['te']['f1'], results[name]['te']['loss'], time.time() - run_time), flush=True)
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
             results[name]['tr']['f1'], results[name]['tr']['loss'], time.time() - run_time), flush=True)
    decision = percep.decision_function(X=x_te)
    results[name]['te']['auc'] = roc_auc_score(y_true=y_te, y_score=percep.decision_function(X=x_te))
    results[name]['te']['b_acc'] = balanced_accuracy_score(y_true=y_te, y_pred=percep.predict(X=x_te))
    results[name]['te']['f1'] = f1_score(y_true=y_te, y_pred=percep.predict(X=x_te))
    results[name]['te']['loss'] = log_loss(y_true=y_te, y_pred=softmax(np.c_[-decision, decision]))
    print('Perceptron on test auc: %.6f acc: %.6f f1: %.6f loss: %.6f run_time: %.2f'
          % (results[name]['te']['auc'], results[name]['te']['b_acc'],
             results[name]['te']['f1'], results[name]['te']['loss'], time.time() - run_time), flush=True)
    return results


def run_single_compare(para):
    rand_state, data_name, em_perplexity, data_ind, list_methods = para
    np.random.seed(rand_state)
    results = {_: {__: dict() for __ in ['tr', 'te']} for _ in list_methods}
    data = pkl.load(open(root_path + '%02d_%s/t_sne_2d_%s.pkl'
                         % (data_ind[data_name], data_name, data_name), 'rb'))
    n, data_y = len(data['x_tr']), data['y_tr']
    embeddings = data['embeddings'][em_perplexity]
    rand_perm = np.random.permutation(n)
    x_tr, y_tr = embeddings[rand_perm[:n // 2]], data_y[rand_perm[:n // 2]]
    x_te, y_te = embeddings[rand_perm[n // 2:]], data_y[rand_perm[n // 2:]]

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
    return rand_state, results


def run_parallel(dataset, num_cpus):
    num_trials, em_perplexity = 100, 50.0
    data_ind = {'splice': 1, 'mushrooms': 2, 'australian': 3, 'spambase': 4,
                'ionosphere': 5, 'fourclass': 6, 'breast_cancer': 7, 'pima': 8,
                'yeast_cyt': 9, 'german': 10, 'svmguide3': 11, 'a9a': 12,
                'spectf': 13, 'pen_digits': 14, 'page_blocks': 15, 'opt_digits': 16,
                'ijcnn1': 17, 'letter_a': 18, 'yeast_me1': 19, 'w8a': 20}
    list_methods = ['opt-auc', 'lr', 'wei-lr', 'svm', 'wei-svm', 'ridge', 'wei-ridge', 'percep', 'wei-percep']
    para_space = [(trial_i, dataset, em_perplexity, data_ind, list_methods)
                  for trial_i in range(num_trials)]
    pool = multiprocessing.Pool(processes=num_cpus)
    batch_results = pool.map(run_single_compare, para_space)
    pool.close()
    pool.join()
    results_mat = {trial_i: results for trial_i, results in batch_results}
    pkl.dump(results_mat, open('results_2d_%02d_%s.pkl' % (data_ind[dataset], dataset), 'wb'))


def main():
    run_parallel(dataset=sys.argv[1], num_cpus=int(sys.argv[2]))


if __name__ == '__main__':
    main()
