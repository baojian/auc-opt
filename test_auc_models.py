# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle as pkl
import multiprocessing
import tempfile
from itertools import product
from functools import reduce
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from data_preprocess import get_data
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC as SVM
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from auc_opt_3d import opt_auc_3d_algo

root_path = "/home/baojian/data/aistats22-auc-opt/"
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

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


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


def cal_best_threshold_te(y_te, scores, threshold):
    auc = roc_auc_score(y_true=y_te, y_score=scores)
    y_pred = [1 if _ > threshold else -1 for _ in scores]
    b_acc = balanced_accuracy_score(y_true=y_te, y_pred=y_pred)
    f1 = f1_score(y_true=y_te, y_pred=y_pred)
    acc = accuracy_score(y_true=y_te, y_pred=y_pred)
    confusion_mat = confusion_matrix(y_true=y_te, y_pred=y_pred)
    n_posi = len([_ for _ in y_te if _ > 0])
    n_nega = len([_ for _ in y_te if _ < 0])
    am = (confusion_mat[0][0] / n_nega + confusion_mat[1][1] / n_posi) / 2.
    loss = log_loss(y_true=y_te, y_pred=y_pred)
    return auc, b_acc, f1, acc, am, loss


def decision_func_rank_boost(x_tr, alpha, threshold, rankfeat):
    alpha = np.asarray(alpha)
    threshold = np.asarray(threshold)
    rankfeat = np.asarray(rankfeat, dtype=int)
    y_pred = np.zeros(len(x_tr))
    for i in range(len(alpha)):
        y_pred += alpha[i] * (x_tr[:, rankfeat[i]] > threshold[i])
    return y_pred


def decision_func_svm_perf(x_tr, y_tr, model):
    f_tr = tempfile.NamedTemporaryFile()
    for index, item in enumerate(x_tr):
        str_ = [b'%d' % int(y_tr[index])]
        for index_2, val in enumerate(item):
            str_.append(b'%d:%.15f' % (index_2 + 1, val))
        str_ = b' '.join(str_) + b'\n'
        f_tr.write(str_)
    f_tr.seek(0)
    file_model = tempfile.NamedTemporaryFile()
    file_pred = tempfile.NamedTemporaryFile()
    file_model.writelines(model)
    file_model.seek(0)
    pred_cmd = "./svm_perf/svm_perf_classify -v 0 -y 0  %s %s %s"
    os.system(pred_cmd % (f_tr.name, file_model.name, file_pred.name))
    file_pred.seek(0)
    tr_scores = np.asarray([float(_) for _ in file_pred.readlines()])
    f_tr.close()
    file_model.close()
    file_pred.close()
    return tr_scores


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


def pred_tr_te_auc(method, trial_id, y_tr, y_te1, y_te2, y_te3,
                   tr_scores, te1_scores, te2_scores, te3_scores, start_time, start_tr_time):
    tr_auc, tr_b_acc, tr_f1, tr_acc, tr_am, tr_loss, threshold = cal_best_threshold_tr(y_tr=y_tr, scores=tr_scores)
    te1_auc, te1_b_acc, te1_f1, te1_acc, te1_am, te1_loss = \
        cal_best_threshold_te(y_te=y_te1, scores=te1_scores, threshold=threshold)
    te2_auc, te2_b_acc, te2_f1, te2_acc, te2_am, te2_loss = \
        cal_best_threshold_te(y_te=y_te2, scores=te2_scores, threshold=threshold)
    te3_auc, te3_b_acc, te3_f1, te3_acc, te3_am, te3_loss = \
        cal_best_threshold_te(y_te=y_te3, scores=te3_scores, threshold=threshold)
    re = {'trial_id': trial_id, 'rand_state': trial_id,
          'tr': {'auc': tr_auc, 'f1': tr_f1, 'b_acc': tr_b_acc, 'acc': tr_acc, 'am': tr_am, 'loss': tr_loss,
                 'train_time': time.time() - start_tr_time},
          'te1': {'auc': te1_auc, 'f1': te1_f1, 'acc': te1_acc, 'b_acc': te1_b_acc, 'am': te1_am, 'loss': te1_loss},
          'te2': {'auc': te2_auc, 'f1': te2_f1, 'acc': te2_acc, 'b_acc': te2_b_acc, 'am': te2_am, 'loss': te2_loss},
          'te3': {'auc': te3_auc, 'f1': te3_f1, 'acc': te3_acc, 'b_acc': te3_b_acc, 'am': te3_am, 'loss': te3_loss}}
    print('%15s-%03d tr_auc: %.6f test1_auc: %.6f test2_auc: %.6f test3_auc: %.6f run_time: %.6f'
          % (method, trial_id, re['tr']['auc'], re['te1']['auc'], re['te2']['auc'],
             re['te3']['auc'], time.time() - start_time), flush=True)
    results = dict()
    results[method] = re
    return results


def pred_tr_te_std(method, trial_id, y_tr, y_te1, y_te2, y_te3, tr_scores, te1_scores, te2_scores, te3_scores,
                   tr_pred, te1_pred, te2_pred, te3_pred, start_time, start_tr_time):
    tr_auc = roc_auc_score(y_true=y_tr, y_score=tr_scores)
    tr_b_acc = balanced_accuracy_score(y_true=y_tr, y_pred=tr_pred)
    tr_f1 = f1_score(y_true=y_tr, y_pred=tr_pred)
    tr_acc = accuracy_score(y_true=y_tr, y_pred=tr_pred)
    n_posi = len([_ for _ in y_tr if _ > 0])
    n_nega = len([_ for _ in y_tr if _ < 0])
    tr_confusion_mat = confusion_matrix(y_true=y_tr, y_pred=tr_pred)
    tr_am = (tr_confusion_mat[0][0] / n_nega + tr_confusion_mat[1][1] / n_posi) / 2.
    tr_loss = log_loss(y_true=y_tr, y_pred=tr_pred)

    n_posi1 = len([_ for _ in y_te1 if _ > 0])
    n_nega1 = len([_ for _ in y_te1 if _ < 0])
    te1_auc = roc_auc_score(y_true=y_te1, y_score=te1_scores)
    te1_b_acc = balanced_accuracy_score(y_true=y_te1, y_pred=te1_pred)
    te1_f1 = f1_score(y_true=y_te1, y_pred=te1_pred)
    te1_acc = accuracy_score(y_true=y_te1, y_pred=te1_pred)
    te1_confusion_mat = confusion_matrix(y_true=y_te1, y_pred=te1_pred)
    te1_am = (te1_confusion_mat[0][0] / n_nega1 + te1_confusion_mat[1][1] / n_posi1) / 2.
    te1_loss = log_loss(y_true=y_te1, y_pred=te1_pred)

    n_posi2 = len([_ for _ in y_te2 if _ > 0])
    n_nega2 = len([_ for _ in y_te2 if _ < 0])
    te2_auc = roc_auc_score(y_true=y_te2, y_score=te2_scores)
    te2_b_acc = balanced_accuracy_score(y_true=y_te2, y_pred=te2_pred)
    te2_f1 = f1_score(y_true=y_te2, y_pred=te2_pred)
    te2_acc = accuracy_score(y_true=y_te2, y_pred=te2_pred)
    te2_confusion_mat = confusion_matrix(y_true=y_te2, y_pred=te2_pred)
    te2_am = (te2_confusion_mat[0][0] / n_nega2 + te2_confusion_mat[1][1] / n_posi2) / 2.
    te2_loss = log_loss(y_true=y_te2, y_pred=te2_pred)

    n_posi3 = len([_ for _ in y_te3 if _ > 0])
    n_nega3 = len([_ for _ in y_te3 if _ < 0])
    te3_auc = roc_auc_score(y_true=y_te3, y_score=te3_scores)
    te3_b_acc = balanced_accuracy_score(y_true=y_te3, y_pred=te3_pred)
    te3_f1 = f1_score(y_true=y_te3, y_pred=te3_pred)
    te3_acc = accuracy_score(y_true=y_te3, y_pred=te3_pred)
    te3_confusion_mat = confusion_matrix(y_true=y_te3, y_pred=te3_pred)
    te3_am = (te3_confusion_mat[0][0] / n_nega3 + te3_confusion_mat[1][1] / n_posi3) / 2.
    te3_loss = log_loss(y_true=y_te3, y_pred=te3_pred)

    re = {'trial_id': trial_id, 'rand_state': trial_id,
          'tr': {'auc': tr_auc, 'f1': tr_f1, 'b_acc': tr_b_acc, 'acc': tr_acc, 'am': tr_am, 'loss': tr_loss,
                 'train_time': time.time() - start_tr_time},
          'te1': {'auc': te1_auc, 'f1': te1_f1, 'acc': te1_acc, 'b_acc': te1_b_acc, 'am': te1_am, 'loss': te1_loss},
          'te2': {'auc': te2_auc, 'f1': te2_f1, 'acc': te2_acc, 'b_acc': te2_b_acc, 'am': te2_am, 'loss': te2_loss},
          'te3': {'auc': te3_auc, 'f1': te3_f1, 'acc': te3_acc, 'b_acc': te3_b_acc, 'am': te3_am, 'loss': te3_loss}}
    print('%15s-%03d tr_auc: %.6f test1_auc: %.6f test2_auc: %.6f test3_auc: %.6f run_time: %.6f'
          % (method, trial_id, re['tr']['auc'], re['te1']['auc'], re['te2']['auc'],
             re['te3']['auc'], time.time() - start_time), flush=True)
    results = dict()
    results[method] = re
    return results


def get_standard_data(data, trial_id, std_type='StandardScaler'):
    tr_index = data['trial_%d_tr_indices' % trial_id]
    te_index = data['trial_%d_te_indices' % trial_id]
    x_tr, y_tr = data['x_tr'][tr_index], data['y_tr'][tr_index]
    # follow the same distribution as training
    x_te1, y_te1 = data['x_tr'][te_index], data['y_tr'][te_index]
    # the number of positive samples is equal to the number of negative
    num_posi = min(len([_ for _ in y_te1 if _ > 0]), len([_ for _ in y_te1 if _ < 0]))
    x_te2, y_te2 = [], []
    n_p, n_n = 0, 0
    for ind, _ in enumerate(y_te1):
        if _ == 1 and n_p < num_posi:
            n_p += 1
            x_te2.append(x_te1[ind])
            y_te2.append(y_te1[ind])
        elif _ == 1 and n_p >= num_posi:
            continue
        elif _ == -1 and n_n < num_posi:
            n_n += 1
            x_te2.append(x_te1[ind])
            y_te2.append(y_te1[ind])
        else:
            continue
    x_te2, y_te2 = np.asarray(x_te2), np.asarray(y_te2)
    # the number of positive / the number of negative =
    # the number of negative / the number of positive
    ratio = len([_ for _ in y_te1 if _ < 0]) / len([_ for _ in y_te1 if _ < 0])
    num_nega = int(num_posi / ratio) if int(num_posi / ratio) > 0 else 1
    x_te3, y_te3 = [], []
    n_p, n_n = 0, 0
    for ind, _ in enumerate(y_te1):
        if _ == 1 and n_p < num_posi:
            n_p += 1
            x_te3.append(x_te1[ind])
            y_te3.append(y_te1[ind])
        elif _ == 1 and n_p >= num_posi:
            continue
        elif _ == -1 and n_n < num_nega:
            n_n += 1
            x_te3.append(x_te1[ind])
            y_te3.append(y_te1[ind])
        else:
            continue
    x_te3, y_te3 = np.asarray(x_te3), np.asarray(y_te3)
    if std_type == 'StandardScaler':
        std_normalize = StandardScaler().fit(x_tr)
    elif std_type == 'MinMaxScaler':
        std_normalize = MinMaxScaler(feature_range=(-1, 1)).fit(x_tr)
    elif std_type == 'Normalizer':
        std_normalize = Normalizer(norm='l2').fit(x_tr)
    else:
        std_normalize = StandardScaler().fit(x_tr)

    trans_x_tr = std_normalize.transform(np.array(x_tr))
    trans_x_te1 = std_normalize.transform(np.array(x_te1))
    trans_x_te2 = std_normalize.transform(np.array(x_te2))
    trans_x_te3 = std_normalize.transform(np.array(x_te3))
    num_posi = 0
    num_nega = 0
    index = len(y_tr)
    for i in range(len(y_tr)):
        if num_posi * num_nega > 2000:
            index = i
            break
        num_posi += (1 if y_tr[i] == 1. else 0)
        num_nega += (1 if y_tr[i] == -1. else 0)
    return trans_x_tr[:index], y_tr[:index], trans_x_te1, y_te1, trans_x_te2, y_te2, trans_x_te3, y_te3


def get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind, std_type='StandardScaler'):
    tr_index = data['trial_%d_tr_indices' % trial_id]
    sub_x_tr = data['x_tr'][tr_index[sub_tr_ind]]
    sub_y_tr = data['y_tr'][tr_index[sub_tr_ind]]
    sub_x_te = data['x_tr'][tr_index[sub_te_ind]]
    sub_y_te = data['y_tr'][tr_index[sub_te_ind]]
    if std_type == 'StandardScaler':
        std_normalize = StandardScaler().fit(sub_x_tr)
    elif std_type == 'MinMaxScaler':
        std_normalize = MinMaxScaler(feature_range=(-1, 1)).fit(sub_x_tr)
    elif std_type == 'Normalizer':
        std_normalize = Normalizer(norm='l2').fit(sub_x_tr)
    else:
        std_normalize = StandardScaler().fit(sub_x_tr)
    trans_sub_x_tr = std_normalize.transform(np.array(sub_x_tr))
    trans_sub_x_te = std_normalize.transform(np.array(sub_x_te))
    return trans_sub_x_tr, sub_y_tr, trans_sub_x_te, sub_y_te


def run_algo_opt_auc(para):
    start_time = time.time()
    data, trial_id = para
    method = 'opt_auc'
    start_tr_time = time.time()
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    w, auc, train_time = c_opt_auc(np.asarray(x_tr, dtype=np.float64), np.asarray(y_tr, dtype=np.float64), 2e-16)
    tr_scores = np.dot(x_tr, w)
    te1_scores = np.dot(x_te1, w)
    te2_scores = np.dot(x_te2, w)
    te3_scores = np.dot(x_te3, w)
    re = pred_tr_te_auc(method, trial_id, y_tr, y_te1, y_te2, y_te3,
                        tr_scores, te1_scores, te2_scores, te3_scores, start_time, start_tr_time)
    re[method]['w'] = w
    return {trial_id: re}


def run_algo_opt_auc_3d(para):
    start_time = time.time()
    data, trial_id = para
    method = 'opt_auc_3d'
    start_tr_time = time.time()
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    w, auc, train_time = opt_auc_3d_algo(
        np.asarray(x_tr, dtype=np.float64), np.asarray(y_tr, dtype=np.float64))
    tr_scores = np.dot(x_tr, w)
    te1_scores = np.dot(x_te1, w)
    te2_scores = np.dot(x_te2, w)
    te3_scores = np.dot(x_te3, w)
    re = pred_tr_te_auc(method, trial_id, y_tr, y_te1, y_te2, y_te3,
                        tr_scores, te1_scores, te2_scores, te3_scores, start_time, start_tr_time)
    re[method]['w'] = w
    return {trial_id: re}


def run_algo_rank_boost(para):
    start_time = time.time()
    data, trial_id, method = para
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    list_t_iter, k_fold = [10, 50, 100, 200, 300, 500, 800, 1000], 5
    auc_matrix = np.zeros(shape=(len(list_t_iter), k_fold))
    for (ind_xi, t) in enumerate(list_t_iter):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(x_tr), 1)))):
            sub_x_tr, sub_y_tr, sub_x_te, sub_y_te = get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind)
            alpha, threshold, rankfeat, _, run_time = c_rank_boost(
                np.asarray(sub_x_tr, dtype=np.float64), np.asarray(sub_y_tr, dtype=np.float64), int(t))
            try:
                auc_matrix[ind_xi][ind] = roc_auc_score(
                    y_true=sub_y_te, y_score=decision_func_rank_boost(sub_x_te, alpha, threshold, rankfeat))
            except ValueError:
                pass
    start_tr_time = time.time()
    best_t = list_t_iter[int(np.argmax(np.mean(auc_matrix, axis=1)))]
    alpha, threshold, rankfeat, _, run_time = c_rank_boost(
        np.asarray(x_tr, dtype=np.float64), np.asarray(y_tr, dtype=np.float64), int(best_t))
    tr_scores = decision_func_rank_boost(x_tr, alpha, threshold, rankfeat)
    te1_scores = decision_func_rank_boost(x_te1, alpha, threshold, rankfeat)
    te2_scores = decision_func_rank_boost(x_te2, alpha, threshold, rankfeat)
    te3_scores = decision_func_rank_boost(x_te3, alpha, threshold, rankfeat)
    re = pred_tr_te_auc(method, trial_id, y_tr, y_te1, y_te2, y_te3,
                        tr_scores, te1_scores, te2_scores, te3_scores, start_time, start_tr_time)
    re[method]['w'] = {'alpha_threshold_rankfeat': [alpha, threshold, rankfeat]}
    re[method]['best_t'] = best_t
    return {trial_id: re}


def run_algo_spam_l2(para):
    data, trial_id = para
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_c = np.arange(1., 101., 9)
    num_passes = 100
    if data['name'] == 'spectrometer':
        list_c = np.logspace(-5, 3, 10)
        num_passes = 500
    if data['name'] == 'w7a':
        list_c = np.logspace(-5, 3, 10)
        num_passes = 200
    list_l2 = np.logspace(-5, 5, 10)
    method = 'spam'
    step_len, verbose, record_aucs, stop_eps, k_fold = 1e8, 0, 0, 1e-6, 5
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    start_time = time.time()
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    index, auc_matrix = 0, dict()
    for (ind_xi, para_xi), (ind_l2, para_l2) in product(enumerate(list_c), enumerate(list_l2)):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(x_tr), 1)))):
            sub_x_tr, sub_y_tr, sub_x_te, sub_y_te = get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind)
            _ = c_algo_spam(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_xi, 0.0, para_l2)
            wt, aucs, rts, epochs = _
            if np.isfinite(np.asarray(wt)).all():
                scores = np.dot(sub_x_te, wt)
                if np.isfinite(np.asarray(scores)).all():
                    try:
                        auc = roc_auc_score(y_true=sub_y_te, y_score=scores)
                    except ValueError:
                        auc = 0.0
                        pass
                else:
                    auc = 0.0
            else:
                auc = 0.0
            if (ind_xi, ind_l2) not in auc_matrix:
                auc_matrix[(ind_xi, ind_l2)] = []
            auc_matrix[(ind_xi, ind_l2)].append(auc)
            index += 1
        auc_matrix[(ind_xi, ind_l2)] = np.mean(auc_matrix[(ind_xi, ind_l2)])
    start_tr_time = time.time()
    best_ind_xi, best_ind_l2 = max(auc_matrix, key=auc_matrix.get)
    best_c, best_l2 = list_c[best_ind_xi], list_l2[best_ind_l2]
    _ = c_algo_spam(np.asarray(x_tr, dtype=np.float64), __, __, __, np.asarray(y_tr, dtype=np.float64),
                    0, data['p'], global_paras, best_c, 0.0, best_l2)
    wt, aucs, rts, epochs = _
    wt = wt if np.isfinite(np.asarray(wt)).all() else np.zeros(data['p'])
    tr_scores, te1_scores = np.dot(x_tr, wt), np.dot(x_te1, wt)
    te2_scores, te3_scores = np.dot(x_te2, wt), np.dot(x_te3, wt)
    re = pred_tr_te_auc(method, trial_id, y_tr, y_te1, y_te2, y_te3,
                        tr_scores, te1_scores, te2_scores, te3_scores, start_time, start_tr_time)
    re[method]['w'] = wt
    re[method]['best_c'] = best_c
    re[method]['best_l2'] = best_l2
    return {trial_id: re}


def run_algo_spauc_l2(para):
    data, trial_id = para
    __ = np.empty(shape=(1,), dtype=float)
    # candidate parameters
    list_mu = list(10. ** np.asarray([-7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5]))
    list_l2 = np.logspace(-5, 5, 10)
    num_passes = 100
    if data['name'] == 'spectrometer':
        list_mu = list(10. ** np.asarray([-2., -1.5, -1., -0.5, 0.0, 0.5, 1., 1.5, 2.0, 2.5]))
        list_l2 = np.logspace(-6, 5, 10)
        num_passes = 500
    method = 'spauc'
    step_len, verbose, record_aucs, stop_eps, k_fold = 1e8, 0, 0, 1e-6, 5
    global_paras = np.asarray([num_passes, step_len, verbose, record_aucs, stop_eps], dtype=float)
    start_time = time.time()
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    index, auc_matrix = 0, dict()
    for (ind_mu, para_mu), (ind_l2, para_l2) in product(enumerate(list_mu), enumerate(list_l2)):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(x_tr), 1)))):
            sub_x_tr, sub_y_tr, sub_x_te, sub_y_te = get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind)
            _ = c_algo_spauc(sub_x_tr, __, __, __, sub_y_tr, 0, data['p'], global_paras, para_mu, 0.0, para_l2)
            wt, aucs, rts, epochs = _
            if np.isfinite(np.asarray(wt)).all():
                scores = np.dot(sub_x_te, wt)
                if np.isfinite(np.asarray(scores)).all():
                    try:
                        auc = roc_auc_score(y_true=sub_y_te, y_score=scores)
                    except ValueError:
                        auc = 0.0
                        pass
                else:
                    auc = 0.0
            else:
                auc = 0.0
            if (ind_mu, ind_l2) not in auc_matrix:
                auc_matrix[(ind_mu, ind_l2)] = []
            auc_matrix[(ind_mu, ind_l2)].append(auc)
            index += 1
        auc_matrix[(ind_mu, ind_l2)] = np.mean(auc_matrix[(ind_mu, ind_l2)])
        # print(para_mu, para_l2, auc_matrix[(ind_mu, ind_l2)])
    start_tr_time = time.time()
    best_ind_mu, best_ind_l2 = max(auc_matrix, key=auc_matrix.get)
    best_mu, best_l2 = list_mu[best_ind_mu], list_l2[best_ind_l2]
    # print(best_mu, best_l2, auc_matrix[(best_ind_mu, best_ind_l2)])
    _ = c_algo_spauc(np.asarray(x_tr, dtype=np.float64), __, __, __, np.asarray(y_tr, dtype=np.float64),
                     0, data['p'], global_paras, best_mu, 0.0, best_l2)
    wt, aucs, rts, epochs = _
    # for some bad case, it has nan or inf
    wt = wt if np.isfinite(np.asarray(wt)).all() else np.zeros(data['p'])
    # print(wt[:5])
    tr_scores, te1_scores = np.dot(x_tr, wt), np.dot(x_te1, wt)
    te2_scores, te3_scores = np.dot(x_te2, wt), np.dot(x_te3, wt)
    re = pred_tr_te_auc(method, trial_id, y_tr, y_te1, y_te2, y_te3,
                        tr_scores, te1_scores, te2_scores, te3_scores, start_time, start_tr_time)
    re[method]['w'] = wt
    re[method]['best_mu'] = best_mu
    re[method]['best_l2'] = best_l2
    return {trial_id: re}


def run_algo_svm_perf(para):
    data, trial_id, kernel = para
    list_c, k_fold = np.asarray([2. ** _ for _ in np.arange(-20, 11, 2, dtype=float)]), 5
    method = 'svm_perf_lin' if kernel == 'linear' else 'svm_perf_rbf'
    if kernel == 'linear':
        list_c = np.asarray([2. ** _ for _ in np.arange(-20, 11, 2, dtype=float)])
        if data['name'] == 'ecoli_imu':
            list_c = list_c[:11]  # too large C, the algorithm cannot stop.
        if data['name'] == 'spectrometer':
            list_c = list_c[:12]  # too large C, the algorithm cannot stop.
        if data['name'] == 'pen_digits_5':
            list_c = list_c[:12]  # too large C, the algorithm cannot stop.
    else:
        list_c = [1.]
        if data['name'] == 'ecoli_imu':
            list_c = [0.001, 0.01, 0.1, 1.0, 10., 100.]
        if data['name'] == 'australian':
            list_c = [2. ** _ for _ in range(8)]
        if data['name'] == 'fourclass':
            list_c = np.asarray([2. ** _ for _ in np.arange(-6, 7, dtype=float)])
    if len(list_c) == 1:
        start_time = time.time()
        best_c = 20.0
        x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
        start_tr_time = time.time()
        tr_scores, te1_scores, model = cmd_svm_perf(x_tr, y_tr, x_te1, y_te1, best_c, kernel)
        tr_scores, te2_scores, model = cmd_svm_perf(x_tr, y_tr, x_te2, y_te2, best_c, kernel)
        tr_scores, te3_scores, model = cmd_svm_perf(x_tr, y_tr, x_te3, y_te3, best_c, kernel)
        re = pred_tr_te_auc(method, trial_id, y_tr, y_te1, y_te2, y_te3,
                            tr_scores, te1_scores, te2_scores, te3_scores, start_time, start_tr_time)
        re[method]['w'] = model
        re[method]['best_c'] = best_c
        return {trial_id: re}
    start_time = time.time()
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    auc_matrix = np.zeros(shape=(len(list_c), k_fold))
    for (ind_xi, para_xi) in enumerate(list_c):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(x_tr), 1)))):
            sub_x_tr, sub_y_tr, sub_x_te, sub_y_te = get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind)
            tr_scores, te_scores, _ = cmd_svm_perf(sub_x_tr, sub_y_tr, sub_x_te, sub_y_te, para_xi, kernel)
            try:
                auc_matrix[ind_xi][ind] = roc_auc_score(y_true=sub_y_te, y_score=te_scores)
            except ValueError:
                pass

    best_c = list_c[int(np.argmax(np.mean(auc_matrix, axis=1)))]
    start_tr_time = time.time()
    tr_scores, te1_scores, model = cmd_svm_perf(x_tr, y_tr, x_te1, y_te1, best_c, kernel)
    tr_scores, te2_scores, model = cmd_svm_perf(x_tr, y_tr, x_te2, y_te2, best_c, kernel)
    tr_scores, te3_scores, model = cmd_svm_perf(x_tr, y_tr, x_te3, y_te3, best_c, kernel)
    re = pred_tr_te_auc(method, trial_id, y_tr, y_te1, y_te2, y_te3,
                        tr_scores, te1_scores, te2_scores, te3_scores, start_time, start_tr_time)
    re[method]['w'] = model
    re[method]['best_c'] = best_c
    return {trial_id: re}


def run_algo_adaboost(para):
    start_time = time.time()
    data, trial_id = para
    method = 'adaboost'
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    list_n_est, k_fold = [2, 3, 4, 5, 6, 10, 50, 100, 150, 200], 5
    auc_matrix = np.zeros(shape=(len(list_n_est), k_fold))
    for (ind_xi, n_est) in enumerate(list_n_est):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(x_tr), 1)))):
            adaboost = AdaBoost(base_estimator=None, n_estimators=n_est,
                                learning_rate=0.1, algorithm='SAMME.R', random_state=trial_id)
            sub_x_tr, sub_y_tr, sub_x_te, sub_y_te = get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind)
            adaboost.fit(X=sub_x_tr, y=sub_y_tr)
            try:
                auc_matrix[ind_xi][ind] = roc_auc_score(
                    y_true=sub_y_te, y_score=adaboost.decision_function(X=sub_x_te))
            except ValueError:
                pass
    start_tr_time = time.time()
    best_n_est = list_n_est[int(np.argmax(np.mean(auc_matrix, axis=1)))]
    adaboost = AdaBoost(base_estimator=None, n_estimators=best_n_est,
                        learning_rate=0.1, algorithm='SAMME.R', random_state=trial_id)
    adaboost.fit(X=x_tr, y=y_tr)
    tr_pred, te1_pred = adaboost.predict(X=x_tr), adaboost.predict(X=x_te1)
    te2_pred, te3_pred = adaboost.predict(X=x_te2), adaboost.predict(X=x_te3)
    tr_scores, te1_scores = adaboost.decision_function(X=x_tr), adaboost.decision_function(X=x_te1)
    te2_scores, te3_scores = adaboost.decision_function(X=x_te2), adaboost.decision_function(X=x_te3)
    re = pred_tr_te_std(method, trial_id, y_tr, y_te1, y_te2, y_te3, tr_scores, te1_scores, te2_scores, te3_scores,
                        tr_pred, te1_pred, te2_pred, te3_pred, start_time, start_tr_time)
    re[method]['w'] = adaboost
    re[method]['n_est'] = best_n_est
    return {trial_id: re}


def run_algo_c_svm(para):
    data, trial_id, class_weight = para
    list_c, k_fold = np.logspace(-6, 6, 20), 5
    method = 'c_svm' if class_weight is None else 'b_c_svm'
    start_time = time.time()
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    auc_matrix = np.zeros(shape=(len(list_c), k_fold))
    for (ind_xi, para_xi) in enumerate(list_c):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(x_tr), 1)))):
            lin_svm = SVM(penalty='l2', loss='hinge', dual=True, tol=1e-5,
                          C=para_xi, multi_class='ovr', fit_intercept=True,
                          intercept_scaling=1, class_weight=class_weight, verbose=0,
                          random_state=trial_id, max_iter=5000)
            sub_x_tr, sub_y_tr, sub_x_te, sub_y_te = get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind)
            lin_svm.fit(X=sub_x_tr, y=sub_y_tr)
            try:
                auc_matrix[ind_xi][ind] = roc_auc_score(
                    y_true=sub_y_te, y_score=lin_svm.decision_function(X=sub_x_te))
            except ValueError:
                pass

    start_tr_time = time.time()
    best_c = list_c[int(np.argmax(np.mean(auc_matrix, axis=1)))]
    lin_svm = SVM(penalty='l2', loss='hinge', dual=True, tol=1e-5,
                  C=best_c, multi_class='ovr', fit_intercept=True,
                  intercept_scaling=1, class_weight=class_weight, verbose=0,
                  random_state=trial_id, max_iter=5000)
    lin_svm.fit(X=x_tr, y=y_tr)

    tr_pred, te1_pred = lin_svm.predict(X=x_tr), lin_svm.predict(X=x_te1)
    te2_pred, te3_pred = lin_svm.predict(X=x_te2), lin_svm.predict(X=x_te3)
    tr_scores, te1_scores = lin_svm.decision_function(X=x_tr), lin_svm.decision_function(X=x_te1)
    te2_scores, te3_scores = lin_svm.decision_function(X=x_te2), lin_svm.decision_function(X=x_te3)

    re = pred_tr_te_std(method, trial_id, y_tr, y_te1, y_te2, y_te3, tr_scores, te1_scores, te2_scores, te3_scores,
                        tr_pred, te1_pred, te2_pred, te3_pred, start_time, start_tr_time)
    re[method]['w'] = lin_svm
    re[method]['best_c'] = best_c
    return {trial_id: re}


def run_algo_rbf_svm(para):
    data, trial_id, class_weight = para
    list_c, k_fold, start_time = np.logspace(-6, 6, 20), 5, time.time()
    if data['name'] == 'australian':
        list_c = np.logspace(-5, 5, 12)
    method = 'rbf_svm' if class_weight is None else 'b_rbf_svm'
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    auc_matrix = np.zeros(shape=(len(list_c), k_fold))
    for (ind_xi, para_xi) in enumerate(list_c):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(x_tr), 1)))):
            rbf_svm = SVC(C=para_xi, kernel='rbf', degree=3, gamma='scale',
                          coef0=0.0, shrinking=True, probability=False,
                          tol=1e-5, cache_size=1000, class_weight=class_weight,
                          verbose=False, max_iter=-1, decision_function_shape='ovr',
                          break_ties=False, random_state=trial_id)
            sub_x_tr, sub_y_tr, sub_x_te, sub_y_te = get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind)
            rbf_svm.fit(X=sub_x_tr, y=sub_y_tr)
            scores = rbf_svm.decision_function(X=sub_x_te)
            try:
                auc_matrix[ind_xi][ind] = roc_auc_score(y_true=sub_y_te, y_score=scores)
            except ValueError:
                pass
    start_tr_time = time.time()
    best_c = list_c[int(np.argmax(np.mean(auc_matrix, axis=1)))]
    rbf_svm = SVC(C=best_c, kernel='rbf', degree=3, gamma='scale',
                  coef0=0.0, shrinking=True, probability=False,
                  tol=1e-5, cache_size=1000, class_weight=class_weight,
                  verbose=False, max_iter=-1, decision_function_shape='ovr',
                  break_ties=False, random_state=trial_id)
    rbf_svm.fit(X=x_tr, y=y_tr)
    tr_pred, te1_pred = rbf_svm.predict(X=x_tr), rbf_svm.predict(X=x_te1)
    te2_pred, te3_pred = rbf_svm.predict(X=x_te2), rbf_svm.predict(X=x_te3)
    tr_scores, te1_scores = rbf_svm.decision_function(X=x_tr), rbf_svm.decision_function(X=x_te1)
    te2_scores, te3_scores = rbf_svm.decision_function(X=x_te2), rbf_svm.decision_function(X=x_te3)
    re = pred_tr_te_std(method, trial_id, y_tr, y_te1, y_te2, y_te3, tr_scores, te1_scores, te2_scores, te3_scores,
                        tr_pred, te1_pred, te2_pred, te3_pred, start_time, start_tr_time)
    re[method]['w'] = rbf_svm
    re[method]['best_c'] = best_c
    return {trial_id: re}


def run_algo_lr(para):
    data, trial_id, class_weight = para
    list_c, k_fold = np.logspace(-6, 6, 20), 5
    method = 'lr' if class_weight is None else 'b_lr'
    start_time = time.time()
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    auc_matrix = np.zeros(shape=(len(list_c), k_fold))
    for (ind_xi, para_xi) in enumerate(list_c):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(x_tr), 1)))):
            lr = LR(penalty='l2', dual=False, tol=1e-5, C=para_xi, fit_intercept=True,
                    intercept_scaling=1, class_weight=class_weight, random_state=trial_id,
                    solver='lbfgs', max_iter=5000, multi_class='auto', verbose=0,
                    warm_start=False, n_jobs=None, l1_ratio=None)
            sub_x_tr, sub_y_tr, sub_x_te, sub_y_te = get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind)
            lr.fit(X=sub_x_tr, y=sub_y_tr)
            try:
                auc = roc_auc_score(y_true=sub_y_te, y_score=lr.decision_function(X=sub_x_te))
                auc_matrix[ind_xi][ind] = auc
            except ValueError:
                pass

    best_c = list_c[int(np.argmax(np.mean(auc_matrix, axis=1)))]
    start_tr_time = time.time()
    lr = LR(penalty='l2', dual=False, tol=1e-5, C=best_c, fit_intercept=True,
            intercept_scaling=1, class_weight=class_weight, random_state=trial_id,
            solver='lbfgs', max_iter=5000, multi_class='auto', verbose=0,
            warm_start=False, n_jobs=None, l1_ratio=None)
    lr.fit(X=x_tr, y=y_tr)
    tr_pred, te1_pred = lr.predict(X=x_tr), lr.predict(X=x_te1)
    te2_pred, te3_pred = lr.predict(X=x_te2), lr.predict(X=x_te3)
    tr_scores, te1_scores = lr.decision_function(X=x_tr), lr.decision_function(X=x_te1)
    te2_scores, te3_scores = lr.decision_function(X=x_te2), lr.decision_function(X=x_te3)
    re = pred_tr_te_std(method, trial_id, y_tr, y_te1, y_te2, y_te3, tr_scores, te1_scores, te2_scores, te3_scores,
                        tr_pred, te1_pred, te2_pred, te3_pred, start_time, start_tr_time)
    re[method]['w'] = lr
    re[method]['best_c'] = best_c
    return {trial_id: re}


def run_algo_rf(para):
    data, trial_id, class_weight = para
    list_n_est, k_fold = [10, 50, 100, 150, 200, 300, 400, 500], 5
    min_leaf = 1
    if data['name'] == 'ecoli_imu' or data['name'] == 'australian':
        list_n_est = [5, 10, 20, 30, 40, 50, 100]
        min_leaf = 10
    if data['name'] == 'australian':
        list_n_est = [5, 10, 20, 30, 40, 50, 100]
        min_leaf = 1

    start_time = time.time()
    method = 'rf' if class_weight is None else 'b_rf'
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    auc_matrix = np.zeros(shape=(len(list_n_est), k_fold))
    for (ind_xi, n_est) in enumerate(list_n_est):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(x_tr), 1)))):
            rand_forest = RF(
                n_estimators=n_est, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=min_leaf,
                min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=trial_id,
                verbose=0, warm_start=False, class_weight=class_weight, ccp_alpha=0.0, max_samples=None)
            sub_x_tr, sub_y_tr, sub_x_te, sub_y_te = get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind)
            rand_forest.fit(X=sub_x_tr, y=sub_y_tr)
            try:
                auc_matrix[ind_xi][ind] = roc_auc_score(
                    y_true=sub_y_te, y_score=rand_forest.predict_proba(X=sub_x_te)[:, 1])
            except ValueError:
                pass
    start_tr_time = time.time()
    best_n_est = list_n_est[int(np.argmax(np.mean(auc_matrix, axis=1)))]
    rand_forest = RF(
        n_estimators=best_n_est, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=min_leaf,
        min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
        min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=trial_id, verbose=0,
        warm_start=False, class_weight=class_weight, ccp_alpha=0.0, max_samples=None)
    rand_forest.fit(X=x_tr, y=y_tr)
    tr_pred, te1_pred = rand_forest.predict(X=x_tr), rand_forest.predict(X=x_te1)
    te2_pred, te3_pred = rand_forest.predict(X=x_te2), rand_forest.predict(X=x_te3)
    tr_scores, te1_scores = rand_forest.predict_proba(X=x_tr)[:, 1], rand_forest.predict_proba(X=x_te1)[:, 1]
    te2_scores, te3_scores = rand_forest.predict_proba(X=x_te2)[:, 1], rand_forest.predict_proba(X=x_te3)[:, 1]
    re = pred_tr_te_std(method, trial_id, y_tr, y_te1, y_te2, y_te3, tr_scores, te1_scores, te2_scores, te3_scores,
                        tr_pred, te1_pred, te2_pred, te3_pred, start_time, start_tr_time)
    re[method]['w'] = rand_forest
    re[method]['best_n_est'] = best_n_est
    return {trial_id: re}


def run_algo_gb(para):
    data, trial_id = para
    list_n_est, k_fold = [10, 50, 100, 200, 300, 500, 800, 1000], 5
    if data['name'] == 'ecoli_imu':
        list_n_est = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    if data['name'] == 'ecoli_imu':
        list_n_est = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    start_time = time.time()
    method = 'gb'
    x_tr, y_tr, x_te1, y_te1, x_te2, y_te2, x_te3, y_te3 = get_standard_data(data, trial_id)
    auc_matrix = np.zeros(shape=(len(list_n_est), k_fold))
    for (ind_xi, n_est) in enumerate(list_n_est):
        kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
        for ind, (sub_tr_ind, sub_te_ind) in enumerate(kf.split(np.zeros(shape=(len(x_tr), 1)))):
            grad_boost = GB(
                loss='deviance', learning_rate=0.1, n_estimators=n_est, subsample=1.0, criterion='friedman_mse',
                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_depth=3,
                min_impurity_decrease=0., min_impurity_split=None, init=None, random_state=trial_id,
                max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated',
                validation_fraction=0.0, n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0)
            sub_x_tr, sub_y_tr, sub_x_te, sub_y_te = get_standard_data_sub(data, trial_id, sub_tr_ind, sub_te_ind)
            grad_boost.fit(X=sub_x_tr, y=sub_y_tr)
            try:
                auc_matrix[ind_xi][ind] = roc_auc_score(
                    y_true=sub_y_te, y_score=grad_boost.predict_proba(X=sub_x_te)[:, 1])
            except ValueError:
                pass
    start_tr_time = time.time()
    best_n_est = list_n_est[int(np.argmax(np.mean(auc_matrix, axis=1)))]
    grad_boost = GB(
        loss='deviance', learning_rate=0.1, n_estimators=best_n_est, subsample=1.0, criterion='friedman_mse',
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_depth=3, min_impurity_decrease=0.,
        min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None,
        warm_start=False, presort='deprecated', validation_fraction=0.0, n_iter_no_change=None, tol=1e-4,
        ccp_alpha=0.0)
    grad_boost.fit(X=x_tr, y=y_tr)
    tr_pred, te1_pred = grad_boost.predict(X=x_tr), grad_boost.predict(X=x_te1)
    te2_pred, te3_pred = grad_boost.predict(X=x_te2), grad_boost.predict(X=x_te3)
    tr_scores, te1_scores = grad_boost.predict_proba(X=x_tr)[:, 1], grad_boost.predict_proba(X=x_te1)[:, 1]
    te2_scores, te3_scores = grad_boost.predict_proba(X=x_te2)[:, 1], grad_boost.predict_proba(X=x_te3)[:, 1]
    re = pred_tr_te_std(method, trial_id, y_tr, y_te1, y_te2, y_te3, tr_scores, te1_scores, te2_scores, te3_scores,
                        tr_pred, te1_pred, te2_pred, te3_pred, start_time, start_tr_time)
    re[method]['w'] = grad_boost
    re[method]['best_n_est'] = best_n_est
    return {trial_id: re}


def parallel_by_method_dataset(dtype, dataset, method, num_cpus):
    num_trials, split_ratio = 50, 0.5
    data = get_data(dtype=dtype, dataset=dataset, num_trials=num_trials, split_ratio=split_ratio)
    pool = multiprocessing.Pool(processes=num_cpus)
    if method == 'rank_boost':
        para_space = [(data, trial_i, method) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_rank_boost, para_space)
    elif method == 'adaboost':
        para_space = [(data, trial_i) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_adaboost, para_space)
    elif method == 'opt_auc':
        para_space = [(data, trial_i) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_opt_auc, para_space)
    elif method == 'opt_auc_3d':
        para_space = [(data, trial_i) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_opt_auc_3d, para_space)
    elif method == 'c_svm':
        para_space = [(data, trial_i, None) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_c_svm, para_space)
    elif method == 'b_c_svm':
        para_space = [(data, trial_i, 'balanced') for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_c_svm, para_space)
    elif method == 'svm_perf_lin':
        para_space = [(data, trial_i, 'linear') for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_svm_perf, para_space)
    elif method == 'svm_perf_rbf':
        para_space = [(data, trial_i, 'rbf') for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_svm_perf, para_space)
    elif method == 'rbf_svm':
        para_space = [(data, trial_i, None) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_rbf_svm, para_space)
    elif method == 'b_rbf_svm':
        para_space = [(data, trial_i, 'balanced') for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_rbf_svm, para_space)
    elif method == 'lr':
        para_space = [(data, trial_i, None) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_lr, para_space)
    elif method == 'b_lr':
        para_space = [(data, trial_i, 'balanced') for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_lr, para_space)
    elif method == 'rf':
        para_space = [(data, trial_i, None) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_rf, para_space)
    elif method == 'b_rf':
        para_space = [(data, trial_i, 'balanced') for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_rf, para_space)
    elif method == 'gb':
        para_space = [(data, trial_i) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_gb, para_space)
    elif method == 'spam':
        para_space = [(data, trial_i) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_spam_l2, para_space)
    elif method == 'spauc':
        para_space = [(data, trial_i) for trial_i in range(num_trials)]
        results_pool = pool.map(run_algo_spauc_l2, para_space)
    else:
        results_pool = None
    pool.close()
    pool.join()
    print(np.mean([results_pool[_][_][method]['tr']['auc'] for _ in range(num_trials)]))
    print(np.mean([results_pool[_][_][method]['te1']['auc'] for _ in range(num_trials)]))
    print(np.mean([results_pool[_][_][method]['te2']['auc'] for _ in range(num_trials)]))
    print(np.mean([results_pool[_][_][method]['te3']['auc'] for _ in range(num_trials)]))
    pkl.dump(reduce(lambda a, b: {**a, **b}, results_pool),
             open(root_path + 'datasets/%s/results_%s_%s_%s.pkl' % (dataset, dtype, dataset, method), 'wb'))


def test_icml21():
    if sys.argv[1] == 'single':
        parallel_by_method_dataset(dtype=sys.argv[2], dataset=sys.argv[3],
                                   method=sys.argv[4], num_cpus=int(sys.argv[5]))
    elif sys.argv[1] == 'single_trial':
        dtype = sys.argv[2]
        dataset = sys.argv[3]
        num_trials = 210
        split_ratio = 0.5
        data = get_data(dtype=dtype, dataset=dataset, num_trials=num_trials, split_ratio=split_ratio)
        run_algo_spauc_l2((data, int(sys.argv[4])))
    elif sys.argv[1] == 'part':
        list_method = ['spauc', 'svm_perf_lin', 'svm_perf_rbf', 'rbf_svm', 'b_rbf_svm']
        for method in list_method:
            parallel_by_method_dataset(dtype=sys.argv[2], dataset=sys.argv[3],
                                       method=method, num_cpus=int(sys.argv[4]))
    elif sys.argv[1] == 'all':
        if sys.argv[2] == 'tsne':
            list_method = ['opt_auc', 'rank_boost', 'adaboost', 'c_svm', 'b_c_svm', 'lr', 'b_lr', 'rf', 'b_rf', 'gb',
                           'spam', 'spauc', 'svm_perf_lin', 'rbf_svm', 'b_rbf_svm']
        else:
            list_method = ['rank_boost', 'adaboost', 'c_svm', 'b_c_svm', 'lr', 'b_lr', 'rf', 'b_rf', 'gb',
                           'spam', 'spauc', 'svm_perf_lin', 'rbf_svm', 'b_rbf_svm']
        for method in list_method:
            parallel_by_method_dataset(dtype=sys.argv[2], dataset=sys.argv[3],
                                       method=method, num_cpus=int(sys.argv[4]))


def main():
    dtype = "tsne-3d"
    num_cpus = 25
    for dataset in [sys.argv[1]]:
        if sys.argv[2] == "baseline":
            for method in ["c_svm", "b_c_svm", "lr", "b_lr", "svm_perf_lin", "spauc", "spam"]:
                parallel_by_method_dataset(dtype=dtype, dataset=dataset, method=method, num_cpus=num_cpus)
        if sys.argv[2] == "opt-auc":
            parallel_by_method_dataset(dtype=dtype, dataset=dataset, method="opt_auc_3d", num_cpus=num_cpus)


if __name__ == '__main__':
    dtype = "tsne-3d"
    num_cpus = 19
    # for dataset in ["arrhythmia_06", "australian", "banana", "breast_cancer", "car_eval_34", "cardio_3"]:
    for dataset in ["car_eval_4", "coil_2000", "ecoli_imu", "fourclass", "german", "ionophere", "pima"]:
        parallel_by_method_dataset(dtype=dtype, dataset=dataset, method="opt_auc_3d", num_cpus=num_cpus)
