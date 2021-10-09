# -*- coding: utf-8 -*-
import numpy as np
import time
from itertools import product
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

try:
    from libopt_auc_3 import c_auc_score
    from libopt_auc_3 import c_roc_curve
    from libopt_auc_3 import c_opt_auc
except ImportError:
    pass


def linear_auc_opt(x_tr, y_tr, min_eps=2e-16):
    """
    --- OptAUC ---
    Give a set of points, the algorithm learn a linear classifier
    so that the AUC score is maximized. This only works for 2 dimension.
    TODO need to consider the following:
    1. the vertical lines
    2. the prevision between lines.
    """
    start_time = time.time()
    x_tr = np.asarray(x_tr, dtype=np.float64)
    y_tr = np.asarray(y_tr, dtype=np.float64)
    n, d = x_tr.shape
    t_posi = len(np.argwhere(y_tr > 0.0))
    t_nega = len(y_tr) - t_posi
    slope_dict, min_slope, max_slope = dict(), np.inf, -np.inf
    for i, j in product(range(n), range(n)):
        diff = x_tr[i] - x_tr[j]
        if i >= j or diff[0] == 0.0 or y_tr[i] == y_tr[j]:
            continue
        else:
            slope = -diff[1] / diff[0]
            min_slope = min(min_slope, slope)
            max_slope = max(max_slope, slope)
            if slope not in slope_dict:
                slope_dict[slope] = {-1: set(), 1: set()}
            slope_dict[slope][y_tr[i]].add(i)
            slope_dict[slope][y_tr[j]].add(j)
    granularity = 1. / (t_posi * t_nega)
    prev_eps, cur_eps = 1., 1.
    w_init = np.asarray([float(min_slope - prev_eps), 1.])
    auc = roc_auc_score(y_true=y_tr, y_score=np.dot(x_tr, w_init))
    auc_opt, w_opt = auc, w_init
    sorted_list = sorted(slope_dict.items(), key=lambda __: __[0])
    updated_list = [[sorted_list[0][0], sorted_list[0][1]]]
    eps_list = []
    for ii, item in enumerate(sorted_list[1:]):
        next_slope = float(sorted_list[ii + 1][0])
        cur_slope = float(sorted_list[ii][0])
        eps = (next_slope - cur_slope) / 2.
        if eps <= min_eps:
            updated_list[-1][1][-1] = updated_list[-1][1][-1].union(
                item[1][-1])
            updated_list[-1][1][1] = updated_list[-1][1][1].union(item[1][1])
        else:
            updated_list.append([item[0], item[1]])
            eps_list.append(eps)
    eps_list.append(1.)
    del sorted_list
    del slope_dict
    for ind, (slope, candidates) in enumerate(updated_list):
        cur_eps = eps_list[ind]
        w_cur = np.asarray([slope + cur_eps, 1.])
        if len(candidates[1]) == 1 and len(candidates[-1]) == 1:
            ii = list(candidates[1])[0]
            jj = list(candidates[-1])[0]
            if np.dot(x_tr[ii], w_cur) > np.dot(x_tr[jj], w_cur):
                c = 1.0
            else:
                c = -1.0
        else:
            # some collinear, parallel, or duplicated points occur.
            w_prev = np.asarray([slope - prev_eps, 1.])
            list_posi_cur = np.dot(x_tr[list(candidates[1]), :], w_cur)
            list_nega_cur = np.dot(x_tr[list(candidates[-1]), :], w_cur)
            list_posi_prev = np.dot(x_tr[list(candidates[1]), :], w_prev)
            list_nega_prev = np.dot(x_tr[list(candidates[-1]), :], w_prev)
            tol_correct_cur = 0
            for score_i, score_j in product(list_posi_cur, list_nega_cur):
                if score_i > score_j:
                    tol_correct_cur += 1
            tol_correct_prev = 0
            for score_i, score_j in product(list_posi_prev, list_nega_prev):
                if score_i > score_j:
                    tol_correct_prev += 1
            c = tol_correct_cur - tol_correct_prev
        auc = auc + c * granularity
        if auc_opt < auc:
            auc_opt = auc
            w_opt = w_cur
        if auc_opt < (1. - auc):
            auc_opt = 1. - auc
            w_opt = -w_cur
        prev_eps = cur_eps
    return auc_opt, w_opt, time.time() - start_time


def get_2d_samples(n, p=2, nega_mu=0.0, posi_mu=1.0, scale=1.,
                   rand_state=17, posi_ratio=0.1):
    np.random.seed(rand_state)
    x_tr = np.random.normal(loc=nega_mu, scale=scale, size=(n, p))
    x_tr[0:int(posi_ratio * n), :] += posi_mu
    y_tr = -np.ones(n)
    y_tr[0:int(posi_ratio * n)] = 1.0
    x_te = np.random.normal(loc=nega_mu, scale=scale, size=(n, p))
    x_te[0:int(posi_ratio * n), :] += posi_mu
    y_te = -np.ones(n)
    y_te[0:int(posi_ratio * n)] = 1.0
    return x_tr, y_tr, x_te, y_te


def random_test():
    from sklearn.metrics import auc
    np.random.seed(17)
    n, d, num_trials, max_tol = 1000, 10, 1000, 1e-15
    for trial_i in range(num_trials):
        y_tr = np.ones(shape=n)
        y_tr[np.random.permutation(n)[:n // 2]] = -1
        x_tr = np.random.normal(loc=0.0, scale=1.0, size=(n, d))
        w = np.random.normal(loc=0.0, scale=1., size=d)
        scores = np.dot(x_tr, w)
        auc_true = roc_auc_score(y_true=y_tr, y_score=scores)
        _, _, auc_test = c_roc_curve(y_tr, scores)
        if np.abs(auc_true - auc_test) < max_tol:
            print('trial-%03d pass gap: %.6e' %
                  (trial_i, auc_true - auc_test), end=' ')
        else:
            print('trial-%03d fail gap: %.6e' %
                  (trial_i, auc_true - auc_test), end=' ')
        fpr, tpr, auc2 = c_roc_curve(y_tr, scores)
        print('%.6e %.6e' % (auc(x=fpr, y=tpr) - auc_true,
                             (auc2 - auc_true)))


def run_test_auc():
    n, posi_ratio = 2000, 0.5
    for rand_state in range(100):
        x_tr, y_tr, x_te, y_te = get_2d_samples(
            n=n, rand_state=rand_state, posi_ratio=posi_ratio)
        w, auc, c_run_time = c_opt_auc(x_tr, y_tr, 2e-16)
        t_auc, t_w, py_run_time = linear_auc_opt(x_tr, y_tr, 2e-16)
        speedup = py_run_time / c_run_time
        print('auc-gap: %.4e speedup: %03.2f %.2f'
              % (np.abs(auc - t_auc), speedup, c_run_time))


def main():
    run_test_auc()


if __name__ == '__main__':
    main()
