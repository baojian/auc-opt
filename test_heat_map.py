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


def get_z_scores(x_tr, h, results):
    x_min, x_max = x_tr[:, 0].min() - .5, x_tr[:, 0].max() + .5
    y_min, y_max = x_tr[:, 1].min() - .5, x_tr[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    z_scores = dict()
    for ind, method in enumerate(['opt_auc', 'lr', 'b_lr', 'c_svm', 'b_c_svm', 'spam', 'spauc']):
        w = results[method][method]['w']
        if method == 'lr' or method == 'b_lr':
            w = results[method][method]['w'].coef_[0]
        if method == 'c_svm' or method == 'b_c_svm':
            w = results[method][method]['w'].coef_[0]
        z_values = np.dot(mesh_points, np.asarray(w)).reshape(xx.shape)
        range_z = np.max(z_values) - np.min(z_values)
        z_values = 2. * (z_values - np.min(z_values)) / range_z - 1.
        z_scores[method] = z_values
    for ind, method in enumerate(['gb', 'adaboost', 'rf', 'rank_boost', 'svm_perf_rbf', 'rbf_svm', 'svm_perf_lin']):
        if method == 'gb' or method == 'rf':
            z_values = results[method][method]['w'].predict_proba(X=mesh_points)[:, 1].reshape(xx.shape)
        elif method == 'adaboost' or method == 'rbf_svm':
            z_values = results[method][method]['w'].decision_function(X=mesh_points).reshape(xx.shape)
        elif method == 'rank_boost':
            alpha, threshold, rank_feat = results[method][method]['w']['alpha_threshold_rankfeat']
            z_values = decision_func_rank_boost(mesh_points, alpha, threshold, rank_feat).reshape(xx.shape)
        elif method == 'svm_perf_rbf' or method == 'svm_perf_lin':
            z_values = decision_func_svm_perf(mesh_points, np.ones(len(mesh_points)),
                                              results[method][method]['w']).reshape(xx.shape)
        else:
            z_values = results[method][method]['w'].predict_proba(X=mesh_points)[:, 1].reshape(xx.shape)
        range_z = np.max(z_values) - np.min(z_values)
        z_values = 2. * (z_values - np.min(z_values)) / range_z - 1.
        z_scores[method] = z_values
    return z_scores


def draw_heat_map_fig(results, x_tr, y_tr, x_te, y_te, flag='te', show_bar=False):
    (fig, ax), h = plt.subplots(2, 7, figsize=(14, 4)), 0.05
    fig.patch.set_visible(False)
    from matplotlib import rcParams
    rcParams['axes.titlepad'] = 5
    for i, j in product(range(2), range(7)):
        ax[i, j].spines["top"].set_visible(False)
        ax[i, j].spines["left"].set_visible(False)
        ax[i, j].spines["right"].set_visible(False)
        ax[i, j].spines["bottom"].set_visible(False)
        if flag == 'te':
            ax[i, j].scatter(x_tr[:, 0][np.argwhere(y_tr > 0)],
                             x_tr[:, 1][np.argwhere(y_tr > 0)], c='r', s=2, label='posi')
            ax[i, j].scatter(x_tr[:, 0][np.argwhere(y_tr < 0)],
                             x_tr[:, 1][np.argwhere(y_tr < 0)], c='b', s=2, label='nega')
        else:
            ax[i, j].scatter(x_te[:, 0][np.argwhere(y_te > 0)],
                             x_te[:, 1][np.argwhere(y_te > 0)], c='r', s=2, label='posi')
            ax[i, j].scatter(x_te[:, 0][np.argwhere(y_te < 0)],
                             x_te[:, 1][np.argwhere(y_te < 0)], c='b', s=2, label='nega')
    x_min, x_max = x_tr[:, 0].min() - .5, x_tr[:, 0].max() + .5
    y_min, y_max = x_tr[:, 1].min() - .5, x_tr[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z_scores = get_z_scores(x_tr, h, results)
    label_list = ['LR', 'B-LR', 'C-SVM', 'B-C-SVM', 'GB', 'RF', 'RBF-SVM']
    for ind, method in enumerate(['lr', 'b_lr', 'c_svm', 'b_c_svm', 'gb', 'rf', 'rbf_svm']):
        if flag == 'tr':
            auc = results[method][method]['tr']['auc']
        else:
            auc = results[method][method]['te']['auc']
        z_values = z_scores[method]
        print(np.min(z_values), np.max(z_values))
        pp = np.arange(x_min, x_max, h)
        if method in ['lr', 'b_lr', 'c_svm', 'b_c_svm']:
            w = results[method][method]['w'].coef_[0]
            # ax[0, ind].plot(pp, [-(w[0] * _) / w[1] for _ in pp], linestyle='dashed', zorder=-1)
        ax[0, ind].contourf(xx, yy, z_values, cmap=plt.cm.RdBu_r, alpha=1., levels=20, zorder=-1)
        ax[0, ind].contour(xx, yy, z_values, 20, linewidths=0.5, colors='k', alpha=0.1, zorder=-1)
        ax[0, ind].set_title('%s' % (label_list[ind]))
        if show_bar:
            plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=mpl.colors.Normalize(
                vmin=np.min(z_values), vmax=np.max(z_values))), ax=ax[0, ind], orientation="horizontal")
        ax[0, ind].set_xlim(xx.min(), xx.max())
        ax[0, ind].set_ylim(yy.min(), yy.max())
        ax[0, ind].set_xticks(())
        ax[0, ind].set_yticks(())
    label_list = ['LO-AUC', 'SPAM', 'SPAUC', 'SVM-Perf-Lin', 'SVM-Perf-RBF', 'AdaBoost', 'RankBoost']
    for ind, method in enumerate(['opt_auc', 'spam', 'spauc', 'svm_perf_lin',
                                  'svm_perf_rbf', 'adaboost', 'rank_boost']):
        if flag == 'tr':
            auc = results[method][method]['tr']['auc']
        else:
            auc = results[method][method]['te']['auc']
        z_values = z_scores[method]
        print(np.min(z_values), np.max(z_values))
        pp = np.arange(x_min, x_max, h)
        if method in ['opt_auc', 'spam', 'spauc']:
            w = results[method][method]['w']
            # ax[1, ind].plot(pp, [-(w[0] * _) / w[1] for _ in pp], linestyle='dashed', zorder=-1)
        if method in ['svm_perf_lin']:
            w = results[method][method]['w']
            w = [float(_.split(b':')[1]) for _ in w[-1].split(b' ')[1:3]]
            # ax[1, ind].plot(pp, [-(w[0] * _) / w[1] for _ in pp], linestyle='dashed', zorder=-1)
        ax[1, ind].contourf(xx, yy, z_values, cmap=plt.cm.RdBu_r, alpha=1., levels=20, zorder=-1)
        ax[1, ind].contour(xx, yy, z_values, 20, linewidths=0.5, colors='k', alpha=0.1, zorder=-1)
        ax[1, ind].set_title('%s' % (label_list[ind]))
        if show_bar:
            plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=mpl.colors.Normalize(
                vmin=np.min(z_values), vmax=np.max(z_values))), ax=ax[1, ind], orientation="horizontal")
        ax[1, ind].set_xlim(xx.min(), xx.max())
        ax[1, ind].set_ylim(yy.min(), yy.max())
        ax[1, ind].set_xticks(())
        ax[1, ind].set_yticks(())
    file_name = 'heat_map_fourclass.pdf'
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s' % file_name
    plt.subplots_adjust(wspace=0.01, hspace=0.15)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()


def test_on_fourclass():
    file_name = '/data/auc-logistic/results/real/fourclass_heat_map.pkl'
    if os.path.exists(file_name):
        return pkl.load(open(file_name, 'rb'))
    dataset, num_trials = 'fourclass', 1
    data = get_data(dataset=dataset, num_trials=num_trials)
    trial_id = 0
    results = dict()
    results['opt_auc'] = run_algo_opt_auc((data, trial_id))[trial_id]
    results['lr'] = run_algo_lr((data, trial_id, None))[trial_id]
    results['b_lr'] = run_algo_lr((data, trial_id, 'balanced'))[trial_id]
    results['c_svm'] = run_algo_c_svm((data, trial_id, None))[trial_id]
    results['b_c_svm'] = run_algo_c_svm((data, trial_id, 'balanced'))[trial_id]
    results['gb'] = run_algo_gb((data, trial_id))[trial_id]
    results['rf'] = run_algo_rf((data, trial_id))[trial_id]
    results['adaboost'] = run_algo_adaboost((data, trial_id))[trial_id]
    results['rank_boost'] = run_algo_rank_boost((data, trial_id))[trial_id]
    results['rbf_svm'] = run_algo_rbf_svm((data, trial_id))[trial_id]
    results['spam'] = run_algo_spam_l2((data, trial_id))[trial_id]
    results['spauc'] = run_algo_spauc_l2((data, trial_id))[trial_id]
    results['svm_perf_lin'] = run_algo_svm_perf((data, trial_id, 'linear'))[trial_id]
    results['svm_perf_rbf'] = run_algo_svm_perf((data, trial_id, 'rbf'))[trial_id]
    pkl.dump(results, open(file_name, 'wb'))
    return results


def main():
    file_name = '/data/auc-logistic/results/real/fourclass_heat_map.pkl'
    results = pkl.load(open(file_name, 'rb'))
    for method in results:
        print('%15s %.10f %.10f' % (method, results[method][method]['tr']['auc'],
                                    results[method][method]['te']['auc']))
    dataset, num_trials, trial_id = 'fourclass', 1, 0
    data = get_data(dataset=dataset, num_trials=num_trials)
    tr_index = data['trial_%d_tr_indices' % trial_id]
    te_index = data['trial_%d_te_indices' % trial_id]
    x_te, y_te = data['x_tr'][te_index], data['y_tr'][te_index]
    x_tr, y_tr = data['x_tr'][tr_index], data['y_tr'][tr_index]
    draw_heat_map_fig(results, x_tr, y_tr, x_te, y_te)


if __name__ == '__main__':
    main()
