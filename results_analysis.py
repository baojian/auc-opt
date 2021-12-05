# -*- coding: utf-8 -*-
import os
import sys
import time
import operator
import warnings
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.stats import ttest_ind
from sklearn.metrics import adjusted_rand_score

root_path = "/data/auc-opt-datasets/datasets/"


# root_path = "/home/baojian/data/aistats22-auc-opt/datasets/"


def get_summarized_data(dataset_name, dtype, num_trials, perplexity):
    if dtype == 'real':
        method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'rbf_svm', 'b_rbf_svm', 'rf', 'b_rf',
                       'spauc', 'spam', 'gb', 'rank_boost', 'adaboost', 'svm_perf_lin', 'svm_perf_rbf']
    elif dtype == 'tsne-2d':
        method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'opt_auc_2d', 'spauc', 'spam', 'svm_perf_lin']
    elif dtype == 'tsne-3d':
        method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'opt_auc_3d', 'opt_auc_3d_reg', 'spauc', 'spam', 'svm_perf_lin']
    else:
        return 0
    results = dict()
    dataset_list = [dataset_name]
    for dataset in dataset_list:
        results[dataset] = dict()
        results[dataset][dtype] = dict()
        for trial_i in range(num_trials):
            results[dataset][dtype][trial_i] = dict()
            for method in method_list:
                results[dataset][dtype][trial_i][method] = dict()
    for dataset in dataset_list:
        for method in method_list:
            file = root_path + '%s/results_%s_%s_%s_%d.pkl' % (dataset, dtype, dataset, method, perplexity)
            if os.path.exists(file):
                re = pkl.load(open(file, 'rb'))
                print(dataset, dtype, method)
                for trial_i in re:
                    for label in ['tr', 'te1', 'te2', 'te3']:
                        results[dataset][dtype][trial_i][method][label] = re[trial_i][method][label]
                        print(label, re[trial_i][method][label]['auc'])
            else:
                print(file)
            print('------vs------')
        print('-------------')
        file = root_path + '%s/all_results_%s_%s_%d.pkl' % (dataset, dtype, dataset, perplexity)
        pkl.dump(results, open(file, 'wb'))


def show_tr_auc_te_auc(dataset=None):
    line_style = ['-', '-', '-', '-', '-', '-', '-', '-', '--', '--', '--', '--', '--', '--', '--', '--']
    num_trials = 210
    list_datasets = [dataset]
    for dataset in list_datasets:
        results = pkl.load(open(root_path + '%s/results_all_%s.pkl' % (dataset, dataset), 'rb'))
        for tag in ['real', 'tsne']:
            if tag == 'real':
                method_list = ['rank_boost', 'adaboost', 'spam', 'spauc', 'svm_perf_lin', 'svm_perf_rbf',
                               'c_svm', 'b_c_svm', 'lr', 'b_lr', 'rf', 'b_rf', 'gb', 'rbf_svm', 'b_rbf_svm']
                method_label_list = ['RankBoost', 'AdaBoost', 'SPAM', 'SPAUC', 'SVM-PL', 'SVM-PNL',
                                     'C-SVM', 'B-SVM', 'LR', 'B-LR', 'RF', 'B-RF', 'GB', 'RBF-SVM', 'B-RBF-SVM']
            else:
                method_list = ['opt_auc', 'rank_boost', 'adaboost', 'spam', 'spauc', 'svm_perf_lin', 'svm_perf_rbf',
                               'c_svm', 'b_c_svm', 'lr', 'b_lr', 'rf', 'b_rf', 'gb', 'rbf_svm', 'b_rbf_svm']
                method_label_list = ['LO-AUC', 'RankBoost', 'AdaBoost', 'SPAM', 'SPAUC', 'SVM-PL', 'SVM-PNL',
                                     'C-SVM', 'B-SVM', 'LR', 'B-LR', 'RF', 'B-RF', 'GB', 'RBF-SVM', 'B-RBF-SVM']
            fig, ax = plt.subplots(2, 2, figsize=(16, 12))
            for ind, method in enumerate(method_list):
                if len(results[dataset][tag][0][method]) != 0:
                    re = sorted([results[dataset][tag][_][method]['tr']['auc'] for _ in range(num_trials)])[5:205]
                    print(dataset, tag, method, 'train', re[:5], np.mean(re))
                    ax[0, 0].plot(re, label=method_label_list[ind], linestyle=line_style[ind])
                    ax[0, 0].set_title('training AUC')
            for ind, method in enumerate(method_list):
                if len(results[dataset][tag][0][method]) != 0:
                    re = sorted([results[dataset][tag][_][method]['te1']['auc'] for _ in range(num_trials)])[5:205]
                    print(dataset, tag, method, 'test', re[:5], np.mean(re))
                    ax[0, 1].plot(re, label=method_label_list[ind], linestyle=line_style[ind])
                    ax[0, 1].set_title('testing 1 AUC')
            for ind, method in enumerate(method_list):
                if len(results[dataset][tag][0][method]) != 0:
                    re = sorted([results[dataset][tag][_][method]['te2']['auc'] for _ in range(num_trials)])[5:205]
                    ax[1, 0].plot(re, label=method_label_list[ind], linestyle=line_style[ind])
                    ax[1, 0].set_title('testing 2 AUC')
            for ind, method in enumerate(method_list):
                if len(results[dataset][tag][0][method]) != 0:
                    re = sorted([results[dataset][tag][_][method]['te3']['auc'] for _ in range(num_trials)])[5:205]
                    ax[1, 1].plot(re, label=method_label_list[ind], linestyle=line_style[ind])
                    ax[1, 1].set_title('testing 3 AUC')
            ax[0, 0].legend(loc='lower right', ncol=2)
            ax[0, 1].legend(loc='lower right', ncol=2)
            ax[1, 0].legend(loc='lower right', ncol=2)
            ax[1, 1].legend(loc='lower right', ncol=2)
            f_name = root_path + '%s/fig_%s_auc.pdf' % (dataset, tag)
            plt.subplots_adjust(wspace=0.1, hspace=.15)
            fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.1, format='pdf')
            plt.close()


def show_tr_acc_f1_te_acc_f1(dataset=None):
    line_style = ['-', '-', '-', '-', '-', '-', '-', '-', '--', '--', '--', '--', '--', '--', '--', '--']
    num_trials = 210
    list_datasets = [dataset]
    for dataset in list_datasets:
        results = pkl.load(open(root_path + '%s/results_all_%s.pkl' % (dataset, dataset), 'rb'))
        for tag in ['real', 'tsne']:
            if tag == 'real':
                method_list = ['rank_boost', 'adaboost', 'spam', 'spauc', 'svm_perf_lin', 'svm_perf_rbf',
                               'c_svm', 'b_c_svm', 'lr', 'b_lr', 'rf', 'b_rf', 'gb', 'rbf_svm', 'b_rbf_svm']
                method_label_list = ['RankBoost', 'AdaBoost', 'SPAM', 'SPAUC', 'SVM-PL', 'SVM-PNL',
                                     'C-SVM', 'B-SVM', 'LR', 'B-LR', 'RF', 'B-RF', 'GB', 'RBF-SVM', 'B-RBF-SVM']
            else:
                method_list = ['opt_auc', 'rank_boost', 'adaboost', 'spam', 'spauc', 'svm_perf_lin', 'svm_perf_rbf',
                               'c_svm', 'b_c_svm', 'lr', 'b_lr', 'rf', 'b_rf', 'gb', 'rbf_svm', 'b_rbf_svm']
                method_label_list = ['LO-AUC', 'RankBoost', 'AdaBoost', 'SPAM', 'SPAUC', 'SVM-PL', 'SVM-PNL',
                                     'C-SVM', 'B-SVM', 'LR', 'B-LR', 'RF', 'B-RF', 'GB', 'RBF-SVM', 'B-RBF-SVM']
            for metric in ['b_acc', 'f1']:
                fig, ax = plt.subplots(2, 2, figsize=(16, 12))
                for ind, method in enumerate(method_list):
                    if len(results[dataset][tag][0][method]) != 0:
                        re = sorted([results[dataset][tag][_][method]['tr'][metric] for _ in range(num_trials)])[5:205]
                        print(dataset, tag, method, 'train', re[:5], np.mean(re))
                        ax[0, 0].plot(re, label=method_label_list[ind], linestyle=line_style[ind])
                        ax[0, 0].set_title('training %s' % metric)
                for ind, method in enumerate(method_list):
                    if len(results[dataset][tag][0][method]) != 0:
                        re = sorted([results[dataset][tag][_][method]['te1'][metric] for _ in range(num_trials)])[
                             5:205]
                        print(dataset, tag, method, 'test', re[:5], np.mean(re))
                        ax[0, 1].plot(re, label=method_label_list[ind], linestyle=line_style[ind])
                        ax[0, 1].set_title('testing 1 %s' % metric)
                for ind, method in enumerate(method_list):
                    if len(results[dataset][tag][0][method]) != 0:
                        re = sorted([results[dataset][tag][_][method]['te2'][metric] for _ in range(num_trials)])[
                             5:205]
                        ax[1, 0].plot(re, label=method_label_list[ind], linestyle=line_style[ind])
                        ax[1, 0].set_title('testing 2 %s' % metric)
                for ind, method in enumerate(method_list):
                    if len(results[dataset][tag][0][method]) != 0:
                        re = sorted([results[dataset][tag][_][method]['te3'][metric] for _ in range(num_trials)])[
                             5:205]
                        ax[1, 1].plot(re, label=method_label_list[ind], linestyle=line_style[ind])
                        ax[1, 1].set_title('testing 3 %s' % metric)
                ax[0, 0].legend(loc='lower right', ncol=2)
                ax[0, 1].legend(loc='lower right', ncol=2)
                ax[1, 0].legend(loc='lower right', ncol=2)
                ax[1, 1].legend(loc='lower right', ncol=2)
                f_name = root_path + '%s/fig_%s_%s.pdf' % (dataset, tag, metric)
                plt.subplots_adjust(wspace=0.1, hspace=.15)
                fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.1, format='pdf')
                plt.close()


def tsne_results(tag):
    list_datasets = [
        'abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer', 'cardio_3',
        'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german', 'ionosphere',
        'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
        'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks', 'pen_digits_0', 'pen_digits_5', 'pima',
        'satimage_4', 'scene', 'seismic', 'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf',
        'spectrometer', 'splice', 'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
        'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt', 'yeast_me1', 'yeast_me2', 'yeast_ml8']
    method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'spauc', 'spam', 'svm_perf_lin', 'opt_auc']
    method_label_list = ['SVM', 'B-SVM', 'LR', 'B-LR', 'SPAUC', 'SPAM', 'SVM-Perf', 'AUC-opt']
    rank_tr_auc_matrix = []
    tr_auc_matrix = []
    rank_te_auc_matrix = []
    te_auc_matrix = []
    te_b_acc_matrix = []
    rank_te_b_acc_matrix = []
    te_f1_matrix = []
    rank_te_f1_matrix = []
    num_trials = 210
    eff_range = range(5, num_trials - 5)
    for ind, dataset in enumerate(list_datasets):
        results = pkl.load(open(root_path + '%s/results_all_%s.pkl' % (dataset, dataset), 'rb'))
        list_vals = []
        for ind_method, method in enumerate(method_list):
            list_vals.append(np.mean([results[dataset]['tsne'][_][method]['tr']['auc'] for _ in eff_range]))
        list_vals = np.asarray(list_vals)
        tr_auc_matrix.append(list_vals)
        temp = (-np.asarray(list_vals)).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(list_vals))
        rank_tr_auc_matrix.append(ranks + 1)

        list_vals = []
        for ind_method, method in enumerate(method_list):
            list_vals.append(np.mean([results[dataset]['tsne'][_][method]['te1']['auc'] for _ in eff_range]))
        te_auc_matrix.append(list_vals)
        temp = (-np.asarray(list_vals)).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(list_vals))
        rank_te_auc_matrix.append(ranks + 1)

        list_vals = []
        for ind_method, method in enumerate(method_list):
            list_vals.append(np.mean([results[dataset]['tsne'][_][method]['te1']['b_acc'] for _ in eff_range]))
        te_b_acc_matrix.append(list_vals)
        temp = (-np.asarray(list_vals)).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(list_vals))
        rank_te_b_acc_matrix.append(ranks + 1)

        list_vals = []
        for ind_method, method in enumerate(method_list):
            list_vals.append(np.mean([results[dataset]['tsne'][_][method]['te1']['f1'] for _ in eff_range]))
        te_f1_matrix.append(list_vals)
        temp = (-np.asarray(list_vals)).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(list_vals))
        rank_te_f1_matrix.append(ranks + 1)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.))
    color = 'tab:red'
    if tag == 'tr_auc':
        re_mat = np.asarray(tr_auc_matrix)
        rank_mat = np.asarray(rank_tr_auc_matrix)
        ax.set_yticks([0.72, 0.726, 0.732, 0.738, 0.744, 0.75])
        ax.set_ylim([0.715, 0.753])
        ax.set_xticks(range(8))
        print(sorted(np.mean(re_mat, axis=0))[::-1])
        plt.plot(sorted(np.mean(re_mat, axis=0))[::-1],
                 color=color, marker='o', markersize=4, label='Training AUC')
        ax.tick_params(axis='y', direction='in', labelcolor=color, color=color)
        ax.tick_params(axis='x', direction='in')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticklabels([method_label_list[_] for _ in np.argsort(np.mean(re_mat, axis=0))[::-1]],
                           rotation=30, fontsize=8)
        ax.grid(color='gray', linewidth=0.4, linestyle='--', dashes=(5, 5))
        color = 'tab:blue'
        ax2 = ax.twinx()
        ax2.set_ylim([0.42, 7.99])
        ax2.set_xlim([-0.9, 7.9])
        ax2.plot(sorted(np.mean(rank_mat, axis=0)),
                 color=color, marker='D', markersize=4, label='Friedman Rank')
        ax2.tick_params(axis='y', direction='in', labelcolor=color, color=color)
        f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/tsne_rank_train_auc.pdf'
        ax.legend(loc='lower center', bbox_to_anchor=(0.47, 0.1), frameon=True, framealpha=1., edgecolor='white')
        ax2.legend(loc='lower center', frameon=True, framealpha=1.0, edgecolor='white')
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
        plt.close()
    elif tag == 'te_auc':
        re_mat = te_auc_matrix
        rank_mat = rank_te_auc_matrix
        ax.set_yticks([0.715, 0.72, 0.725, 0.73, 0.735, 0.74])
        ax.set_ylim([0.71, 0.745])
        ax.set_xticks(range(8))
        print(sorted(np.mean(re_mat, axis=0))[::-1])
        plt.plot(sorted(np.mean(re_mat, axis=0))[::-1],
                 color=color, marker='o', markersize=4, label='Testing AUC')
        ax.tick_params(axis='y', direction='in', labelcolor=color, color=color)
        ax.tick_params(axis='x', direction='in')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticklabels([method_label_list[_] for _ in np.argsort(np.mean(re_mat, axis=0))[::-1]],
                           rotation=30, fontsize=8)
        ax.grid(color='gray', linewidth=0.4, linestyle='--', dashes=(5, 5))
        color = 'tab:blue'
        ax2 = ax.twinx()
        ax2.set_ylim([1.6, 7.39])
        ax2.set_xlim([-.9, 7.9])
        ax2.plot(sorted(np.mean(rank_mat, axis=0)),
                 color=color, marker='D', markersize=4, label='Friedman Rank')
        ax2.tick_params(axis='y', direction='in', labelcolor=color, color=color)
        f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/tsne_rank_test_auc.pdf'
        ax.legend(loc='lower center', bbox_to_anchor=(0.47, 0.1), frameon=True, framealpha=1., edgecolor='white')
        ax2.legend(loc='lower center', frameon=True, framealpha=1.0, edgecolor='white')
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
        plt.close()
    elif tag == 'b_acc':
        re_mat = te_b_acc_matrix
        rank_mat = rank_te_b_acc_matrix
        ax.set_yticks([0.55, 0.58, 0.61, 0.64, 0.67, 0.70, 0.73])
        ax.set_ylim([0.535, 0.745])
        ax.set_xticks(range(8))
        print(sorted(np.mean(re_mat, axis=0))[::-1])
        plt.plot(sorted(np.mean(re_mat, axis=0))[::-1],
                 color=color, marker='o', markersize=4, label='Balanced Accuracy')
        ax.tick_params(axis='y', direction='in', labelcolor=color, color=color)
        ax.tick_params(axis='x', direction='in')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticklabels([method_label_list[_] for _ in np.argsort(np.mean(re_mat, axis=0))[::-1]],
                           rotation=30, fontsize=8)
        ax.grid(color='gray', linewidth=0.4, linestyle='--', dashes=(5, 5))
        color = 'tab:blue'
        ax2 = ax.twinx()
        ax2.set_ylim([1.5, 8.5])
        ax2.set_xlim([-.9, 7.9])
        ax2.plot(sorted(np.mean(rank_mat, axis=0)),
                 color=color, marker='D', markersize=4, label='Friedman Rank')
        ax2.tick_params(axis='y', direction='in', labelcolor=color, color=color)
        f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/tsne_rank_test_b_acc.pdf'
        ax.legend(loc='center left', bbox_to_anchor=(-0.01, 0.55), frameon=True, framealpha=1., edgecolor='white')
        ax2.legend(loc='center left', bbox_to_anchor=(-0.01, 0.65), frameon=True, framealpha=1.0, edgecolor='white')
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
        plt.close()
    elif tag == 'f1':
        re_mat = te_f1_matrix
        rank_mat = rank_te_f1_matrix
        ax.set_yticks([0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
        ax.set_ylim([0.1, 0.45])
        ax.set_xticks(range(8))
        print(sorted(np.mean(re_mat, axis=0))[::-1])
        plt.plot(sorted(np.mean(re_mat, axis=0))[::-1],
                 color=color, marker='o', markersize=4, label='Testing F1')
        ax.tick_params(axis='y', direction='in', labelcolor=color, color=color)
        ax.tick_params(axis='x', direction='in')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticklabels([method_label_list[_] for _ in np.argsort(np.mean(re_mat, axis=0))[::-1]],
                           rotation=30, fontsize=8)
        ax.grid(color='gray', linewidth=0.4, linestyle='--', dashes=(5, 5))
        color = 'tab:blue'
        ax2 = ax.twinx()
        ax2.set_yticks([2, 3, 4, 5, 6, 7])
        ax2.set_ylim([1., 8])
        ax2.set_xlim([-.9, 7.9])
        ax2.plot(sorted(np.mean(rank_mat, axis=0)),
                 color=color, marker='D', markersize=4, label='Friedman Rank')
        ax2.tick_params(axis='y', direction='in', labelcolor=color, color=color)
        f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/tsne_rank_test_f1.pdf'
        ax.legend(loc='center left', bbox_to_anchor=(-0.01, 0.5), frameon=True, framealpha=1., edgecolor='white')
        ax2.legend(loc='center left', bbox_to_anchor=(-0.01, 0.6), frameon=True, framealpha=1.0, edgecolor='white')
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
        plt.close()


def std_auc(tag):
    list_datasets = [
        'abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer', 'cardio_3',
        'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german', 'ionosphere',
        'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
        'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks', 'pen_digits_0', 'pen_digits_5', 'pima',
        'satimage_4', 'scene', 'seismic', 'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf',
        'spectrometer', 'splice', 'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
        'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt', 'yeast_me1', 'yeast_me2', 'yeast_ml8']
    method_list = ['opt_auc', 'c_svm', 'b_c_svm', 'lr', 'b_lr', 'spauc', 'spam', 'svm_perf_lin']
    method_label_list = ['AUC-opt', 'SVM', 'B-SVM', 'LR', 'B-LR', 'SPAUC', 'SPAM', 'SVM-Perf']
    rank_tr_auc_matrix = []
    tr_auc_matrix = []
    rank_te_auc_matrix = []
    te_auc_matrix = []
    te_b_acc_matrix = []
    rank_te_b_acc_matrix = []
    te_f1_matrix = []
    rank_te_f1_matrix = []
    num_trials = 210
    eff_range = range(5, num_trials - 5)
    for ind, _ in enumerate(list_datasets):
        dataset = _
        results = pkl.load(open(root_path + '%s/results_all_%s.pkl' % (dataset, dataset), 'rb'))
        list_vals = []
        for ind_method, method in enumerate(method_list):
            list_vals.append(np.var([results[dataset]['tsne'][_][method]['tr']['auc'] for _ in eff_range]))
        tr_auc_matrix.append(list_vals)
        temp = (-np.asarray(list_vals)).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(list_vals))
        rank_tr_auc_matrix.append(ranks + 1)
        list_vals = []
        for ind_method, method in enumerate(method_list):
            list_vals.append(np.var([results[dataset]['tsne'][_][method]['te1']['auc'] for _ in eff_range]))
        te_auc_matrix.append(list_vals)
        temp = (-np.asarray(list_vals)).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(list_vals))
        rank_te_auc_matrix.append(ranks + 1)
        list_vals = []
        for ind_method, method in enumerate(method_list):
            list_vals.append(np.var([results[dataset]['tsne'][_][method]['te1']['b_acc'] for _ in eff_range]))
        te_b_acc_matrix.append(list_vals)
        temp = (-np.asarray(list_vals)).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(list_vals))
        rank_te_b_acc_matrix.append(ranks + 1)
        list_vals = []
        for ind_method, method in enumerate(method_list):
            list_vals.append(np.var([results[dataset]['tsne'][_][method]['te1']['f1'] for _ in eff_range]))
        te_f1_matrix.append(list_vals)
        temp = (-np.asarray(list_vals)).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(list_vals))
        rank_te_f1_matrix.append(ranks + 1)
    print([method_list[_] for _ in np.argsort(np.mean(tr_auc_matrix, axis=0))])
    print([method_list[_] for _ in np.argsort(np.mean(te_auc_matrix, axis=0))])
    print([method_list[_] for _ in np.argsort(np.mean(te_b_acc_matrix, axis=0))])
    print([method_list[_] for _ in np.argsort(np.mean(te_f1_matrix, axis=0))])
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.))
    marker_list = ['D', 'v', 'o', 'H', 'P', '*', '>', 's']
    color_list = ['r', 'b', 'g', 'y', 'orange', 'purple', 'm', 'c']
    if tag == 'tr_auc':
        re_mat = np.asarray(tr_auc_matrix)
        part_data = 30
        for ind_method, method in enumerate(method_list):
            plt.plot(range(part_data, len(re_mat)), sorted(re_mat[:, ind_method])[part_data:],
                     linewidth=1., marker=marker_list[ind_method],
                     markersize=4., label=method_label_list[ind_method], color=color_list[ind_method])
        ax.set_ylabel('Variance of Training AUC')
        ax.set_ylim([0.0, 0.015])
        ax.tick_params(axis='y', direction='in')
        ax.tick_params(axis='x', direction='in')
        ax.set_xticks([])
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Dataset')
        ax.grid(color='gray', linewidth=0.4, linestyle='--', dashes=(5, 5))
        f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/tsne_rank_train_auc_var.pdf'
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        ax.legend(ncol=2, handletextpad=0.05, columnspacing=0.1, fontsize=8.,
                  frameon=True, framealpha=1., edgecolor='white')
        fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
        plt.close()
    elif tag == 'te_auc':
        re_mat = np.asarray(te_auc_matrix)
        part_data = 30
        for ind_method, method in enumerate(method_list):
            plt.plot(range(part_data, len(re_mat)), sorted(re_mat[:, ind_method])[part_data:],
                     linewidth=1., marker=marker_list[ind_method],
                     markersize=4., label=method_label_list[ind_method], color=color_list[ind_method])
        ax.set_ylabel('Variance of Testing AUC')
        ax.set_ylim([0.0, 0.015])
        ax.tick_params(axis='y', direction='in')
        ax.tick_params(axis='x', direction='in')
        ax.set_xticks([])
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Dataset')
        ax.grid(color='gray', linewidth=0.4, linestyle='--', dashes=(5, 5))
        f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/tsne_rank_test_auc_var.pdf'
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        ax.legend(ncol=2, handletextpad=0.05, columnspacing=0.1, fontsize=8.,
                  frameon=True, framealpha=1., edgecolor='white')
        fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
        plt.close()
    elif tag == 'b_acc':
        re_mat = np.asarray(te_b_acc_matrix)
        part_data = 30
        for ind_method, method in enumerate(method_list):
            plt.plot(range(part_data, len(re_mat)), sorted(re_mat[:, ind_method])[part_data:],
                     linewidth=1., marker=marker_list[ind_method],
                     markersize=4., label=method_label_list[ind_method], color=color_list[ind_method])
        ax.set_ylabel('Variance of Balanced Accuracy')
        ax.tick_params(axis='y', direction='in')
        ax.tick_params(axis='x', direction='in')
        ax.set_ylim([0.0, 0.03])
        ax.set_xticks([])
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Dataset')
        ax.grid(color='gray', linewidth=0.4, linestyle='--', dashes=(5, 5))
        f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/tsne_rank_test_b_acc_var.pdf'
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        ax.legend(ncol=2, handletextpad=0.05, columnspacing=0.1, fontsize=8.,
                  frameon=True, framealpha=1., edgecolor='white')
        fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
        plt.close()
    elif tag == 'f1':
        re_mat = np.asarray(te_f1_matrix)
        part_data = 30
        for ind_method, method in enumerate(method_list):
            plt.plot(range(part_data, len(re_mat)), sorted(re_mat[:, ind_method])[part_data:],
                     linewidth=1., marker=marker_list[ind_method],
                     markersize=4., label=method_label_list[ind_method], color=color_list[ind_method])
        ax.set_ylabel('Variance of F1 score')
        ax.set_ylim([0.0, 0.06])
        ax.tick_params(axis='y', direction='in')
        ax.tick_params(axis='x', direction='in')
        ax.set_xticks([])
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Dataset')
        ax.grid(color='gray', linewidth=0.4, linestyle='--', dashes=(5, 5))
        f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/tsne_rank_test_f1_var.pdf'
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        ax.legend(ncol=2, handletextpad=0.05, columnspacing=0.1, fontsize=8.,
                  frameon=True, framealpha=1., edgecolor='white')
        fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
        plt.close()


def run_time_tsne(tag):
    list_datasets = [
        'abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer', 'cardio_3',
        'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german', 'ionosphere',
        'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
        'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks', 'pen_digits_0', 'pen_digits_5', 'pima',
        'satimage_4', 'scene', 'seismic', 'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf',
        'spectrometer', 'splice', 'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
        'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt', 'yeast_me1', 'yeast_me2', 'yeast_ml8']
    method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'spauc', 'spam', 'svm_perf_lin', 'opt_auc']
    method_label_list = ['C-SVM', 'B-SVM', 'LR', 'B-LR', 'SPAUC', 'SPAM', 'SVM-PE', 'LO-AUC']
    run_time_matrix = []
    rank_run_time_matrix = []
    num_trials = 210
    eff_range = range(5, num_trials - 5)
    for ind, dataset in enumerate(list_datasets):
        results = pkl.load(open(root_path + '%s/results_all_%s.pkl' % (dataset, dataset), 'rb'))
        list_vals = []
        for ind_method, method in enumerate(method_list):
            list_vals.append(np.mean([results[dataset][tag][_][method]['tr']['train_time'] for _ in eff_range]))
        run_time_matrix.append(list_vals)
        temp = (-np.asarray(list_vals)).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(list_vals))
        rank_run_time_matrix.append(ranks + 1)
    run_time_matrix = np.asarray(run_time_matrix)
    posit_ratio_list = [0.007661000718218817, 0.09360785252573617, 0.05530973451327434, 0.4449275362318841,
                        0.4483018867924528, 0.34992679355783307, 0.08278457196613359, 0.0775462962962963,
                        0.03761574074074074, 0.059661983302789656, 0.10416666666666667, 0.3561484918793503,
                        0.3, 0.358974358974359, 0.07695267410542517, 0.03945,
                        0.0367, 0.06666666666666667, 0.023249575248144506, 0.48202855736090594,
                        0.04375667022411953, 0.09857651245551602, 0.09857651245551602, 0.028785488958990538,
                        0.10232048236798831, 0.10398471615720524, 0.09597889374090247, 0.3489583333333333,
                        0.09728049728049729, 0.07353552139592855, 0.06578947368421052, 0.09263357571925387,
                        0.04895608351331893, 0.39404477287546186, 0.20599250936329588,
                        0.0847457627118644, 0.5190551181102362, 0.26246105919003115, 0.0612407211028632,
                        0.07522567703109329, 0.2576832151300236, 0.2565011820330969, 0.23522458628841608,
                        0.09090909090909091, 0.028205865439907992, 0.03736218864842793, 0.3119946091644205,
                        0.029649595687331536, 0.03436657681940701, 0.07364501448076127]
    speedup = dict()
    for ind, dataset in enumerate(list_datasets):
        speedup[posit_ratio_list[ind]] = min(run_time_matrix[ind][:-1]) / run_time_matrix[ind][-1]
    plt.plot([speedup[_] for _ in sorted(speedup)[::-1]])
    plt.show()
    for ind_method, method in enumerate(method_list):
        print(ind_method, '%15s %.4f %.4f' %
              (method, float(np.mean(run_time_matrix[:, ind_method])),
               float(np.std(run_time_matrix[:, ind_method]))))


def run_time_real():
    list_datasets = [
        'abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer', 'cardio_3',
        'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german', 'ionosphere',
        'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
        'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks', 'pen_digits_0', 'pen_digits_5', 'pima',
        'satimage_4', 'scene', 'seismic', 'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf',
        'spectrometer', 'splice', 'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
        'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt', 'yeast_me1', 'yeast_me2', 'yeast_ml8']
    method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'rf', 'b_rf', 'gb', 'rbf_svm', 'b_rbf_svm',
                   'rank_boost', 'adaboost', 'spam', 'spauc', 'svm_perf_lin']
    label_list = ['SVM', 'B-SVM', 'LR', 'B-LR', 'RF', 'B-RF', 'GB', 'RBF-SVM', 'B-RBF-SVM',
                  'RankBoost', 'AdaBoost', 'SPAM', 'SPAUC', 'SVM-PL']
    run_time_matrix = []
    rank_run_time_matrix = []
    num_trials = 210
    eff_range = range(5, num_trials - 5)
    for ind, dataset in enumerate(list_datasets):
        results = pkl.load(open(root_path + '%s/results_all_%s.pkl' % (dataset, dataset), 'rb'))
        list_vals = []
        for ind_method, method in enumerate(method_list):
            print(dataset, method)
            if dataset == 'w7a' and method == 'rbf_svm' or dataset == 'w7a' and method == 'b_rbf_svm':
                continue
            if len(results[dataset]['real'][0][method]) != 0:
                list_vals.append(np.mean([results[dataset]['real'][_][method]['tr']['train_time'] for _ in eff_range]))
            else:
                list_vals.append(0.0)
        run_time_matrix.append(list_vals)
        temp = (-np.asarray(list_vals)).argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(list_vals))
        rank_run_time_matrix.append(ranks + 1)
    run_time_matrix = np.asarray(run_time_matrix)
    posit_ratio_list = [0.007661000718218817, 0.09360785252573617, 0.05530973451327434, 0.4449275362318841,
                        0.4483018867924528, 0.34992679355783307, 0.08278457196613359, 0.0775462962962963,
                        0.03761574074074074, 0.059661983302789656, 0.10416666666666667, 0.3561484918793503,
                        0.3, 0.358974358974359, 0.07695267410542517, 0.03945,
                        0.0367, 0.06666666666666667, 0.023249575248144506, 0.48202855736090594,
                        0.04375667022411953, 0.09857651245551602, 0.09857651245551602, 0.028785488958990538,
                        0.10232048236798831, 0.10398471615720524, 0.09597889374090247, 0.3489583333333333,
                        0.09728049728049729, 0.07353552139592855, 0.06578947368421052, 0.09263357571925387,
                        0.04895608351331893, 0.39404477287546186, 0.20599250936329588,
                        0.0847457627118644, 0.5190551181102362, 0.26246105919003115, 0.0612407211028632,
                        0.07522567703109329, 0.2576832151300236, 0.2565011820330969, 0.23522458628841608,
                        0.09090909090909091, 0.028205865439907992, 0.03736218864842793, 0.3119946091644205,
                        0.029649595687331536, 0.03436657681940701, 0.07364501448076127]
    speedup = dict()
    for ind, dataset in enumerate(list_datasets):
        speedup[posit_ratio_list[ind]] = min(run_time_matrix[ind][:-1]) / run_time_matrix[ind][-1]
    plt.plot([speedup[_] for _ in sorted(speedup)[::-1]])
    plt.show()
    for ind_method, method in enumerate(method_list):
        print(ind_method, '%15s %.4f %.4f' %
              (method, float(np.mean(run_time_matrix[:, ind_method])),
               float(np.std(run_time_matrix[:, ind_method]))))


def t_test(list_datasets, method_list, num_trials, perplexity, dtype, significant_level, data_eval_type):
    tr_auc_matrix = []
    eff_range = range(0, num_trials)
    for ind, dataset in enumerate(list_datasets):
        print(dataset)
        results = pkl.load(open(root_path + '%s/all_results_%s_%s_%d.pkl' %
                                (dataset, dtype, dataset, perplexity), 'rb'))
        list_vals = []
        for ind_method, method in enumerate(method_list):
            list_vals.append([results[dataset][dtype][_][method][data_eval_type]['auc'] for _ in eff_range])
        tr_auc_matrix.append(list_vals)
    tr_auc_matrix = np.asarray(tr_auc_matrix)
    t_test_matrix = np.zeros((len(method_list), len(method_list)))
    for ind, dataset in enumerate(list_datasets):
        for ind1, method1 in enumerate(method_list):
            for ind2, method2 in enumerate(method_list):
                if ind1 == ind2:
                    continue
                stat, p_val = ttest_ind(a=tr_auc_matrix[ind][ind1], b=tr_auc_matrix[ind][ind2])
                m1 = np.mean(tr_auc_matrix[ind][ind1])
                m2 = np.mean(tr_auc_matrix[ind][ind2])
                if p_val <= significant_level and m1 > m2:
                    t_test_matrix[ind1][ind2] += 1
    return t_test_matrix


def find_the_worst_case():
    list_datasets = [
        'abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer', 'cardio_3',
        'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german', 'ionosphere',
        'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
        'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks', 'pen_digits_0', 'pen_digits_5', 'pima',
        'satimage_4', 'scene', 'seismic', 'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf',
        'spectrometer', 'splice', 'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
        'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt', 'yeast_me1', 'yeast_me2', 'yeast_ml8']
    method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'spauc', 'spam', 'svm_perf_lin', 'opt_auc']
    gaps = []
    zzz = []
    significant_gaps_approx = []
    significant_gaps_stand = []
    significant_gaps_both = []
    significant_gaps_all = []
    insignificant_gaps_approx = []
    arr_fill_style = ['none'] * 50
    arr_color_style = ['white'] * 50
    for ind, dataset in enumerate(list_datasets):
        results = pkl.load(open(root_path + '%s/results_all_%s.pkl' % (dataset, dataset), 'rb'))
        opt_auc, best_approx, best_gap, index_i = 0.0, 0.0, 0.0, 0
        t1, t2 = [], []
        for i in range(210):
            for ind_method, method in enumerate(method_list):
                if method == 'opt_auc':
                    opt_auc = results[dataset]['tsne'][i][method]['tr']['auc']
                    t1.append(opt_auc)
                else:
                    if best_approx < results[dataset]['tsne'][i][method]['tr']['auc']:
                        best_approx = results[dataset]['tsne'][i][method]['tr']['auc']
            gap = opt_auc - best_approx
            if best_gap < gap:
                best_gap = gap
                index_i = i
        gaps.append(best_gap)
        flag = True
        for method in ['spauc', 'spam', 'svm_perf_lin']:
            if method == 'opt_auc':
                continue
            t2 = [results[dataset]['tsne'][i][method]['tr']['auc'] for i in range(210)]
            stat, p_val = ttest_ind(a=t1, b=t2)
            if p_val > 0.05:
                flag = False
        if flag:
            significant_gaps_approx.append(best_gap)
            arr_fill_style[ind] = 'bottom'
            arr_color_style[ind] = 'tab:green'
        else:
            insignificant_gaps_approx.append(best_gap)
        flag2 = True
        for method in ['c_svm', 'b_c_svm', 'lr', 'b_lr']:
            if method == 'opt_auc':
                continue
            t2 = [results[dataset]['tsne'][i][method]['tr']['auc'] for i in range(210)]
            stat, p_val = ttest_ind(a=t1, b=t2)
            if p_val > 0.05:
                flag2 = False
        if flag2:
            significant_gaps_stand.append(best_gap)
            arr_fill_style[ind] = 'top'
            arr_color_style[ind] = 'tab:green'
        else:
            insignificant_gaps_approx.append(best_gap)
        if flag and flag2:
            significant_gaps_both.append(best_gap)
            arr_fill_style[ind] = 'full'
            arr_color_style[ind] = 'tab:green'
        significant_gaps_all.append([best_gap, arr_fill_style[ind], arr_color_style[ind]])
        print(ind, index_i, best_gap, flag)
        zzz.append([ind, index_i, best_gap, flag])
    significant_gaps_all.sort(key=lambda x: x[0], reverse=True)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for ind, (gap, fill_style, color) in enumerate(significant_gaps_all):
        if ind == 0:
            ax.plot(ind, gap, fillstyle=fill_style, marker='s', markersize=4,
                    markeredgewidth=0.5, color=color, markeredgecolor='tab:green',
                    label='Significant to Both')
        elif ind == 6:
            ax.plot(ind, gap, fillstyle=fill_style, marker='s', markersize=4,
                    markeredgewidth=0.5, color=color, markeredgecolor='tab:green',
                    label='Significant to Approximate AUC Optimizers')
        elif ind == 15:
            ax.plot(ind, gap, fillstyle=fill_style, marker='s', markersize=4,
                    markeredgewidth=0.5, color=color, markeredgecolor='tab:green',
                    label='Significant to Standard Classifiers')
        elif ind == 19:
            ax.plot(ind, gap, fillstyle=fill_style, marker='s', markersize=4,
                    markeredgewidth=0.5, color=color, markeredgecolor='tab:green',
                    label='No Significance')
        else:
            ax.plot(ind, gap, fillstyle=fill_style, marker='s', markersize=4,
                    markeredgewidth=0.5, color=color, markeredgecolor='tab:green')
    ax.set_ylabel('Gap')
    ax.set_xlabel('Dataset')
    leg = ax.legend(frameon=False)
    for line in leg.get_lines():
        line.set_linewidth(0.0)
    file_name = 'example_plot_t_test.pdf'
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s' % file_name
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()


def t_test_real(tag):
    list_datasets = [
        'abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer', 'cardio_3',
        'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german', 'ionosphere',
        'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
        'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks', 'pen_digits_0', 'pen_digits_5', 'pima',
        'satimage_4', 'scene', 'seismic', 'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf',
        'spectrometer', 'splice', 'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
        'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt', 'yeast_me1', 'yeast_me2', 'yeast_ml8']
    method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'rf', 'b_rf', 'gb', 'rbf_svm', 'b_rbf_svm',
                   'rank_boost', 'adaboost', 'spam', 'spauc', 'svm_perf_lin']
    label_list = ['SVM', 'B-SVM', 'LR', 'B-LR', 'RF', 'B-RF', 'GB', 'RBF-SVM', 'B-RBF-SVM',
                  'RankBoost', 'AdaBoost', 'SPAM', 'SPAUC', 'SVM-PL']
    tr_auc_matrix = []
    num_trials = 210
    eff_range = range(5, num_trials - 5)
    for ind, dataset in enumerate(list_datasets):
        results = pkl.load(open(root_path + '%s/results_all_%s.pkl' % (dataset, dataset), 'rb'))
        list_vals = []
        for ind_method, method in enumerate(method_list):
            if len(results[dataset]['real'][5][method]) != 0:
                list_vals.append([results[dataset]['real'][_][method][tag]['auc'] for _ in eff_range])
            else:
                list_vals.append([0.0 for _ in eff_range])
        tr_auc_matrix.append(list_vals)
    tr_auc_matrix = np.asarray(tr_auc_matrix)
    t_test_matrix = np.zeros((15, 15))
    for ind, dataset in enumerate(list_datasets):
        for ind1, method1 in enumerate(method_list):
            for ind2, method2 in enumerate(method_list):
                if ind1 == ind2:
                    continue
                stat, p_val = ttest_ind(a=tr_auc_matrix[ind][ind1], b=tr_auc_matrix[ind][ind2])
                m1 = np.mean(tr_auc_matrix[ind][ind1])
                m2 = np.mean(tr_auc_matrix[ind][ind2])
                if p_val <= 0.05 and m1 > m2:
                    t_test_matrix[ind1][ind2] += 1
    for ind1, method1 in enumerate(method_list):
        print('%15s ' % label_list[ind1], end=' ')
        all_x = []
        for ind2, method2 in enumerate(method_list):
            if ind1 < 9 and ind2 >= 9:
                print(' & \cellcolor{blue!30!white}%02d' % t_test_matrix[ind1][ind2], end=' ')
                all_x.append(t_test_matrix[ind1][ind2])
            elif ind1 >= 9 and ind2 < 9:
                print(' & \cellcolor{red!30!white}%02d' % t_test_matrix[ind1][ind2], end=' ')
                all_x.append(t_test_matrix[ind1][ind2])
            elif ind1 == ind2:
                print(' & -', end=' ')
            else:
                print(' & %02d' % t_test_matrix[ind1][ind2], end=' ')
                all_x.append(t_test_matrix[ind1][ind2])
        print(' & %.2f \\\\' % np.mean(all_x))
    print('-' * 80)
    return t_test_matrix


def generate_results_2d():
    list_datasets = [
        'abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer', 'cardio_3',
        'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german', 'ionosphere',
        'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
        'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks', 'pen_digits_0', 'pen_digits_5', 'pima',
        'satimage_4', 'scene', 'seismic', 'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf',
        'spectrometer', 'splice', 'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
        'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt', 'yeast_me1', 'yeast_me2', 'yeast_ml8']
    dtype, num_trials, perplexity, significant_level = 'tsne-2d', 200, 40, 0.05
    for dataset in list_datasets:
        get_summarized_data(dataset_name=dataset, dtype=dtype, num_trials=num_trials, perplexity=perplexity)
    method_title = ['SVM', 'B-SVM', 'LR', 'B-LR', 'SVM-Perf', 'SPAUC', 'SPAM', 'AUC-opt']
    method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'svm_perf_lin', 'spauc', 'spam', 'opt_auc_2d']
    t_test_mat1 = t_test(list_datasets=list_datasets, method_list=method_list, num_trials=num_trials,
                         perplexity=perplexity, dtype=dtype, significant_level=significant_level, data_eval_type="tr")
    t_test_mat2 = t_test(list_datasets=list_datasets, method_list=method_list, num_trials=num_trials,
                         perplexity=perplexity, dtype=dtype, significant_level=significant_level, data_eval_type="te1")
    for ind, (item1, item2) in enumerate(zip(t_test_mat1, t_test_mat2)):
        list_ = list(item1)
        list_.extend(item2)
        print(" & ", end=' ')
        print(method_title[ind], end=' ')
        print('& ', end=' ')
        print(' & '.join([str(int(_)) if ind2 != ind else '-' for ind2, _ in enumerate(list_)]), end=" ")
        print('\\\\')


def generate_results_3d():
    list_datasets = [
        'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
        'optical_digits_0', 'optical_digits_8', 'satimage_4', 'scene', 'seismic', 'sick_euthyroid',
        'solar_flare_m0', 'spambase', 'spectf', 'spectrometer', 'splice']
    dtype, num_trials, perplexity, significant_level = 'tsne-3d', 50, 50, 0.05
    for dataset in list_datasets:
        get_summarized_data(dataset_name=dataset, dtype=dtype, num_trials=num_trials, perplexity=perplexity)
    method_title = ['SVM', 'B-SVM', 'LR', 'B-LR', 'SVM-Perf', 'SPAUC', 'SPAM', 'AUC-opt-Reg', 'AUC-opt-3d-Reg']
    method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'svm_perf_lin', 'spauc', 'spam', 'opt_auc_3d', 'opt_auc_3d_reg_50']
    t_test_mat1 = t_test(list_datasets=list_datasets, method_list=method_list, num_trials=num_trials,
                         perplexity=perplexity, dtype=dtype, significant_level=significant_level, data_eval_type="tr")
    t_test_mat2 = t_test(list_datasets=list_datasets, method_list=method_list, num_trials=num_trials,
                         perplexity=perplexity, dtype=dtype, significant_level=significant_level, data_eval_type="te1")
    for ind, (item1, item2) in enumerate(zip(t_test_mat1, t_test_mat2)):
        list_ = list(item1)
        list_.extend(item2)
        print(" & ", end=' ')
        print(method_title[ind], end=' ')
        print('& ', end=' ')
        print(' & '.join([str(int(_)) if ind2 != ind else '-' for ind2, _ in enumerate(list_)]), end=" ")
        print('\\\\')


def main():
    generate_results_3d()
    exit()
    if sys.argv[1] == 'sum_data':
        get_summarized_data(dataset_name=sys.argv[2])
        show_tr_auc_te_auc(dataset=sys.argv[2])
        show_tr_acc_f1_te_acc_f1(dataset=sys.argv[2])
    elif sys.argv[1] == 'tsne_train_auc':
        tsne_results(tag='f1')
    elif sys.argv[1] == 'std':
        std_auc(tag='b_acc')
        std_auc(tag='f1')
    elif sys.argv[1] == 'run_time_tsne':
        run_time_tsne(tag='tr')
    elif sys.argv[1] == 'run_time_real':
        run_time_real()
    elif sys.argv[1] == 't_test_real':
        t_test_mat1 = t_test_real('tr')
        t_test_mat2 = t_test_real('te1')
        for item1, item2 in zip(t_test_mat1, t_test_mat2):
            list_ = list(item1)
            list_.extend(item2)
            print(' & '.join([str(int(_)) for _ in list_]))
    elif sys.argv[1] == 'worst_case':
        find_the_worst_case()


if __name__ == '__main__':
    main()
