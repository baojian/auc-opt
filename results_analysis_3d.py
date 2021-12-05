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

root_path = "/data/auc-opt-datasets/datasets/"


def get_summarized_data(dataset_name, dtype, num_trials, perplexity):
    method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'opt_auc_3d', 'opt_auc_3d_reg_50', 'spauc', 'spam', 'svm_perf_lin']
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
            file = root_path + '%s/results_%s_%s_%s.pkl' % (dataset, dtype, dataset, method)
            if os.path.exists(file):
                re = pkl.load(open(file, 'rb'))
                print(dataset, dtype, method)
                for trial_i in re:
                    for label in ['tr', 'te1', 'te2', 'te3']:
                        if method == 'opt_auc_3d_reg_50':
                            results[dataset][dtype][trial_i][method][label] = re[trial_i]['opt_auc_3d_reg'][label]
                        else:
                            results[dataset][dtype][trial_i][method][label] = re[trial_i][method][label]
            else:
                print(file)
            print('------vs------')
        print('-------------')
        file = root_path + '%s/all_results_%s_%s_%d.pkl' % (dataset, dtype, dataset, perplexity)
        pkl.dump(results, open(file, 'wb'))


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
                print(method1, method2, m1, m2)
                if p_val <= significant_level and m1 > m2:
                    t_test_matrix[ind1][ind2] += 1
    return t_test_matrix


def generate_results_3d():
    list_datasets = [
        'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
        'optical_digits_0', 'optical_digits_8', 'satimage_4', 'scene', 'seismic', 'sick_euthyroid',
        'solar_flare_m0', 'spambase', 'spectf', 'spectrometer', 'splice']
    dtype, num_trials, perplexity, significant_level = 'tsne-3d', 50, 50, 0.05
    for dataset in list_datasets:
        get_summarized_data(dataset_name=dataset, dtype=dtype, num_trials=num_trials, perplexity=perplexity)
    method_title = ['SVM', 'B-SVM', 'LR', 'B-LR', 'SVM-Perf', 'SPAUC', 'SPAM', 'AUC-opt', 'AUC-opt-3d-Reg']
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


if __name__ == '__main__':
    main()
