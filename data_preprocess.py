# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
import pickle as pkl
import multiprocessing
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

"""
ecoli:          https://archive.ics.uci.edu/ml/datasets/Ecoli
splice:         https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)
breast_cancer:  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#breast-cancer
australian:     http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
ijcnn1:         https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
german:         https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
pima:           https://archive.ics.uci.edu/ml/datasets/diabetes
page_blocks:    https://archive.ics.uci.edu/ml/machine-learning-databases/page-blocks/
pen_digits_0:   https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
optical_digits_0:   https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
letter_a:           https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
yeast_cyt:          https://archive.ics.uci.edu/ml/datasets/Yeast
spectf:             https://archive.ics.uci.edu/ml/datasets/Yeast
ionosphere:         https://archive.ics.uci.edu/ml/datasets/ionosphere
"""

root_path = "/data/auc-opt-datasets/datasets/"


def get_real_data(dataset):
    x_tr, y_tr = [], []
    with open(root_path + '%s/input_%s.txt' % (dataset, dataset)) as f:
        for ind, each_line in enumerate(f.readlines()):
            each_line = each_line.rstrip()
            if each_line.startswith('#'):
                continue
            if ind == 1:
                n, p = int(each_line.split(' ')[0]), int(each_line.split(' ')[1])
                continue
            items = each_line.split(' ')
            indices = [int(_.split(':')[0]) for _ in items[:-1]]
            vals = np.asarray([float(_.split(':')[1]) for _ in items[:-1]])
            y_tr.append(int(float(items[-1])))
            sample = np.zeros(p)
            sample[indices] = vals
            x_tr.append(sample)
    return np.asarray(x_tr, dtype=np.float64), np.asarray(y_tr, dtype=np.float64)


def get_tsne_data(dataset, dtype, perplexity):
    assert dtype == '3d' or dtype == '2d'
    if dtype == '3d':
        data = pkl.load(open(root_path + '%s/t_sne_3d_%s.pkl' % (dataset, dataset), 'rb'))
    elif dtype == "2d":
        data = pkl.load(open(root_path + '%s/t_sne_2d_%s.pkl' % (dataset, dataset), 'rb'))
    else:
        print("error unknown data type")
        return 0
    # fourclass is 2d datasets, no need to do projection.
    key = ('standard', perplexity) if dataset != 'fourclass' else ('original', perplexity)
    x_tr = data[key]['embeddings']
    y_tr = data[key]['y_tr']
    return np.asarray(x_tr), np.asarray(y_tr)


def _get_data(x_tr, y_tr, data_name, num_trials, split_ratio, verbose):
    """
    # 1. data standardization.
    https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    https://datascience.stackexchange.com/questions/27615/should-we-apply-normalization-to-test-data-as-well
    https://jamesmccaffrey.wordpress.com/2019/01/04/how-to-normalize-training-and-test-data-for-machine-learning/
    https://sebastianraschka.com/faq/docs/scale-training-test.html
    sklearn.preprocessing.StandardScaler
    """
    np.random.seed(17)
    x_tr = np.asarray(x_tr, dtype=float)
    y_tr = np.asarray(y_tr, dtype=float)
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%s/' % data_name),
            'p': len(x_tr[0]),
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': np.count_nonzero(y_tr > 0),
            'num_nega': np.count_nonzero(y_tr < 0),
            'x_tr': x_tr,
            'y_tr': y_tr}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    if verbose > 0:
        print('# of classes: %d' % len(np.unique(y_tr)), end=' ')
        print('# of samples: %d' % data['n'])
        print('# of positive: %d' % data['num_posi'], end=' ')
        print('# of negative: %d' % data['num_nega'])
        print('# of features: %d' % data['p'], end=' ')
        print('posi_ratio: %.5f' % data['posi_ratio'])
    # generate different trials of data
    for _ in range(num_trials):
        all_folds = dict()
        # to make sure that there are least 2 positive samples in sub_y_tr and sub_y_te
        k_fold, min_posi = 5, 2
        while True:
            all_indices = np.random.permutation(data['n'])
            data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
            assert data['n'] == len(data['trial_%d_all_indices' % _])
            tr_indices = all_indices[:int(len(all_indices) * split_ratio)]
            data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
            te_indices = all_indices[int(len(all_indices) * split_ratio):]
            data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
            n_tr = len(data['trial_%d_tr_indices' % _])
            n_te = len(data['trial_%d_te_indices' % _])
            assert data['n'] == (n_tr + n_te)
            all_folds[_] = []
            flag = True
            kf = KFold(n_splits=k_fold, shuffle=False)  # Folding is fixed.
            for ind, (sub_tr_ind, sub_te_ind) in \
                    enumerate(kf.split(np.zeros(shape=(len(tr_indices), 1)))):
                sub_y_tr = data['y_tr'][tr_indices[sub_tr_ind]]
                sub_y_te = data['y_tr'][tr_indices[sub_te_ind]]
                if len(np.unique(sub_y_tr)) == 1 or len(np.unique(sub_y_te)) == 1 or \
                        len([_ for _ in sub_y_te if _ > 0]) < min_posi:
                    flag = False
                    break
                all_folds[_].append([sub_y_tr, sub_y_te])
            if flag:
                break
        if verbose > 0:
            print('----')
            for sub_y_tr, sub_y_te in all_folds[_]:
                print('trial_%d' % _, 'sub_posi_tr: %d' % np.count_nonzero(sub_y_tr > 0),
                      'sub_posi_te: %d' % np.count_nonzero(sub_y_te > 0),
                      'posi_tr: %d' % np.count_nonzero(y_tr[tr_indices] > 0),
                      'posi_te: %d' % np.count_nonzero(y_tr[te_indices] > 0))

    return data


def t_sne_2d(para):
    data_name, model_name, data_x, data_y, perplexity, rand_state = para
    start_time = time.time()
    t_sne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12.0, learning_rate=200.0,
                 n_iter=5000, n_iter_without_progress=300, min_grad_norm=1e-7, metric="euclidean",
                 init="random", verbose=0, random_state=rand_state, method='barnes_hut', angle=0.5,
                 n_jobs=None)
    t_sne.fit_transform(X=data_x, y=data_y)
    print(model_name, perplexity, 'run_time', time.time() - start_time)
    return {(model_name, perplexity): {
        'data_name': data_name, 'x_tr': data_x, 'y_tr': data_y,
        'rand_state': rand_state, 'embeddings': t_sne.embedding_,
        't_sne_paras': {
            'n_components': 2, 'perplexity': perplexity, 'early_exaggeration': 12.0,
            'learning_rate': 200.0, 'n_iter': 5000, 'n_iter_without_progress': 300,
            'min_grad_norm': 1e-7, 'metric': "euclidean", 'init': "random", 'verbose': 0,
            'random_state': rand_state, 'method': 'barnes_hut', 'angle': 0.5, 'n_jobs': 'None'}}}


def t_sne_3d(para):
    data_name, model_name, data_x, data_y, perplexity, rand_state = para
    start_time = time.time()
    t_sne = TSNE(n_components=3, perplexity=perplexity, early_exaggeration=12.0, learning_rate=200.0,
                 n_iter=5000, n_iter_without_progress=300, min_grad_norm=1e-7, metric="euclidean",
                 init="random", verbose=0, random_state=rand_state, method='barnes_hut', angle=0.5,
                 n_jobs=None)
    t_sne.fit_transform(X=data_x, y=data_y)
    print(model_name, perplexity, 'run_time', time.time() - start_time)
    return {(model_name, perplexity): {
        'data_name': data_name, 'x_tr': data_x, 'y_tr': data_y,
        'rand_state': rand_state, 'embeddings': t_sne.embedding_,
        't_sne_paras': {
            'n_components': 3, 'perplexity': perplexity, 'early_exaggeration': 12.0,
            'learning_rate': 200.0, 'n_iter': 5000, 'n_iter_without_progress': 300,
            'min_grad_norm': 1e-7, 'metric': "euclidean", 'init': "random", 'verbose': 0,
            'random_state': rand_state, 'method': 'barnes_hut', 'angle': 0.5, 'n_jobs': 'None'}}}


def run_single_tsne_2d(dataset):
    x_tr, y_tr = get_real_data(dataset=dataset)
    rand_state = 17
    para_space = []
    for model_name, model in zip(
            ['original', 'norm', 'min_max', 'standard'],
            [None, Normalizer(norm='l2'), MinMaxScaler(feature_range=(-1., 1.)), StandardScaler()]):
        if model is not None:
            x_tr = model.fit_transform(X=x_tr)
        for perplexity in [10, 20, 30, 40, 50]:
            para_space.append((dataset, model_name, x_tr, y_tr, perplexity, rand_state))
    pool = multiprocessing.Pool(processes=len(para_space))
    results_pool = pool.map(t_sne_2d, para_space)
    pool.close()
    pool.join()
    results = dict()
    for item in results_pool:
        for key in item:
            results[key] = item[key]
    pkl.dump(results, open(root_path + '%s/t_sne_2d_%s.pkl' % (dataset, dataset), 'wb'))


def run_single_tsne_3d(dataset):
    x_tr, y_tr = get_real_data(dataset=dataset)
    rand_state = 17
    para_space = []
    for model_name, model in zip(['original', 'norm', 'min_max', 'standard'],
                                 [None, Normalizer(norm='l2'), MinMaxScaler(feature_range=(-1., 1.)),
                                  StandardScaler()]):
        if model is not None:
            x_tr = model.fit_transform(X=x_tr)
        for perplexity in [10, 20, 30, 40, 50]:
            para_space.append((dataset, model_name, x_tr, y_tr, perplexity, rand_state))
    pool = multiprocessing.Pool(processes=len(para_space))
    results_pool = pool.map(t_sne_3d, para_space)
    pool.close()
    pool.join()
    results = dict()
    for item in results_pool:
        for key in item:
            results[key] = item[key]
    pkl.dump(results, open(root_path + '%s/t_sne_3d_%s.pkl' % (dataset, dataset), 'wb'))


def draw_t_sne(data_name):
    import matplotlib.pyplot as plt
    data = pkl.load(open(root_path + '%s/t_sne_2d_%s.pkl'
                         % (data_name, data_name), 'rb'))
    fig, ax = plt.subplots(4, 5, figsize=(15, 12))
    for ind_model, model_name in enumerate(['original', 'norm', 'min_max', 'standard']):
        for ind, perplexity in enumerate([10, 20, 30, 40, 50]):
            x_tr, y_tr = data[(model_name, perplexity)]['x_tr'], data[(model_name, perplexity)]['y_tr']
            embeddings = data[(model_name, perplexity)]['embeddings']
            ax[ind_model, ind].scatter(embeddings[np.argwhere(y_tr > 0), 0],
                                       embeddings[np.argwhere(y_tr > 0), 1],
                                       c='r', marker='o', facecolor='None', s=5.)
            ax[ind_model, ind].scatter(embeddings[np.argwhere(y_tr < 0), 0],
                                       embeddings[np.argwhere(y_tr < 0), 1],
                                       c='b', marker='+', facecolor='None', s=5.)
            ax[ind_model, ind].set_title('%s (per-%02d)' % (model_name, perplexity))
    f_name = root_path + '%s/t_sne_2d_%s.pdf' % (data_name, data_name)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.1, format='pdf')
    plt.close()


def get_data(dtype, dataset, num_trials, split_ratio, perplexity, verbose=0):
    if dataset == 'simu' or dataset == 'splice' or dataset == 'a9a' or dataset == 'breast_cancer' or \
            dataset == 'banana' or dataset == 'ecoli_imu' or dataset == 'mushrooms' or dataset == 'australian' or \
            dataset == 'spambase' or dataset == 'ionosphere' or dataset == 'fourclass' or \
            dataset == 'german' or dataset == 'svmguide3' or dataset == 'spectf' or \
            dataset == 'pen_digits_0' or dataset == 'page_blocks' or dataset == 'optical_digits_0' or \
            dataset == 'w8a' or dataset == 'yeast_me1' or dataset == 'letter_a' or dataset == 'ijcnn1' or \
            dataset == 'vowel_hid' or dataset == 'vehicle_bus' or dataset == 'vehicle_saab' or \
            dataset == 'vehicle_van' or dataset == 'abalone_19' or dataset == 'seismic' or \
            dataset == 'protein_homo' or dataset == 'mammography' or dataset == 'cardio_3' or \
            dataset == 'ozone_level' or dataset == 'w7a' or dataset == 'yeast_me2' or \
            dataset == 'letter_z' or dataset == 'wine_quality' or dataset == 'car_eval_4' or \
            dataset == 'oil' or dataset == 'solar_flare_m0' or dataset == 'arrhythmia_06' or \
            dataset == 'coil_2000' or dataset == 'thyroid_sick' or dataset == 'libras_move' or \
            dataset == 'scene' or dataset == 'yeast_ml8' or dataset == 'yeast_cyt' or dataset == 'us_crime' or \
            dataset == 'isolet' or dataset == 'car_eval_34' or dataset == 'spectrometer' or \
            dataset == 'sick_euthyroid' or dataset == 'abalone_7' or dataset == 'pen_digits_5' or \
            dataset == 'satimage_4' or dataset == 'optical_digits_8' or dataset == 'pima':
        if dtype == "real":
            x_tr, y_tr = get_real_data(dataset=dataset)
        elif dtype == "tsne-2d":
            x_tr, y_tr = get_tsne_data(dataset=dataset, dtype="2d", perplexity=perplexity)
        elif dtype == "tsne-3d":
            x_tr, y_tr = get_tsne_data(dataset=dataset, dtype="3d", perplexity=perplexity)
        else:
            print(f"unknown type of dataset.")
            return 0
        return _get_data(x_tr, y_tr, dataset, num_trials, split_ratio, verbose=verbose)
    else:
        print('error of data %s' % dataset)


def fetch_imbalance_dataset(dataset, num_trials, split_ratio, verbose):
    from imblearn.datasets import fetch_datasets
    protein_homo = fetch_datasets()[dataset]
    x_tr, y_tr = protein_homo['data'], protein_homo['target']
    print(np.count_nonzero(y_tr > 0), x_tr.shape)
    return _get_data(x_tr=x_tr, y_tr=y_tr, data_name=dataset,
                     num_trials=num_trials, split_ratio=split_ratio, verbose=verbose)


def fetched_data():
    from imblearn.datasets import fetch_datasets
    data_set = 'ecoli'
    data = fetch_datasets()[data_set]
    x_tr, y_tr = data['data'], data['target']
    if not os.path.exists('/data/auc-logistic/%s/' % data_set):
        os.mkdir('/data/auc-logistic/%s/' % data_set)
    f_w = open('/data/auc-logistic/%s/raw_%s' % (data_set, data_set), 'w')
    f_w.write(str(len(x_tr)) + ' ' + str(len(x_tr[0])) + '\n')
    for sample, label in zip(x_tr, y_tr):
        for ind, val in zip(range(len(sample)), sample):
            f_w.write(str(ind) + ':' + str(val) + ' ')
        f_w.write(str(label) + '\n')
    f_w.close()


def generate_seismic():
    from sklearn.preprocessing import OneHotEncoder
    from itertools import product
    x_tr, y_tr = [], []
    data_name = 'seismic'
    f_path = '%s/raw_%s_old' % (data_name, data_name)
    cat_x_tr, num_x_tr = [], []
    with open(os.path.join(root_path, f_path)) as f:
        for ind, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(',')
            x_tr.append([_ for _ in items[:-1]])
            cat_x_tr.append([items[0], items[1], items[2], items[3]])
            num_x_tr.append([_ for ind, _ in enumerate(x_tr[-1]) if ind not in [0, 1, 2, 7]])
            y_tr.append(1 if items[-1] == '1' else -1)
    enc = OneHotEncoder(handle_unknown='ignore')
    x = []
    for (a, b, c, d) in product(np.unique([_[0] for _ in x_tr]),
                                np.unique([_[1] for _ in x_tr]),
                                np.unique([_[2] for _ in x_tr]),
                                np.unique([_[7] for _ in x_tr])):
        x.append([a, b, c, d])
    enc.fit(x)
    cat_x_tr = enc.transform(cat_x_tr).toarray()
    x_tr = []
    for item1, item2 in zip(cat_x_tr, num_x_tr):
        line = [_ for _ in item1]
        line.extend([float(_) for _ in item2])
        x_tr.append(line)
    x_tr, y_tr = np.asarray(x_tr), np.asarray(y_tr)
    f_w = open('/data/auc-logistic/%s/raw_%s' % (data_name, data_name), 'w')
    f_w.write(str(len(x_tr)) + ' ' + str(len(x_tr[0])) + '\n')
    for sample, label in zip(x_tr, y_tr):
        for ind, val in zip(range(len(sample)), sample):
            f_w.write(str(ind) + ':' + str(val) + ' ')
        f_w.write(str(label) + '\n')
    f_w.close()
    print(np.count_nonzero(y_tr > 0), x_tr.shape)


def re_name():
    import sys
    dataset = sys.argv[1]
    for method in ['rank_boost', 'adaboost', 'c_svm', 'b_c_svm', 'lr', 'b_lr', 'rf', 'b_rf', 'gb',
                   'spam', 'spauc', 'svm_perf_lin', 'rbf_svm', 'b_rbf_svm', 'svm_perf_rbf']:
        os.rename(root_path + '%s/results_pen_digits_%s.pkl' % (dataset, method),
                  root_path + '%s/results_real_pen_digits_0_%s.pkl' % (dataset, method))


def print_datasets_table():
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['pdf.fonttype'] = 3
    plt.rcParams['ps.fonttype'] = 3
    plt.rcParams["font.size"] = 13
    plt.rcParams['axes.linewidth'] = 0.4
    plt.rcParams['axes.xmargin'] = 0
    datasets = ['abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer',
                'cardio_3', 'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german',
                'ionosphere', 'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography',
                'mushrooms', 'oil', 'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks',
                'pen_digits_0', 'pen_digits_5', 'pima', 'satimage_4', 'scene', 'seismic',
                'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf', 'spectrometer', 'splice',
                'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
                'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt',
                'yeast_me1', 'yeast_me2', 'yeast_ml8']
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
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    plt.plot(range(1, len(posit_ratio_list) + 1), sorted(posit_ratio_list)[::-1],
             color='tab:red', marker='o', markersize=3, markerfacecolor='white')
    ax.set_ylabel("Positive Ratio $(\displaystyle \gamma )$", labelpad=.8)
    ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40, 45])
    ax.set_xlim([0, 51])
    ax.grid(color='gray', linewidth=0.4, linestyle='--', dashes=(5, 5))
    ax.set_xlabel('Dataset', labelpad=.8)
    ax.tick_params(axis='y', direction='in', labelcolor='black', color='black', pad=1.)
    ax.tick_params(axis='x', direction='in', labelcolor='black', color='black', pad=1.)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/posi-ratio.pdf'
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.01, format='pdf')
    plt.close()


def main():
    if sys.argv[1] == 'get_data':
        get_data(dtype='real', dataset=sys.argv[2], num_trials=1, split_ratio=0.5, verbose=1)
    elif sys.argv[1] == 'print_table':
        datasets = ['abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer',
                    'cardio_3', 'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german',
                    'ionosphere', 'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography',
                    'mushrooms', 'oil', 'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks',
                    'pen_digits_0', 'pen_digits_5', 'pima', 'satimage_4', 'scene', 'seismic',
                    'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf', 'spectrometer', 'splice',
                    'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
                    'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt',
                    'yeast_me1', 'yeast_me2', 'yeast_ml8']
        all_datasets = []
        for dataset in datasets:
            data = get_data(dtype='real', dataset=dataset, num_trials=1, split_ratio=.5)
            all_datasets.append(
                [data['posi_ratio'], '%s & %d & %d & %.4f' % (dataset, data['n'], data['p'], data['posi_ratio'])])
        all_datasets.sort(reverse=True, key=lambda x: x[0])
        for item1, item2 in zip(all_datasets[:25], all_datasets[25:]):
            print(' & '.join([item1[1], item2[1]]))
    elif sys.argv[1] == 'print_data':
        print_datasets_table()
    elif sys.argv[1] == 'tsne-2d':
        run_single_tsne_2d(dataset=sys.argv[2])
        draw_t_sne(data_name=sys.argv[2])
    elif sys.argv[1] == 'tsne-3d':
        list_datasets = ['abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer',
                         'cardio_3', 'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german',
                         'ionosphere', 'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography',
                         'mushrooms', 'oil', 'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks',
                         'pen_digits_0', 'pen_digits_5', 'pima', 'satimage_4', 'scene', 'seismic',
                         'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf', 'spectrometer', 'splice',
                         'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
                         'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt',
                         'yeast_me1', 'yeast_me2', 'yeast_ml8']
        for dataset in ['w7a']:
            print(dataset)
            run_single_tsne_3d(dataset=dataset)
    elif sys.argv[1] == 'pre_proc':
        dataset = sys.argv[2]
        f_w = open(root_path + '%s/input_%s.txt' % (dataset, dataset), 'w')
        f_w.write('# n p n_p n_q\n')
        f_w.write('%d %d\n' % (1284, 22))
        num_posi, num_nega = 0, 0
        with open(root_path + '%s/raw_svmguide3_data.txt' % dataset, 'r') as f:
            for ind, each_line in enumerate(f.readlines()):
                items = [_ for _ in each_line.rstrip().split(' ') if _ != '']
                print(items)
                for item in items[1:]:
                    f_w.write('%d:%s ' % (int(item.split(':')[0]) - 1, item.split(':')[1]))
                if items[0] == '-1':
                    f_w.write('-1\n')
                    num_nega += 1
                else:
                    f_w.write('1\n')
                    num_posi += 1
        print(num_posi, num_nega)
        f_w.close()


def draw_t_sne_10():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    list_datasets = ['breast_cancer', 'australian', 'cardio_3', 'mushrooms', 'libras_move',
                     'letter_a', 'ecoli_imu', 'optical_digits_0', 'page_blocks', 'fourclass']
    list_datasets_labels = ['breast-cancer', 'australian', 'cardio-3', 'mushrooms', 'libras-move',
                            'letter[a]', 'ecoli-imu', 'optical-digits[0]', 'page-blocks', 'fourclass']
    for index, dataset in enumerate(list_datasets[0:10]):
        ind_model, ind = index // 5, index % 5
        root = '/home/baojian/Dropbox/pub/2020/NIPS-2020/supp-and-code/datasets/'
        data = pkl.load(open(root + '%s/t_sne_2d_%s.pkl' % (dataset, dataset), 'rb'))
        x_tr, y_tr = data[('standard', 50)]['x_tr'], data[('standard', 50)]['y_tr']
        embeddings = data[('standard', 50)]['embeddings']
        from sklearn.preprocessing import StandardScaler
        embeddings = StandardScaler().fit_transform(X=embeddings)
        ax[ind_model, ind].scatter(embeddings[np.argwhere(y_tr > 0), 0],
                                   embeddings[np.argwhere(y_tr > 0), 1],
                                   c='r', marker='o', facecolor='None', s=5.)
        ax[ind_model, ind].scatter(embeddings[np.argwhere(y_tr < 0), 0],
                                   embeddings[np.argwhere(y_tr < 0), 1],
                                   c='b', marker='+', facecolor='None', s=5.)
        ax[ind_model, ind].set_title('%s' % list_datasets_labels[index], fontsize=18)
        ax[ind_model, ind].tick_params(axis='y', direction='in')
        ax[ind_model, ind].tick_params(axis='x', direction='in')
        ax[ind_model, ind].set_xticks([])
        ax[ind_model, ind].set_yticks([])
    f_name = root_path + 't_sne_2d_0_10.pdf'
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.1, format='pdf')
    plt.close()


if __name__ == '__main__':
    main()
