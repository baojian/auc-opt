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
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import softmax
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import multiprocessing
import pickle as pkl
from scipy.stats import ttest_ind
from sklearn.metrics import adjusted_rand_score

if os.uname()[1] == 'baojian-ThinkPad-T540p':
    root_path = '/data/auc-logistic/'
elif os.uname()[1] == 'pascal':
    root_path = '/mnt/store2/baojian/data/auc-logistic/'
elif os.uname()[1].endswith('.rit.albany.edu'):
    root_path = '/network/rit/lab/ceashpc/bz383376/data/auc-logistic/'


def linear_auc_2d_fast(x_tr, y_tr, eps=1e-10, rand_state=17):
    np.random.seed(rand_state)
    n, d = x_tr.shape
    if d != 2:
        sys.exit('data dimension should be 2!')
    all_slopes = dict()
    for ii, jj in product(range(n), range(n)):
        if ii < jj:
            diff = x_tr[ii] - x_tr[jj]
            if diff[0] != 0.0:
                slope = [-diff[1] / diff[0] - eps, 1.0]
            else:
                # TODO check this
                print('warning: some vertical lines')
                slope = [-diff[1] / 1e-5 - eps, 1.0]
            if np.abs(slope[0]) > 1e1:
                continue
            if np.dot(diff, slope) < 0.0:
                all_slopes[(ii, jj)] = slope
            else:
                all_slopes[(jj, ii)] = slope
    sorted_list = sorted(all_slopes.items(), key=operator.itemgetter(1))
    pair, slope = sorted_list[0]
    auc_list = [roc_auc_score(y_true=y_tr, y_score=np.dot(x_tr, slope))]
    t_posi = len([_ for _ in y_tr if _ > 0])
    granularity = 1. / (t_posi * (len(y_tr) - t_posi))
    best_auc, best_slope = auc_list[-1], slope
    for (ii, jj), slope in sorted_list[:-1]:
        if y_tr[ii] == y_tr[jj]:
            auc_list.append(auc_list[-1])
        elif y_tr[ii] > y_tr[jj]:
            auc_list.append(auc_list[-1] + granularity)
        elif y_tr[ii] < y_tr[jj]:
            auc_list.append(auc_list[-1] - granularity)
        if best_auc < auc_list[-1]:
            best_auc = auc_list[-1]
            best_slope = np.asarray(slope)
            print('positive', auc_list[-1])
        if best_auc < (1. - auc_list[-1]):
            best_auc = 1. - auc_list[-1]
            best_slope = -np.asarray(slope)
            print('negative', 1. - auc_list[-1])
    return np.asarray(auc_list), best_auc, best_slope


def get_granularity(y_tr):
    num_posi = len([_ for _ in y_tr if _ > 0])
    num_pairs = num_posi * (len(y_tr) - num_posi)
    return 1. / num_pairs


def fit_auc_acc_f1_opt_auc(w, x_tr, y_tr):
    scores = np.dot(x_tr, w)
    auc = roc_auc_score(y_true=y_tr, y_score=scores)
    fpr, tpr, thresholds = roc_curve(y_true=y_tr, y_score=scores)
    p = len([_ for _ in y_tr if _ > 0])
    n = len(y_tr) - p
    best_acc, best_f1, best_threshold = -1., -1., -1.0
    for fpr_, tpr_, threshold in zip(fpr, tpr, thresholds):
        tp = tpr_ * p
        fp = fpr_ * n
        tn = n - fp
        fn = p - tp
        acc = (tp + tn) / (p + n)
        if best_acc < acc:
            best_acc = acc
            best_f1 = (2. * tp) / (2. * tp + fp + fn)
            best_threshold = threshold
    decision = np.dot(x_tr, w)
    pred_prob = softmax(np.c_[-decision, decision])
    loss = log_loss(y_true=y_tr, y_pred=pred_prob)
    acc, f1, threshold = best_acc, best_f1, best_threshold
    return auc, acc, f1, loss, threshold


def draw(x_tr):
    import matplotlib.pyplot as plt
    n, d = x_tr.shape
    fig, ax = plt.subplots(1, 1)
    for ii, xx in enumerate(x_tr):
        ax.scatter(xx[0], xx[1])
    for i, txt in enumerate(range(n)):
        ax.annotate(txt, (x_tr[i][0], x_tr[i][1]), fontsize=16)
    plt.show()


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


def test_0():
    results = dict()
    rand_state_list = range(10)
    posi_ratio_list = list(np.arange(0.1, 0.5, 0.02))
    for n in np.arange(100, 5000, step=100):
        results[n] = dict()
        results[n]['best_auc'] = np.zeros(shape=(len(posi_ratio_list), len(rand_state_list)))
        results[n]['lr_auc'] = np.zeros(shape=(len(posi_ratio_list), len(rand_state_list)))
        results[n]['miss_pairs'] = np.zeros(shape=(len(posi_ratio_list), len(rand_state_list)))
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for ind_posi, posi_ratio in enumerate(posi_ratio_list):
            for ind_rand, rand_state in enumerate(rand_state_list):
                x_tr, y_tr, x_te, y_te = get_2d_samples(n=n, rand_state=rand_state,
                                                        posi_ratio=posi_ratio)
                auc_list_fast, best_auc_fast, best_slope = linear_auc_2d_fast(x_tr=x_tr, y_tr=y_tr, rand_state=17)
                lr = LogisticRegression(penalty='l2', tol=1e-5, max_iter=2000, random_state=rand_state)
                lr.fit(X=x_tr, y=y_tr)
                pred_prob = lr.predict_proba(X=x_tr)[:, 1]
                auc = roc_auc_score(y_true=y_tr, y_score=pred_prob)
                diff_auc = best_auc_fast - auc
                num_posi = len([_ for _ in y_tr if _ > 0.0])
                num_pairs = (len(y_tr) - num_posi) * num_posi
                results[n]['best_auc'][ind_posi][ind_rand] = best_auc_fast
                results[n]['lr_auc'][ind_posi][ind_rand] = auc
                results[n]['miss_pairs'][ind_posi][ind_rand] = int(diff_auc * num_pairs)
        ax[0].plot(np.mean(results[n]['best_auc'], axis=1), label='Best')
        ax[0].plot(np.mean(results[n]['lr_auc'], axis=1), label='LR')
        ax[0].legend()
        ax[1].plot(np.mean(results[n]['miss_pairs'], axis=1), label='Missed')
        file_name = 'auc_comp_%04d' % n
        f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s.pdf' % file_name
        fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
        plt.close()


def four_class(x, y):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.scatter(x[:, 0][np.argwhere(y > 0)],
                x[:, 1][np.argwhere(y > 0)], c='r', s=4, label='posi')
    plt.scatter(x[:, 0][np.argwhere(y < 0)],
                x[:, 1][np.argwhere(y < 0)], c='b', s=4, label='nega')
    plt.legend()
    file_name = 'fourclass_points'
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s.pdf' % file_name
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def draw_heat_map_fig(data, data_ind, best_w, lr_w, wei_lr_w, linear_svm_w, perceptron_w,
                      opt_auc_auc, lr_auc, wei_lr_auc, linear_svm_auc, perceptron_auc,
                      x_tr, y_tr, x_te, y_te):
    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    h = .1
    for i in range(5):
        ax[i].scatter(x_tr[:, 0][np.argwhere(y_tr > 0)], x_tr[:, 1][np.argwhere(y_tr > 0)], c='r', s=5, label='posi')
        ax[i].scatter(x_tr[:, 0][np.argwhere(y_tr < 0)], x_tr[:, 1][np.argwhere(y_tr < 0)], c='b', s=5, label='nega')
        if False:
            ax[i].scatter(x_te[:, 0][np.argwhere(y_te > 0)], x_te[:, 1][np.argwhere(y_te > 0)],
                          c='r', s=5, label='posi', alpha=0.5)
            ax[i].scatter(x_te[:, 0][np.argwhere(y_te < 0)],
                          x_te[:, 1][np.argwhere(y_te < 0)], c='b', s=5, label='nega', alpha=0.5)
    x_min, x_max = x_tr[:, 0].min() - .5, x_tr[:, 0].max() + .5
    y_min, y_max = x_tr[:, 1].min() - .5, x_tr[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    z_values = np.dot(mesh_points, best_w)
    z_values = z_values.reshape(xx.shape)
    print(np.min(z_values), np.max(z_values))
    pp = np.arange(x_min, x_max, h)
    ax[0].plot(pp, [-(best_w[0] * _) / best_w[1] for _ in pp], linestyle='dashed', zorder=-1)
    ax[0].contourf(xx, yy, z_values, cmap=plt.cm.RdBu_r, alpha=.8, levels=50, zorder=-1)
    ax[0].set_title('OptAUC:%.5f' % opt_auc_auc)
    if False:
        plt.colorbar(mappable=plt.cm.ScalarMappable(
            cmap=plt.cm.RdBu_r, norm=mpl.colors.Normalize(
                vmin=np.min(z_values), vmax=np.max(z_values))), ax=ax[0],
            orientation="horizontal")

    z_values = np.dot(mesh_points, lr_w)
    z_values = z_values.reshape(xx.shape)
    print(np.min(z_values), np.max(z_values))
    ax[1].plot(pp, [-(lr_w[0] * _) / lr_w[1] for _ in pp], linestyle='dashed', zorder=-1)
    ax[1].contourf(xx, yy, z_values, cmap=plt.cm.RdBu_r, alpha=.8, levels=50, zorder=-1)
    ax[1].set_title('LR: %.5f' % lr_auc)
    if False:
        plt.colorbar(mappable=plt.cm.ScalarMappable(
            cmap=plt.cm.RdBu_r, norm=mpl.colors.Normalize(
                vmin=np.min(z_values), vmax=np.max(z_values))), ax=ax[1],
            orientation="horizontal")

    z_values = np.dot(mesh_points, wei_lr_w)
    z_values = z_values.reshape(xx.shape)
    print(np.min(z_values), np.max(z_values))
    ax[2].plot(pp, [-(wei_lr_w[0] * _) / wei_lr_w[1] for _ in pp], linestyle='dashed', zorder=-1)
    ax[2].contourf(xx, yy, z_values, cmap=plt.cm.RdBu_r, alpha=.8, levels=50, zorder=-1)
    ax[2].set_title('Wei-LR: %.5f' % wei_lr_auc)
    if False:
        plt.colorbar(mappable=plt.cm.ScalarMappable(
            cmap=plt.cm.RdBu_r, norm=mpl.colors.Normalize(
                vmin=np.min(z_values), vmax=np.max(z_values))), ax=ax[2],
            orientation="horizontal")

    z_values = np.dot(mesh_points, linear_svm_w)
    z_values = z_values.reshape(xx.shape)
    print(np.min(z_values), np.max(z_values))
    ax[3].plot(pp, [-(linear_svm_w[0] * _) / linear_svm_w[1] for _ in pp], linestyle='dashed', zorder=-1)
    ax[3].contourf(xx, yy, z_values, cmap=plt.cm.RdBu_r, alpha=.8, levels=50, zorder=-1)
    ax[3].set_title('LinearSVM: %.5f' % linear_svm_auc)
    if False:
        plt.colorbar(mappable=plt.cm.ScalarMappable(
            cmap=plt.cm.RdBu_r, norm=mpl.colors.Normalize(
                vmin=np.min(z_values), vmax=np.max(z_values))), ax=ax[3],
            orientation="horizontal")

    z_values = np.dot(mesh_points, perceptron_w)
    z_values = z_values.reshape(xx.shape)
    print(np.min(z_values), np.max(z_values))
    ax[4].plot(pp, [-(perceptron_w[0] * _) / perceptron_w[1] for _ in pp], linestyle='dashed', zorder=-1)
    ax[4].contourf(xx, yy, z_values, cmap=plt.cm.RdBu_r, alpha=.8, levels=50, zorder=-1)
    ax[4].set_title('Perceptron: %.5f' % perceptron_auc)
    if False:
        plt.colorbar(mappable=plt.cm.ScalarMappable(
            cmap=plt.cm.RdBu_r, norm=mpl.colors.Normalize(
                vmin=np.min(z_values), vmax=np.max(z_values))), ax=ax[4],
            orientation="horizontal")

    for i in range(5):
        ax[i].set_xlim(xx.min(), xx.max())
        ax[i].set_ylim(yy.min(), yy.max())
        ax[i].set_xticks(())
        ax[i].set_yticks(())
    file_name = 't_sne_heat_map_%02d_%s' % (data_ind[data], data)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s.png' % file_name
    plt.subplots_adjust(wspace=0.01, hspace=0.05)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.1, format='png')
    plt.close()
    exit()


def draw_matrix():
    matrix = [[1., 0.9052867361476489, 0.9003573614484587, 0.7778160955780785, 0.7926330363937756],
              [0.9052867361476489, 1., 0.9999342870960848, 0.9711306264932386, 0.9765609446941932],
              [0.9003573614484587, 0.9999342870960848, 1., 0.9738015099539996, 0.9789642787896842],
              [0.7778160955780785, 0.9711306264932386, 0.9738015099539996, 1., 0.9997136417782656],
              [0.7926330363937756, 0.9765609446941932, 0.9789642787896842, 0.9997136417782656, 1.]]
    matrix = [[1., 0.9969953996383635, 0.995017113680539, 0.9999065797990837, 0.9988338837945169],
              [0.9969953996383635, 1., 0.9997506566375914, 0.9958434766808631, 0.9995725249681033],
              [0.995017113680539, 0.9997506566375914, 1., 0.9935613372338002, 0.9986704421345403],
              [0.9999065797990837, 0.9958434766808631, 0.9935613372338002, 1., 0.9980806628204774],
              [0.9988338837945169, 0.9995725249681033, 0.9986704421345403, 0.9980806628204774, 1.]]
    name_list = ["Opt-AUC & ", "LR & ", "Wei-LR & ", "Linear-SVM & ", "Perceptron & "]
    for i in range(5):
        print(name_list[i], end='')
        print(' & '.join(['%.4f' % _ for _ in matrix[i]]), end='')
        print(' \\\\\\hline ')
    exit()


def show_results():
    for dataset in ['01_splice', '03_australian', '04_spambase', '05_ionosphere',
                    '06_fourclass', '07_breast_cancer', '08_pima', '10_german',
                    '11_svmguide3', '13_spectf', '14_pen_digits', '15_page_blocks',
                    '16_opt_digits', '19_yeast_me1']:
        results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
        for trial_i in range(0):
            print('---- ' + dataset + ' ----')
            matrix = np.zeros((5, 5))
            w_list = [results_mat[trial_i]['opt-auc'][0],
                      results_mat[trial_i]['lr'][0],
                      results_mat[trial_i]['wei-lr'][0],
                      results_mat[trial_i]['linear-svm'][0],
                      results_mat[trial_i]['perceptron'][0]]
            for i in range(len(w_list)):
                for j in range(len(w_list)):
                    a, b = np.linalg.norm(w_list[i]), np.linalg.norm(w_list[j])
                    matrix[i][j] = np.dot(w_list[i], w_list[j]) / (a * b)
            name_list = ["Opt-AUC & ", "LR & ", "Wei-LR & ", "Linear-SVM & ", "Perceptron & "]
            for i in range(5):
                print(name_list[i], end='')
                print(' & '.join(['%.4f' % _ for _ in matrix[i]]), end='')
                print(' \\\\\\hline ')

        print('---- ' + dataset + ' ----')
        print('Opt-AUC & ', end='')
        print(' & '.join(['%.5f' % _ for _ in np.mean(results_mat['opt-auc'], axis=0)]), end='')
        print('\\\\\\hline')
        print('LR & ', end='')
        print(' & '.join(['%.5f' % _ for _ in np.mean(results_mat['lr'], axis=0)]), end='')
        print('\\\\\\hline')
        print('Wei-LR & ', end='')
        print(' & '.join(['%.5f' % _ for _ in np.mean(results_mat['wei-lr'], axis=0)]), end='')
        print('\\\\\\hline')
        print('Linear-SVM & ', end='')
        print(' & '.join(['%.5f' % _ for _ in np.mean(results_mat['linear-svm'], axis=0)]), end='')
        print('\\\\\\hline')
        print('Perceptron & ', end='')
        print(' & '.join(['%.5f' % _ for _ in np.mean(results_mat['perceptron'], axis=0)]), end='')
        print('\\\\\\hline')


def show_results_by_metric(metric):
    metric_dict = {'auc': 0, 'acc': 1, 'f1': 2, 'loss': 3}
    index = metric_dict[metric]
    list_datasets = ['01_splice', '03_australian', '04_spambase', '05_ionosphere',
                     '06_fourclass', '07_breast_cancer', '08_pima', '10_german',
                     '11_svmguide3', '13_spectf', '14_pen_digits', '15_page_blocks',
                     '16_opt_digits', '19_yeast_me1']
    results_mat = pkl.load(open('results_2d_14_pen_digits.pkl', 'rb'))
    # results_mat = pkl.load(open('results_2d_01_splice.pkl', 'rb'))
    threshold = results_mat['opt-auc'][0][4]
    w = results_mat[0]['opt-auc'][0]
    scores = np.dot(results_mat[0]['x_tr'], w)
    y_pred = [1 if _ > threshold else -1 for _ in scores]
    f1 = f1_score(y_true=results_mat[0]['y_tr'], y_pred=y_pred, pos_label=1, average='binary')
    precision = precision_score(y_true=results_mat[0]['y_tr'], y_pred=y_pred, pos_label=1, average='binary')
    print(precision)
    recall = recall_score(y_true=results_mat[0]['y_tr'], y_pred=y_pred, pos_label=1, average='binary')
    print(recall)
    print(f1, precision, recall)
    exit()
    for dataset in list_datasets:
        results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
        print('-'.join(dataset.split('_')[1:]), end='')
        re1 = results_mat['opt-auc'][:, index]
        print(' & %.5f' % np.mean(re1), end='')
        re2 = results_mat['lr'][:, index]
        print(' & %.5f' % np.mean(re2), end='')
        re3 = results_mat['wei-lr'][:, index]
        print(' & %.5f' % np.mean(re3), end='')
        re4 = results_mat['linear-svm'][:, index]
        print(' & %.5f' % np.mean(re4), end='')
        re5 = results_mat['perceptron'][:, index]
        print(' & %.5f' % np.mean(re5), end='')
        print(' \\\\\\hline ')
        if False:
            for re_a, re_b in zip([re1, re1, re1, re1], [re2, re3, re4, re5]):
                t_stat, p_val = ttest_ind(a=re_a, b=re_b)
                if p_val <= 0.05:
                    print('%.5f' % p_val, end=' * ')
                else:
                    print('%.5f' % p_val, end=' ')
            print('\n')
    print('---\n\n')
    for dataset in list_datasets:
        results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
        print('-'.join(dataset.split('_')[1:]), end='')
        re1 = results_mat['opt-auc'][:, 5 + index]
        print(' & %.5f' % np.mean(re1), end='')
        re2 = results_mat['lr'][:, 5 + index]
        print(' & %.5f' % np.mean(re2), end='')
        re3 = results_mat['wei-lr'][:, 5 + index]
        print(' & %.5f' % np.mean(re3), end='')
        re4 = results_mat['linear-svm'][:, 5 + index]
        print(' & %.5f' % np.mean(re4), end='')
        re5 = results_mat['perceptron'][:, 5 + index]
        print(' & %.5f' % np.mean(re5), end='')
        print(' \\\\\\hline ')
        if False:
            for re_a, re_b in zip([re1, re1, re1, re1], [re2, re3, re4, re5]):
                t_stat, p_val = ttest_ind(a=re_a, b=re_b)
                if p_val <= 0.05:
                    print('%.5f' % p_val, end=' * ')
                else:
                    print('%.5f' % p_val, end=' ')
            print('\n')


def cal_adjusted_rand_score(x_tr, y_tr, x_te, y_te, w1, w2):
    score1 = np.dot(x_tr, w1)
    score2 = np.dot(x_tr, w2)
    cluster1, cluster2 = [], []
    for (i, yi), (j, yj) in product(enumerate(y_tr), enumerate(y_tr)):
        if yi > yj:
            if score1[i] > score1[j]:
                cluster1.append(1)
            else:
                cluster1.append(0)
            if score2[i] > score2[j]:
                cluster2.append(1)
            else:
                cluster2.append(0)
    rs_tr = adjusted_rand_score(labels_true=cluster1, labels_pred=cluster2)
    score1 = np.dot(x_te, w1)
    score2 = np.dot(x_te, w2)
    cluster1, cluster2 = [], []
    for (i, yi), (j, yj) in product(enumerate(y_te), enumerate(y_te)):
        if yi > yj:
            if score1[i] > score1[j]:
                cluster1.append(1)
            else:
                cluster1.append(0)
            if score2[i] > score2[j]:
                cluster2.append(1)
            else:
                cluster2.append(0)
    rs_te = adjusted_rand_score(labels_true=cluster1, labels_pred=cluster2)
    return rs_tr, rs_te


def show_results_by_rand_index():
    list_datasets = ['01_splice', '03_australian', '04_spambase', '05_ionosphere',
                     '06_fourclass', '07_breast_cancer', '08_pima', '10_german',
                     '11_svmguide3', '13_spectf', '14_pen_digits', '15_page_blocks',
                     '16_opt_digits', '19_yeast_me1']
    for dataset in list_datasets:
        results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
        rand_mat = np.zeros(shape=(20, 4))
        for trial_i in range(20):
            w1 = results_mat[trial_i]['opt-auc'][0]
            for ind_method, method in enumerate(['lr', 'wei-lr', 'linear-svm', 'perceptron']):
                w2 = results_mat[trial_i][method][0]
                x_tr, y_tr = results_mat[trial_i]['x_tr'], results_mat[trial_i]['y_tr']
                x_te, y_te = results_mat[trial_i]['x_te'], results_mat[trial_i]['y_te']
                score1 = np.dot(x_tr, w1)
                score2 = np.dot(x_tr, w2)
                cluster1, cluster2 = [], []
                for (i, yi), (j, yj) in product(enumerate(y_tr), enumerate(y_tr)):
                    if yi > yj:
                        if score1[i] > score1[j]:
                            cluster1.append(1)
                        else:
                            cluster1.append(0)
                        if score2[i] > score2[j]:
                            cluster2.append(1)
                        else:
                            cluster2.append(0)
                rs = adjusted_rand_score(labels_true=cluster1, labels_pred=cluster2)
                rand_mat[trial_i][ind_method] = rs
        print('-'.join(dataset.split('_')[1:]), end=' & ')
        print(' & '.join(['%.5f' % _ for _ in np.mean(rand_mat, axis=0)]), end='')
        print(' \\\\\\hline ')


def draw_heat_plot():
    dataset = '14_pen_digits'
    results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
    trial_i = 0
    data_ind = {'splice': 1, 'mushrooms': 2, 'australian': 3, 'spambase': 4,
                'ionosphere': 5, 'fourclass': 6, 'breast_cancer': 7, 'pima': 8,
                'yeast_cyt': 9, 'german': 10, 'svmguide3': 11, 'a9a': 12,
                'spectf': 13, 'pen_digits': 14, 'page_blocks': 15, 'opt_digits': 16,
                'ijcnn1': 17, 'letter_a': 18, 'yeast_me1': 19, 'w8a': 20}
    draw_heat_map_fig('pen_digits', data_ind,
                      results_mat[trial_i]['opt-auc'][0],
                      results_mat[trial_i]['lr'][0],
                      results_mat[trial_i]['wei-lr'][0],
                      results_mat[trial_i]['linear-svm'][0],
                      results_mat[trial_i]['perceptron'][0],
                      results_mat[trial_i]['opt-auc'][1],
                      results_mat[trial_i]['lr'][1],
                      results_mat[trial_i]['wei-lr'][1],
                      results_mat[trial_i]['linear-svm'][1],
                      results_mat[trial_i]['perceptron'][1],
                      results_mat[trial_i]['x_tr'], results_mat[trial_i]['y_tr'],
                      results_mat[trial_i]['x_te'], results_mat[trial_i]['y_te'])


def auc_curves(dataset):
    results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
    trial_i = 0
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for method in ['opt-auc', 'lr', 'wei-lr', 'linear-svm', 'perceptron']:
        w = results_mat[trial_i][method][0]
        x_tr, y_tr = results_mat[trial_i]['x_tr'], results_mat[trial_i]['y_tr']
        x_te, y_te = results_mat[trial_i]['x_te'], results_mat[trial_i]['y_te']
        fpr, tpr, _ = roc_curve(y_true=y_tr, y_score=np.dot(x_tr, w))
        ax[0].plot(fpr, tpr, label=method)
        fpr, tpr, _ = roc_curve(y_true=y_te, y_score=np.dot(x_te, w))
        ax[1].plot(fpr, tpr, label=method)
    plt.legend()
    plt.show()
    pass


def cal_compare_matrix_2d(measure='f1'):
    list_datasets = ['01_splice', '03_australian', '04_spambase', '05_ionosphere',
                     '06_fourclass', '07_breast_cancer', '08_pima', '10_german',
                     '11_svmguide3', '13_spectf', '14_pen_digits', '15_page_blocks',
                     '16_opt_digits', '19_yeast_me1']
    list_datasets = ['03_australian', '05_ionosphere', '06_fourclass', '07_breast_cancer',
                     '08_pima', '10_german', '11_svmguide3', '13_spectf',
                     '15_page_blocks', '19_yeast_me1']
    num_trials = 100
    for metric in ['tr', 'te']:
        print('--- ' + metric + ' ---')
        comp_mat = np.zeros(shape=(7, 7))
        method_list = ['opt-auc', 'lr', 'wei-lr', 'svm', 'wei-svm', 'percep', 'wei-percep']
        method_label_list = ['Opt-AUC', 'LR', 'B-LR', 'SVM', 'B-SVM', 'Percep', 'B-Percep']
        for dataset in list_datasets:
            results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
            for (ind1, method1), (ind2, method2) in product(
                    enumerate(method_list), enumerate(method_list)):
                for trial_i in range(num_trials):
                    print(dataset, trial_i)
                    if measure == 'loss':
                        auc1 = results_mat[trial_i][method1][metric][measure]
                        auc2 = results_mat[trial_i][method2][metric][measure]
                        # ignore the results that have the same AUCs.
                        if np.abs(auc1 - auc2) <= 1e-15:
                            continue
                        if auc1 < auc2:
                            comp_mat[ind1][ind2] += 1.
                    elif measure == 'rand_index':
                        w1 = results_mat[trial_i][method1]['w']
                        w2 = results_mat[trial_i][method2]['w']
                        x_tr, y_tr = results_mat[trial_i]['x_tr'], results_mat[trial_i]['y_tr']
                        x_te, y_te = results_mat[trial_i]['x_te'], results_mat[trial_i]['y_te']
                        re_tr, re_te = cal_adjusted_rand_score(x_tr, y_tr, x_te, y_te, w1, w2)
                        if metric == 'tr':
                            comp_mat[ind1][ind2] += re_tr
                        else:
                            comp_mat[ind1][ind2] += re_te
                    else:
                        auc1 = results_mat[trial_i][method1][metric][measure]
                        auc2 = results_mat[trial_i][method2][metric][measure]
                        # ignore the results that have the same AUCs.
                        if np.abs(auc1 - auc2) <= 1e-15:
                            continue
                        if auc1 > auc2:
                            comp_mat[ind1][ind2] += 1.
        comp_mat_average = np.copy(comp_mat)
        for i, j in product(range(len(comp_mat)), range(len(comp_mat))):
            if i != j:
                comp_mat_average[i][j] /= (comp_mat[i][j] + comp_mat[j][i])
        for i in range(len(comp_mat)):
            print(method_label_list[i], end=' & ')
            str_ = []
            for j in range(len(comp_mat_average)):
                if i == j:
                    str_.append('-')
                else:
                    str_.append('%.3f' % comp_mat_average[i][j])
            print(' & '.join(str_), end=' ')
            print(' \\\\\\hline ')
        print('---')


def cal_cosine_similiarity():
    list_datasets = ['01_splice', '03_australian', '04_spambase', '05_ionosphere',
                     '06_fourclass', '07_breast_cancer', '08_pima', '10_german',
                     '11_svmguide3', '13_spectf', '14_pen_digits', '15_page_blocks',
                     '16_opt_digits', '19_yeast_me1']
    list_datasets = ['03_australian', '05_ionosphere', '06_fourclass', '07_breast_cancer',
                     '08_pima', '10_german', '11_svmguide3', '13_spectf',
                     '15_page_blocks', '19_yeast_me1']
    num_trials = 100
    for metric in ['w']:
        print('--- ' + metric + ' ---')
        comp_mat = np.zeros(shape=(7, 7))
        method_list = ['opt-auc', 'lr', 'wei-lr', 'svm', 'wei-svm', 'percep', 'wei-percep']
        method_label_list = ['Opt-AUC', 'LR', 'B-LR', 'SVM', 'B-SVM', 'Percep', 'B-Percep']
        for dataset in list_datasets:
            results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
            for (ind1, method1), (ind2, method2) in product(
                    enumerate(method_list), enumerate(method_list)):
                for trial_i in range(num_trials):
                    w1 = results_mat[trial_i][method1][metric]
                    w2 = results_mat[trial_i][method2][metric]
                    comp_mat[ind1][ind2] += np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
        comp_mat /= (num_trials * len(list_datasets))
        for i in range(len(comp_mat)):
            print(method_label_list[i], end=' & ')
            str_ = []
            for j in range(len(comp_mat)):
                if i == j:
                    str_.append('-')
                else:
                    str_.append('%.3f' % comp_mat[i][j])
            print(' & '.join(str_), end=' ')
            print(' \\\\\\hline ')
        print('---')


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


def cal_size():
    data_ind = {'splice': 1, 'mushrooms': 2, 'australian': 3, 'spambase': 4,
                'ionosphere': 5, 'fourclass': 6, 'breast_cancer': 7, 'pima': 8,
                'yeast_cyt': 9, 'german': 10, 'svmguide3': 11, 'a9a': 12,
                'spectf': 13, 'pen_digits': 14, 'page_blocks': 15, 'opt_digits': 16,
                'ijcnn1': 17, 'letter_a': 18, 'yeast_me1': 19, 'w8a': 20}
    for data_name in data_ind:
        data = pkl.load(open('/data/auc-logistic/%02d_%s/t_sne_2d_%s.pkl'
                             % (data_ind[data_name], data_name, data_name), 'rb'))
        n, data_y, embeddings = len(data['x_tr']), data['y_tr'], data['embeddings'][50.]
        rand_perm = np.random.permutation(n)
        x_tr, y_tr = embeddings[rand_perm[:n // 2]], data_y[rand_perm[:n // 2]]
        x_te, y_te = embeddings[rand_perm[n // 2:]], data_y[rand_perm[n // 2:]]
        len_posi = len([_ for _ in y_tr if _ > 0])
        len_nega = len(y_tr) - len_posi
        print(data_name, '%d K items %.4f MB' %
              ((len_posi * len_nega) / 1000, (len_posi * len_nega * 8) / (1024. * 1024.)))


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


def cal_compare_matrix_real(measure='auc'):
    data_ind = {'splice': 1, 'mushrooms': 2, 'australian': 3, 'spambase': 4,
                'ionosphere': 5, 'fourclass': 6, 'breast_cancer': 7, 'pima': 8,
                'yeast_cyt': 9, 'german': 10, 'svmguide3': 11, 'a9a': 12,
                'spectf': 13, 'pen_digits': 14, 'page_blocks': 15, 'opt_digits': 16,
                'ijcnn1': 17, 'letter_a': 18, 'yeast_me1': 19, 'w8a': 20}
    list_datasets = ['splice', 'australian', 'ionosphere', 'fourclass', 'breast_cancer', 'pima', 'yeast_cyt',
                     'german', 'svmguide3', 'spectf', 'page_blocks', 'yeast_me1']
    list_datasets = ['splice', 'australian', 'ionosphere', 'fourclass', 'breast_cancer', 'pima', 'yeast_cyt',
                     'german', 'svmguide3', 'spectf']
    list_datasets = ['splice', 'australian', 'spambase', 'ionosphere', 'breast_cancer', 'pima', 'yeast_cyt',
                     'german', 'svmguide3', 'spectf', 'page_blocks']
    num_trials = 200
    for metric in ['tr', 'te']:
        print('--- ' + metric + ' ---')
        comp_mat = np.zeros(shape=(7, 7))
        method_list = ['rank_boost', 'svm_perf', 'lr', 'wei_lr', 'c_svm', 'wei_c_svm', 'rbf_svm']
        method_label_list = ['RankBoost', 'SVM-Perf', 'LR', 'B-LR', 'SVM', 'B-SVM', 'RBF-SVM']
        for dataset in list_datasets:
            results_mat = dict()
            for method in method_list:
                re = pkl.load(open('/data/auc-logistic/re_%02d_%s_%s.pkl'
                                   % (data_ind[dataset], dataset, method), 'rb'))
                for trial_i in re:
                    if trial_i not in results_mat:
                        results_mat[trial_i] = dict()
                    results_mat[trial_i][method] = re[trial_i]
            for (ind1, method1), (ind2, method2) in product(
                    enumerate(method_list), enumerate(method_list)):
                for trial_i in range(num_trials):
                    auc1 = results_mat[trial_i][method1][metric][measure]
                    auc2 = results_mat[trial_i][method2][metric][measure]
                    # ignore the results that have the same AUCs.
                    if np.abs(auc1 - auc2) <= 1e-15:
                        continue
                    if auc1 > auc2:
                        comp_mat[ind1][ind2] += 1.
        comp_mat_average = np.copy(comp_mat)
        for i, j in product(range(len(comp_mat)), range(len(comp_mat))):
            if i != j:
                comp_mat_average[i][j] /= (comp_mat[i][j] + comp_mat[j][i])
        for i in range(len(comp_mat)):
            print(method_label_list[i], end=' & ')
            str_ = []
            for j in range(len(comp_mat_average)):
                if i == j:
                    str_.append('-')
                else:
                    str_.append('%.3f' % comp_mat_average[i][j])
            print(' & '.join(str_), end=' ')
            print(' \\\\\\hline ')
        print('---')


def cal_compare_matrix_auc_pairs():
    list_datasets = ['03_australian', '05_ionosphere', '06_fourclass', '07_breast_cancer',
                     '08_pima', '10_german', '11_svmguide3', '13_spectf',
                     '15_page_blocks', '19_yeast_me1']
    num_trials = 100
    for metric in ['tr', 'te']:
        print('--- ' + metric + ' ---')
        comp_mat = np.zeros(shape=(7, 7))
        method_list = ['opt-auc', 'lr', 'wei-lr', 'svm', 'wei-svm', 'percep', 'wei-percep']
        method_label_list = ['Opt-AUC', 'LR', 'B-LR', 'SVM', 'B-SVM', 'Percep', 'B-Percep']
        for dataset in list_datasets:
            results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
            for (ind1, method1), (ind2, method2) in product(
                    enumerate(method_list), enumerate(method_list)):
                list_points = []
                for trial_i in range(num_trials):
                    auc1 = results_mat[trial_i][method1][metric]['auc']
                    auc2 = results_mat[trial_i][method2][metric]['auc']
                    list_points.append((auc1, auc2))
                list_points = np.asarray(list_points)
                plt.scatter(list_points[:, 0], list_points[:, 1])
                plt.xlabel('%s' % method1)
                plt.ylabel('%s' % method2)
                plt.xlim([np.min(list_points), np.max(list_points)])
                plt.ylim([np.min(list_points), np.max(list_points)])
                plt.plot([np.min(list_points), np.max(list_points)],
                         [np.min(list_points), np.max(list_points)], c='gray')
                plt.title(dataset)
                plt.show()
        comp_mat_average = np.copy(comp_mat)
        for i, j in product(range(len(comp_mat)), range(len(comp_mat))):
            if i != j:
                comp_mat_average[i][j] /= (comp_mat[i][j] + comp_mat[j][i])
        for i in range(len(comp_mat)):
            print(method_label_list[i], end=' & ')
            str_ = []
            for j in range(len(comp_mat_average)):
                if i == j:
                    str_.append('-')
                else:
                    str_.append('%.3f' % comp_mat_average[i][j])
            print(' & '.join(str_), end=' ')
            print(' \\\\\\hline ')
        print('---')


def cal_compare_matrix_acc_pairs():
    list_datasets = ['03_australian', '05_ionosphere', '06_fourclass', '07_breast_cancer',
                     '08_pima', '10_german', '11_svmguide3', '13_spectf',
                     '15_page_blocks', '19_yeast_me1']
    num_trials = 100
    for metric in ['tr', 'te']:
        print('--- ' + metric + ' ---')
        comp_mat = np.zeros(shape=(7, 7))
        method_list = ['opt-auc', 'lr', 'wei-lr', 'svm', 'wei-svm', 'percep', 'wei-percep']
        method_label_list = ['Opt-AUC', 'LR', 'B-LR', 'SVM', 'B-SVM', 'Percep', 'B-Percep']
        for dataset in list_datasets:
            results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
            for (ind1, method1), (ind2, method2) in product(
                    enumerate(method_list), enumerate(method_list)):
                list_points = []
                for trial_i in range(num_trials):
                    auc1 = results_mat[trial_i][method1][metric]['acc']
                    auc2 = results_mat[trial_i][method2][metric]['acc']
                    list_points.append((auc1, auc2))
                list_points = np.asarray(list_points)
                plt.scatter(list_points[:, 0], list_points[:, 1])
                plt.xlabel('%s' % method1)
                plt.ylabel('%s' % method2)
                plt.xlim([np.min(list_points), np.max(list_points)])
                plt.ylim([np.min(list_points), np.max(list_points)])
                plt.plot([np.min(list_points), np.max(list_points)],
                         [np.min(list_points), np.max(list_points)], c='gray')
                plt.title(dataset)
                plt.show()
        comp_mat_average = np.copy(comp_mat)
        for i, j in product(range(len(comp_mat)), range(len(comp_mat))):
            if i != j:
                comp_mat_average[i][j] /= (comp_mat[i][j] + comp_mat[j][i])
        for i in range(len(comp_mat)):
            print(method_label_list[i], end=' & ')
            str_ = []
            for j in range(len(comp_mat_average)):
                if i == j:
                    str_.append('-')
                else:
                    str_.append('%.3f' % comp_mat_average[i][j])
            print(' & '.join(str_), end=' ')
            print(' \\\\\\hline ')
        print('---')


def cal_compare_matrix_auc_pairs2():
    data_ind = {'splice': 1, 'mushrooms': 2, 'australian': 3, 'spambase': 4,
                'ionosphere': 5, 'fourclass': 6, 'breast_cancer': 7, 'pima': 8,
                'yeast_cyt': 9, 'german': 10, 'svmguide3': 11, 'a9a': 12,
                'spectf': 13, 'pen_digits': 14, 'page_blocks': 15, 'opt_digits': 16,
                'ijcnn1': 17, 'letter_a': 18, 'yeast_me1': 19, 'w8a': 20}
    list_datasets = ['splice', 'australian', 'spambase', 'ionosphere', 'breast_cancer', 'pima', 'yeast_cyt',
                     'german', 'svmguide3', 'spectf', 'page_blocks']
    num_trials = 200
    for metric in ['tr', 'te']:
        print('--- ' + metric + ' ---')
        comp_mat = np.zeros(shape=(7, 7))
        method_list = ['rank_boost', 'svm_perf', 'lr', 'wei_lr', 'c_svm', 'wei_c_svm', 'rbf_svm']
        method_label_list = ['RankBoost', 'SVM-Perf', 'LR', 'B-LR', 'SVM', 'B-SVM', 'RBF-SVM']
        for dataset in list_datasets:
            fig, ax = plt.subplots(7, 7, figsize=(8, 8))
            results_mat = dict()
            for method in method_list:
                re = pkl.load(open('/data/auc-logistic/re_%02d_%s_%s.pkl'
                                   % (data_ind[dataset], dataset, method), 'rb'))
                for trial_i in re:
                    if trial_i not in results_mat:
                        results_mat[trial_i] = dict()
                    results_mat[trial_i][method] = re[trial_i]
            for (ind1, method1), (ind2, method2) in product(
                    enumerate(method_list), enumerate(method_list)):
                list_points = []
                for trial_i in range(num_trials):
                    auc1 = results_mat[trial_i][method1][metric]['auc']
                    auc2 = results_mat[trial_i][method2][metric]['auc']
                    list_points.append((auc1, auc2))

                    # ignore the results that have the same AUCs.
                    if np.abs(auc1 - auc2) <= 1e-15:
                        continue
                    if auc1 > auc2:
                        comp_mat[ind1][ind2] += 1.

                list_points = np.asarray(list_points)
                ax[ind1, ind2].plot([np.min(list_points) - 0.01, np.max(list_points) + 0.01],
                                    [np.min(list_points) - 0.01, np.max(list_points) + 0.01],
                                    c='lightgray', linestyle='dashed', zorder=-1)
                ax[ind1, ind2].scatter(list_points[:, 1], list_points[:, 0], s=5)
                if ind2 == 0:
                    ax[ind1, ind2].set_ylabel('%s' % method_label_list[ind1])
                if ind1 == 6:
                    ax[ind1, ind2].set_xlabel('%s' % method_label_list[ind2])

                ax[ind1, ind2].set_xlim([np.min(list_points) - 0.002, np.max(list_points) + 0.002])
                ax[ind1, ind2].set_ylim([np.min(list_points) - 0.002, np.max(list_points) + 0.002])
                ax[ind1, ind2].set_xticks([])
                ax[ind1, ind2].set_yticks([])

            f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/%s_%s_matrix.pdf' % (metric, dataset)
            plt.subplots_adjust(wspace=0.0, hspace=0.0)
            fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.1, format='pdf')
            plt.close()
        comp_mat_average = np.copy(comp_mat)
        for i, j in product(range(len(comp_mat)), range(len(comp_mat))):
            if i != j:
                comp_mat_average[i][j] /= (comp_mat[i][j] + comp_mat[j][i])
        for i in range(len(comp_mat)):
            print(method_label_list[i], end=' & ')
            str_ = []
            for j in range(len(comp_mat_average)):
                if i == j:
                    str_.append('-')
                else:
                    str_.append('%.3f' % comp_mat_average[i][j])
            print(' & '.join(str_), end=' ')
            print(' \\\\\\hline ')
        print('---')


def average_rank():
    list_datasets = ['03_australian', '05_ionosphere', '06_fourclass', '07_breast_cancer',
                     '08_pima', '10_german', '11_svmguide3', '13_spectf',
                     '15_page_blocks', '19_yeast_me1']
    num_trials = 100
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for metric_index, metric in enumerate(['tr', 'te']):
        print('--- ' + metric + ' ---')
        comp_mat = np.zeros(shape=(len(list_datasets), 7))
        method_list = ['opt-auc', 'lr', 'wei-lr', 'svm', 'wei-svm', 'percep', 'wei-percep']
        method_label_list = ['Opt-AUC', 'LR', 'B-LR', 'SVM', 'B-SVM', 'Percep', 'B-Percep']
        for data_ind, dataset in enumerate(list_datasets):
            results_mat = pkl.load(open('results_2d_%s.pkl' % dataset, 'rb'))
            rank_mat = np.zeros((num_trials, len(method_list)))
            for trial_i in range(num_trials):
                aucs = np.zeros(len(method_list))
                for (ind, method) in enumerate(method_list):
                    aucs[ind] = results_mat[trial_i][method][metric]['auc']
                    if method == 'opt-auc':
                        aucs[ind] += 1e-15
                ranks = np.argsort(aucs)[::-1] + 1
                rank_mat[trial_i] = ranks
            print(np.mean(rank_mat, axis=0))
            comp_mat[data_ind] = list(np.mean(rank_mat, axis=0))
        for i in range(7):
            ax[metric_index].plot(comp_mat[:, i], label=method_label_list[i], marker='D')
        ax[1].legend(bbox_to_anchor=(1.4, 0.9), ncol=1)
        ax[metric_index].set_ylabel('Averaged Rank', fontsize=15)
        ax[metric_index].set_xlabel('Dataset', fontsize=15)
    ax[0].set_title('Training', fontsize=15)
    ax[1].set_title('Testing', fontsize=15)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/rank_auc.pdf'
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def average_rank_real():
    data_ind1 = {'splice': 1, 'mushrooms': 2, 'australian': 3, 'spambase': 4,
                 'ionosphere': 5, 'fourclass': 6, 'breast_cancer': 7, 'pima': 8,
                 'yeast_cyt': 9, 'german': 10, 'svmguide3': 11, 'a9a': 12,
                 'spectf': 13, 'pen_digits': 14, 'page_blocks': 15, 'opt_digits': 16,
                 'ijcnn1': 17, 'letter_a': 18, 'yeast_me1': 19, 'w8a': 20}
    list_datasets = ['splice', 'australian', 'spambase', 'ionosphere', 'breast_cancer', 'pima', 'yeast_cyt',
                     'german', 'svmguide3', 'spectf', 'page_blocks']
    num_trials = 200
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for metric_index, metric in enumerate(['tr', 'te']):
        print('--- ' + metric + ' ---')
        comp_mat = np.zeros(shape=(len(list_datasets), 7))
        method_list = ['rank_boost', 'svm_perf', 'lr', 'wei_lr', 'c_svm', 'wei_c_svm', 'rbf_svm']
        method_label_list = ['RankBoost', 'SVM-Perf', 'LR', 'B-LR', 'SVM', 'B-SVM', 'RBF-SVM']
        for data_ind, dataset in enumerate(list_datasets):
            rank_mat = np.zeros((num_trials, len(method_list)))
            results_mat = {trial_i: {method: None for method in method_list} for trial_i in range(num_trials)}
            for method in method_list:
                re = pkl.load(open('/data/auc-logistic/re_%02d_%s_%s.pkl'
                                   % (data_ind1[dataset], dataset, method), 'rb'))
                for trial_i in range(num_trials):
                    results_mat[trial_i][method] = re[trial_i]
            for trial_i in range(num_trials):
                aucs = np.zeros(len(method_list))
                for (ind, method) in enumerate(method_list):
                    aucs[ind] = results_mat[trial_i][method][metric]['auc']
                ranks = np.argsort(aucs)[::-1] + 1
                rank_mat[trial_i] = ranks
            print(np.mean(rank_mat, axis=0))
            comp_mat[data_ind] = list(np.mean(rank_mat, axis=0))
        for i in range(7):
            ax[metric_index].plot(comp_mat[:, i], label=method_label_list[i], marker='D')
        ax[1].legend(bbox_to_anchor=(1.4, 0.9), ncol=1)
        ax[metric_index].set_ylabel('Averaged Rank', fontsize=15)
        ax[metric_index].set_xlabel('Dataset', fontsize=15)
    ax[0].set_title('Training', fontsize=15)
    ax[1].set_title('Testing', fontsize=15)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/auc-logistic/figs/rank_auc_real.pdf'
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def main():
    cal_compare_matrix_2d()
    exit()
    cal_compare_matrix_real()
    exit()
    cal_compare_matrix_auc_pairs2()


if __name__ == '__main__':
    with open('text.txt', 'r') as f:
        data = [_.rstrip().lstrip().split(' ')[0].split('-')[1] for _ in f.readlines()]
        for ind, item in enumerate(sorted(data)):
            print(ind, item)
        print(len(data))
