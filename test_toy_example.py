# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
import pickle as pkl
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

root_path = '/home/baojian/'


def get_root_path():
    if os.uname()[1] == 'baojian-ThinkPad-T540p':
        root_path = '/data/auc-logistic/'
    elif os.uname()[1] == 'pascal':
        root_path = '/mnt/store2/baojian/data/auc-logistic/'
    elif os.uname()[1].endswith('.rit.albany.edu'):
        root_path = '/network/rit/lab/ceashpc/bz383376/data/auc-logistic/'
    else:
        root_path = '/network/rit/lab/ceashpc/bz383376/data/auc-logistic/'
    return root_path


try:
    import libopt_auc_3

    try:
        from libopt_auc_3 import c_opt_auc

    except ImportError:
        print('cannot find some function(s) in opt_auc')
        exit(0)
except ImportError:
    pass


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


def get_data():
    np.random.seed(1)
    x_nega = np.random.normal(loc=0.0, scale=2.0, size=(700, 2))
    x_posi = np.random.normal(loc=1.0, scale=2.0, size=(100, 2))
    scores_nega = np.asarray([1.4 * _[0] + _[1] - 2. for _ in x_nega])
    x_nega = np.asarray(x_nega[np.argwhere(scores_nega < 0).flatten()])
    scores_posi = np.asarray([1.4 * _[0] + _[1] - 1. for _ in x_posi])
    x_posi = np.asarray(x_posi[np.argwhere(scores_posi > 0).flatten()])
    # outliers are negative samples.
    x_outlier = np.random.normal(loc=10.0, scale=2.0, size=(200, 2))
    x_nega = np.concatenate((x_nega, x_outlier))
    x_tr = np.concatenate((x_posi, x_nega))
    y_tr = np.ones(len(x_tr))
    y_tr[len(x_posi):] *= -1.
    perm = np.random.permutation(len(x_tr))
    x_tr = x_tr[perm]
    y_tr = y_tr[perm]
    h = 0.1
    x_tr = StandardScaler().fit_transform(X=x_tr)
    x_min, x_max = x_tr[:, 0].min() - .1, x_tr[:, 0].max() + .1
    y_min, y_max = x_tr[:, 0].min() - .1, x_tr[:, 0].max() + .1
    pp = np.arange(x_min, x_max, h)
    return x_tr, y_tr, pp, x_min, x_max, y_min, y_max, h


def toy_example():
    x_tr, y_tr, pp, x_min, x_max, y_min, y_max, h = get_data()
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
    ax1.scatter(x_tr[np.argwhere(y_tr < 0), 0], x_tr[np.argwhere(y_tr < 0), 1],
                c='b', marker='_', s=10, alpha=0.8)
    ax1.scatter(x_tr[np.argwhere(y_tr > 0), 0], x_tr[np.argwhere(y_tr > 0), 1],
                c='r', marker='+', s=15, alpha=0.8)
    w_opt, auc, train_time = c_opt_auc(np.asarray(x_tr, dtype=np.float64),
                                       np.asarray(y_tr, dtype=np.float64), 2e-16)
    print('opt-auc', auc)
    fpr, tpr, _ = roc_curve(y_true=y_tr, y_score=np.dot(x_tr, w_opt))
    ax2.plot(fpr, tpr, label='AUC-opt : %.2f' % auc, color='tab:red')
    list_c, k_fold, best_auc, w_svm = np.logspace(-6, 4, 50), 5, -1., None
    for para_xi in list_c:
        tr_scores, te_scores, model = cmd_svm_perf(x_tr, y_tr, x_tr, y_tr, para_xi, 'linear')
        auc = roc_auc_score(y_true=y_tr, y_score=tr_scores)
        print(para_xi, auc)
        if best_auc < auc:
            best_auc = auc
            w_svm = [float(_.split(b':')[1]) for _ in model[-1].split(b' ')[1:3]]
    fpr, tpr, _ = roc_curve(y_true=y_tr, y_score=np.dot(x_tr, w_svm))
    ax2.plot(fpr, tpr, label='SVM-Perf : %.2f' % best_auc, color='tab:green')
    list_c, k_fold, best_auc, w_lr, c_lr = np.logspace(-10, 10, 50), 5, -1., None, None
    for para_xi in list_c:
        lr = LogisticRegression(
            penalty='l2', dual=False, tol=1e-5, C=para_xi, fit_intercept=True,
            intercept_scaling=1, class_weight='balanced', random_state=0,
            solver='liblinear', max_iter=10000, multi_class='auto', verbose=0,
            warm_start=False, n_jobs=None, l1_ratio=None)
        lr.fit(X=x_tr, y=y_tr)
        auc = roc_auc_score(y_true=y_tr, y_score=lr.decision_function(X=x_tr))
        if best_auc < auc:
            best_auc = auc
            w_lr = lr.coef_.flatten()
    fpr, tpr, _ = roc_curve(y_true=y_tr, y_score=np.dot(x_tr, w_lr))
    ax2.plot(fpr, tpr, label='LR : %.2f' % best_auc, color='tab:blue')
    ax2.plot([0, 1], [0, 1], label='Random : 0.50', color='gray', linestyle=':')
    print('lr', best_auc)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax2.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ppx = np.arange(x_min, x_max, h)
    xx, yy = [], []
    for _ in ppx:
        if y_min <= (-(w_opt[0] * _) / w_opt[1] - 1.) <= y_max:
            xx.append(_)
            yy.append(-(w_opt[0] * _) / w_opt[1] - 1.)
    ax1.plot(xx, yy, linestyle='-', zorder=-1, label='AUC-opt', color='tab:red')
    xx, yy = [], []
    for _ in ppx:
        if y_min <= -(w_svm[0] * _) / w_svm[1] <= y_max:
            xx.append(_)
            yy.append(-(w_svm[0] * _) / w_svm[1])
    ax1.plot(xx, yy, linestyle='--', zorder=-1, label='SVM-Perf', color='tab:green')
    xx, yy = [], []
    for _ in ppx:
        if y_min <= -(w_lr[0] * _) / w_lr[1] <= y_max:
            xx.append(_)
            yy.append(-(w_lr[0] * _) / w_lr[1])
    ax1.plot(xx, yy, linestyle='dotted', zorder=-1, label='LR', color='tab:blue')
    print('svm', best_auc)
    ax1.legend(loc='upper left', frameon=False, fontsize=12, handlelength=1.8)
    ax2.legend(loc='lower right', frameon=False, fontsize=12, handlelength=1.5)
    ax1.tick_params(axis='y', direction='in')
    ax1.tick_params(axis='x', direction='in')
    ax2.tick_params(axis='y', direction='in')
    ax2.tick_params(axis='x', direction='in')
    ax1.set_xticks([])
    ax1.set_yticks([])

    f_name = root_path + 'nips-2020-supp/figs/%s' % 'toy-example-samples.pdf'
    fig1.subplots_adjust(wspace=0.01, hspace=0.01)
    fig1.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()
    f_name = root_path + 'nips-2020-supp/figs/%s' % 'toy-example-roc.pdf'
    fig2.subplots_adjust(wspace=0.01, hspace=0.01)
    fig2.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()


def find_the_significance():
    list_datasets = [
        'abalone_19', 'abalone_7', 'arrhythmia_06', 'australian', 'banana', 'breast_cancer', 'cardio_3',
        'car_eval_34', 'car_eval_4', 'coil_2000', 'ecoli_imu', 'fourclass', 'german', 'ionosphere',
        'isolet', 'letter_a', 'letter_z', 'libras_move', 'mammography', 'mushrooms', 'oil',
        'optical_digits_0', 'optical_digits_8', 'ozone_level', 'page_blocks', 'pen_digits_0', 'pen_digits_5', 'pima',
        'satimage_4', 'scene', 'seismic', 'sick_euthyroid', 'solar_flare_m0', 'spambase', 'spectf',
        'spectrometer', 'splice', 'svmguide3', 'thyroid_sick', 'us_crime', 'vehicle_bus', 'vehicle_saab',
        'vehicle_van', 'vowel_hid', 'w7a', 'wine_quality', 'yeast_cyt', 'yeast_me1', 'yeast_me2', 'yeast_ml8']
    method_list = ['c_svm', 'b_c_svm', 'lr', 'b_lr', 'spauc', 'spam', 'svm_perf_lin', 'opt_auc']
    zzz = []
    significant_gaps_all = []
    insignificant_gaps_approx = []
    arr_fill_style = ['none'] * 50
    arr_color_style = ['white'] * 50
    for ind, dataset in enumerate(list_datasets):
        results = pkl.load(open(get_root_path() + '%s/results_all_%s.pkl' % (dataset, dataset), 'rb'))
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
        flag = True
        for method in ['spauc', 'spam', 'svm_perf_lin']:
            if method == 'opt_auc':
                continue
            t2 = [results[dataset]['tsne'][i][method]['tr']['auc'] for i in range(210)]
            stat, p_val = ttest_ind(a=t1, b=t2)
            if p_val > 0.05:
                flag = False
        if flag:
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
            arr_fill_style[ind] = 'top'
            arr_color_style[ind] = 'tab:green'
        else:
            insignificant_gaps_approx.append(best_gap)
        if flag and flag2:
            arr_fill_style[ind] = 'full'
            arr_color_style[ind] = 'tab:green'
        significant_gaps_all.append([best_gap, arr_fill_style[ind], arr_color_style[ind]])
        print(ind, index_i, best_gap, flag)
        zzz.append([ind, index_i, best_gap, flag])
    significant_gaps_all.sort(key=lambda x: x[0], reverse=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for ind, (gap, fill_style, color) in enumerate(significant_gaps_all):
        if ind == 0:
            ax.plot(ind, gap, fillstyle=fill_style, marker='s', markersize=5,
                    markeredgewidth=0.5, color=color, markeredgecolor='tab:green',
                    label='Significant to Both')
        elif ind == 6:
            ax.plot(ind, gap, fillstyle=fill_style, marker='s', markersize=5,
                    markeredgewidth=0.5, color=color, markeredgecolor='tab:green',
                    label='Significant to Approx. AUC Optimizers')
        elif ind == 15:
            ax.plot(ind, gap, fillstyle=fill_style, marker='s', markersize=5,
                    markeredgewidth=0.5, color=color, markeredgecolor='tab:green',
                    label='Significant to Standard Classifiers')
        elif ind == 19:
            ax.plot(ind, gap, fillstyle=fill_style, marker='s', markersize=5,
                    markeredgewidth=0.5, color=color, markeredgecolor='tab:green',
                    label='No Significance')
        else:
            ax.plot(ind, gap, fillstyle=fill_style, marker='s', markersize=5,
                    markeredgewidth=0.5, color=color, markeredgecolor='tab:green')
    ax.plot(sorted([_[0] for _ in significant_gaps_all])[::-1], color='tab:green', linewidth=.5)
    ax.set_ylabel('Gap')
    ax.set_xlabel('Dataset')
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='x', direction='in')
    leg = ax.legend(frameon=False)
    for line in leg.get_lines():
        line.set_linewidth(0.0)
    f_name = root_path + 'nips-2020-supp/figs/%s' % 'significant-tests-2.pdf'
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.savefig(f_name, dpi=300, bbox_inches='tight', pad_inches=.02, format='pdf')
    plt.close()


if __name__ == '__main__':
    toy_example()
