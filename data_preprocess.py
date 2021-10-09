# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

if os.uname()[1] == 'baojian-ThinkPad-T540p':
    root_path = '/data/auc-logistic/'
elif os.uname()[1] == 'pascal':
    root_path = '/mnt/store2/baojian/data/auc-logistic/'
elif os.uname()[1].endswith('.rit.albany.edu'):
    root_path = '/network/rit/lab/ceashpc/bz383376/data/auc-logistic/'


def _get_data(x_tr, y_tr, data_id, data_name, num_trials, split_ratio=5. / 6.):
    np.random.seed(17)
    # 1. data standardization.
    """
    https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    sklearn.preprocessing.StandardScaler
    """
    x_tr = np.asarray(x_tr, dtype=float)
    y_tr = np.asarray(y_tr, dtype=float)
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': len(x_tr[0]),
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            # There is a data leaking, but in general, it will be okay.
            'x_tr': StandardScaler().fit_transform(x_tr),
            'y_tr': y_tr}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('# of classes: %d' % len(np.unique(y_tr)))
    print('# of samples: %d' % data['n'])
    print('# of positive: %d' % data['num_posi'])
    print('# of negative: %d' % data['num_nega'])
    print('# of features: %d' % data['p'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    # generate different trials of data
    for _ in range(num_trials):
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
    return data


def data_preprocess_australian(num_trials):
    """
    Source: http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
    # of classes: 2
    # of samples: 690
    # of positive: 307
    # of negative: 383
    # of features: 14
    :return:
    """
    x_tr, y_tr, features, p = [], [], dict(), 14
    data_name, data_id = 'australian', 4
    f_path = '%02d_%s/raw_%s' % (data_id, data_name, data_name)
    with open(os.path.join(root_path, f_path)) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            values = np.asarray([float(_) for _ in items[:-1]], dtype=float)
            y_tr.append(1 if int(items[-1]) == 1 else -1)
            x_tr.append(values)
    _get_data(x_tr=x_tr, y_tr=y_tr, data_id=data_id,
              data_name=data_name, num_trials=num_trials, split_ratio=5. / 6.)


def data_preprocess_breast_cancer(num_trials):
    """
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#breast-cancer
    # of classes: 2
    # of data: 683
    # of features: 10
    Note: 2 for benign, 4 for malignant
    :return:
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 10
    data_name, data_id = 'breast_cancer', 6
    f_path = '%02d_%s/raw_%s' % (data_id, data_name, data_name)
    with open(os.path.join(root_path, f_path)) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            items = [_ for _ in items if len(_) != 0]
            values = [float(_.split(':')[1]) for _ in items[1:]]
            # 2 for benign, 4 for malignant
            y_tr.append(1 if int(float(items[0])) == 4 else -1)
            x_tr.append(np.asarray(values, dtype=float))
    _get_data(x_tr=x_tr, y_tr=y_tr, data_id=data_id,
              data_name=data_name, num_trials=num_trials, split_ratio=5. / 6.)


def data_preprocess_splice(num_trials):
    """
    https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)
    # of classes: 2
    # of data: 3,175 (testing)
    # of features: 60
    # of posi: 1648
    # of nega: 1527
    :return:
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 60
    data_name, data_id = 'splice', 7
    f_path = '%02d_%s/raw_%s' % (data_id, data_name, data_name)
    with open(os.path.join(root_path, f_path)) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            indices = np.asarray([int(_.split(':')[0]) - 1 for _ in items[1:]], dtype=int)
            values = np.asarray([float(_.split(':')[1]) for _ in items[1:]], dtype=float)
            y_tr.append(int(items[0]))
            sample = np.zeros(p)
            sample[indices] = values
            x_tr.append(sample)
    _get_data(x_tr=x_tr, y_tr=y_tr, data_id=data_id,
              data_name=data_name, num_trials=num_trials, split_ratio=5. / 6.)


def data_preprocess_ijcnn1(num_trials):
    """
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    # of classes: 2
    # of samples: 141691
    # of positive: 13565
    # of negative: 128126
    # of features: 22
    posi_ratio: 0.09574
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 22
    data_name, data_id = 'ijcnn1', 8
    f_path = '%02d_%s/raw_%s' % (data_id, data_name, data_name)
    with open(os.path.join(root_path, f_path)) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            indices = np.asarray([int(_.split(':')[0]) - 1 for _ in items[1:]], dtype=int)
            values = np.asarray([float(_.split(':')[1]) for _ in items[1:]], dtype=float)
            y_tr.append(int(items[0]))
            sample = np.zeros(p)
            sample[indices] = values
            x_tr.append(sample)
    _get_data(x_tr=x_tr, y_tr=y_tr, data_id=data_id,
              data_name=data_name, num_trials=num_trials, split_ratio=5. / 6.)


def data_preprocess_mushrooms(num_trials):
    """
    # of classes: 2
    # of samples: 8124
    # of positive: 3916
    # of negative: 4208
    # of features: 112
    posi_ratio: 0.48203
    :return:
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 112
    data_name, data_id = 'mushrooms', 9
    f_path = '%02d_%s/raw_%s' % (data_id, data_name, data_name)
    with open(os.path.join(root_path, f_path)) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            indices = np.asarray([int(_.split(':')[0]) - 1 for _ in items[1:]], dtype=int)
            values = np.asarray([float(_.split(':')[1]) for _ in items[1:]], dtype=float)
            y_tr.append(1 if int(items[0]) == 1 else -1)
            sample = np.zeros(p)
            sample[indices] = values
            x_tr.append(sample)
    _get_data(x_tr=x_tr, y_tr=y_tr, data_id=data_id,
              data_name=data_name, num_trials=num_trials, split_ratio=5. / 6.)


def data_preprocess_svmguide3(num_trials):
    """
    # of classes: 2
    # of data: 1,243 / 41 (testing)
    # of features: 20
    Files:
        svmguide3
        svmguide3.t (testing)
    :return:
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 20
    data_name, data_id = 'svmguide3', 10
    f_path = '%02d_%s/raw_%s' % (data_id, data_name, data_name)
    with open(os.path.join(root_path, f_path)) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            items = items[:-2]  # the last two are all zeros.
            indices = np.asarray([int(_.split(':')[0]) - 1 for _ in items[1:]], dtype=int)
            values = np.asarray([float(_.split(':')[1]) for _ in items[1:]], dtype=float)
            y_tr.append(1 if int(items[0]) == 1 else -1)
            sample = np.zeros(p)
            sample[indices] = values
            x_tr.append(sample)
    data = {'name': 'svmguide3',
            'data_path': os.path.join(root_path, '10_svmguide3'),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %03d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_te)

        if not os.path.exists(root_path + '10_svmguide3/svmguide3_tr_%03d.dat'):
            f_tr = open(root_path + '10_svmguide3/svmguide3_tr_%03d.dat' % _, 'w')
            f_te = open(root_path + '10_svmguide3/svmguide3_te_%03d.dat' % _, 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    f_tr.write(r'2 qid:1 ' + r' '.join([r'%d:%.f' % (ind, _)
                                                        for ind, _ in zip(range(1, len(values) + 1), values)]))
                else:
                    f_tr.write(r'1 qid:1 ' + r' '.join([r'%d:%.f' % (ind, _)
                                                        for ind, _ in zip(range(1, len(values) + 1), values)]))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    f_te.write(r'2 qid:1 ' + r' '.join([r'%d:%.f' % (ind, _)
                                                        for ind, _ in zip(range(1, len(values) + 1), values)]))
                else:
                    f_te.write(r'1 qid:1 ' + r' '.join([r'%d:%.f' % (ind, _)
                                                        for ind, _ in zip(range(1, len(values) + 1), values)]))
                f_te.write('\n')
            f_te.close()

    return data


def data_preprocess_german(num_trials):
    """
    # of classes: 2
    # of data: 1,000
    # of features: 24
    Files:
        german.numer
        german.numer_scale (scaled to [-1,1])
    :return:
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 24
    with open(os.path.join(root_path, '11_german/german.numer_scale.txt')) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            indices = np.asarray([int(_.split(':')[0]) - 1 for _ in items[1:]], dtype=int)
            values = np.asarray([float(_.split(':')[1]) for _ in items[1:]], dtype=float)
            values /= np.linalg.norm(values)
            y_tr.append(1 if int(items[0]) == 1 else -1)
            sample = np.zeros(p)
            sample[indices] = values
            x_tr.append(sample)
    data = {'name': 'german',
            'data_path': os.path.join(root_path, '11_german/'),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %03d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.3f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_te)

        if not os.path.exists(root_path + '11_german/german_tr_%03d.dat'):
            f_tr = open(root_path + '11_german/german_tr_%03d.dat' % _, 'w')
            f_te = open(root_path + '11_german/german_te_%03d.dat' % _, 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    f_tr.write(r'2 qid:1 ' + r' '.join([r'%d:%.f' % (ind, _)
                                                        for ind, _ in zip(range(1, len(values) + 1), values)]))
                else:
                    f_tr.write(r'1 qid:1 ' + r' '.join([r'%d:%.f' % (ind, _)
                                                        for ind, _ in zip(range(1, len(values) + 1), values)]))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    f_te.write(r'2 qid:1 ' + r' '.join([r'%d:%.f' % (ind, _)
                                                        for ind, _ in zip(range(1, len(values) + 1), values)]))
                else:
                    f_te.write(r'1 qid:1 ' + r' '.join([r'%d:%.f' % (ind, _)
                                                        for ind, _ in zip(range(1, len(values) + 1), values)]))
                f_te.write('\n')
            f_te.close()
    return data


def data_preprocess_fourclass(num_trials):
    """
    # of classes: 2
    # of data: 862
    # of features: 2
    Files:
        fourclass
        fourclass_scale (scaled to [-1,1])
    :return:
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 2
    with open(os.path.join(root_path, '12_fourclass/fourclass_scale.txt')) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            indices = np.asarray([int(_.split(':')[0]) - 1 for _ in items[1:]], dtype=int)
            values = np.asarray([float(_.split(':')[1]) for _ in items[1:]], dtype=float)
            values /= np.linalg.norm(values)
            y_tr.append(1 if int(items[0]) == 1 else -1)
            sample = np.zeros(p)
            sample[indices] = values
            x_tr.append(sample)
    data = {'name': 'fourclass',
            'data_path': os.path.join(root_path, '12_fourclass/'),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %03d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_a9a(num_trials):
    """
    # of classes: 2
    # of data: 32,561 / 16,281 (testing)
    # of features: 123 / 123 (testing)
    Files:
        a9a
        a9a.t (testing)
    :return:
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 123
    with open(os.path.join(root_path, '13_a9a/raw_a9a')) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            indices = np.asarray([int(_.split(':')[0]) - 1 for _ in items[1:]], dtype=int)
            values = np.asarray([float(_.split(':')[1]) for _ in items[1:]], dtype=float)
            values /= np.linalg.norm(values)
            y_tr.append(1 if int(items[0]) == 1 else -1)
            sample = np.zeros(p)
            sample[indices] = values
            x_tr.append(sample)
    data = {'name': 'a9a',
            'data_path': os.path.join(root_path, '13_a9a/'),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    xx = np.sum(data['x_tr'], axis=0)
    print(np.count_nonzero(data['x_tr'][:, -1]))
    print(xx)

    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %03d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_w8a(num_trials):
    """
    # of classes: 2
    # of data: 49,749 / 14,951 (testing)
    # of features: 300 / 300 (testing)
    Files:
        w8a
        w8a.t (testing)
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 300
    with open(os.path.join(root_path, '14_w8a/raw_w8a')) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            if len(items) == 1:  # we remove the zero vectors.
                continue
            indices = np.asarray([int(_.split(':')[0]) - 1 for _ in items[1:]], dtype=int)
            values = np.asarray([float(_.split(':')[1]) for _ in items[1:]], dtype=float)
            values /= np.linalg.norm(values)
            y_tr.append(1 if int(items[0]) == 1 else -1)
            sample = np.zeros(p)
            sample[indices] = values
            x_tr.append(sample)
    data = {'name': 'w8a',
            'data_path': os.path.join(root_path, '14_w8a/'),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %03d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_covtype(num_trials):
    """
    # of classes: 2
    # of data: 581,012
    # of features: 54
    Files:
        covtype.libsvm.binary.bz2
        covtype.libsvm.binary.scale.bz2 (scaled to [0,1])
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 54
    with open(os.path.join(root_path, '15_covtype/covtype.libsvm.binary.scale')) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            indices = np.asarray([int(_.split(':')[0]) - 1 for _ in items[1:]], dtype=int)
            values = np.asarray([float(_.split(':')[1]) for _ in items[1:]], dtype=float)
            values /= np.linalg.norm(values)
            y_tr.append(1 if int(items[0]) == 1 else -1)
            sample = np.zeros(p)
            sample[indices] = values
            x_tr.append(sample)
    data = {'name': 'covtype',
            'data_path': os.path.join(root_path, '15_covtype/'),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %03d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.3f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_spambase(num_trials):
    """
    # of classes: 2
    # of data: 4,601
    # of features: 57
    Files:
        spambase.
    """
    np.random.seed(17)
    x_tr, y_tr, features, p = [], [], dict(), 57
    with open(os.path.join(root_path, '16_spambase/spambase.data')) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(',')
            values = np.asarray([float(_) for _ in items[:-1]], dtype=float)
            values /= np.linalg.norm(values)
            y_tr.append(1 if int(items[-1]) == 1 else -1)
            sample = values
            x_tr.append(sample)
    data = {'name': 'spambase',
            'data_path': os.path.join(root_path, '16_spambase/'),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %03d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_yeast(num_trials):
    """
    https://archive.ics.uci.edu/ml/datasets/Yeast
    # of classes: 10 classes ['MIT', 'NUC', 'CYT', 'ME1', 'EXC', 'ME2', 'ME3', 'VAC', 'POX', 'ERL']
    # of data: 1,484
    # of features: 8 (Numerical)
    # select ['ME1'] as the positive class
    """
    np.random.seed(17)
    x_tr, y_tr, features, p, data_id, data_name = [], [], dict(), 8, 17, 'yeast'
    with open(os.path.join(root_path, '%02d_%s/raw_%s' % (data_id, data_name, data_name))) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            items = [_.lstrip().rstrip() for _ in items if len(_) != 0]
            values = np.asarray([float(_) for _ in items[1:-1]], dtype=float)
            x_tr.append(values / np.linalg.norm(values))  # CYT ME1 ME2- ME3 NUC
            y_tr.append(1 if items[-1] == 'ME1' else -1)
            features[items[-1]] = ''
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %05d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        if not os.path.exists(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _)):
            f_tr = open(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            f_te = open(root_path + '%02d_%s/%s_te_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'1 qid:1 ' + r' '.join(items))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'1 qid:1 ' + r' '.join(items))
                f_te.write('\n')
            f_te.close()
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_pima(num_trials):
    """
    https://archive.ics.uci.edu/ml/datasets/diabetes
    # of classes: 2
    # of data: 768
    # of features: 8
    """
    x_tr, y_tr, features, p, data_id, data_name = [], [], dict(), 8, 18, 'pima'
    with open(os.path.join(root_path, '%02d_%s/raw_%s' % (data_id, data_name, data_name))) as f:
        for each_line in f.readlines():
            if each_line.startswith('Preg'):
                continue
            items = each_line.lstrip().rstrip().split(',')
            items = [_.lstrip().rstrip() for _ in items if len(_) != 0]
            values = np.asarray([float(_) for _ in items[:-1]], dtype=float)
            x_tr.append(values / np.linalg.norm(values))
            y_tr.append(-1 if items[-1] == '0' else 1)
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %05d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        if not os.path.exists(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _)):
            f_tr = open(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            f_te = open(root_path + '%02d_%s/%s_te_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'1 qid:1 ' + r' '.join(items))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'1 qid:1 ' + r' '.join(items))
                f_te.write('\n')
            f_te.close()
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_page_blocks(num_trials):
    """
    https://archive.ics.uci.edu/ml/machine-learning-databases/page-blocks/
    # of classes: 5 [1,2,3,4,5]
    # of data: 5,473
    # of features: 10 (Numerical)
    # select [2,3,4,5] as the positive class
    """
    np.random.seed(17)
    x_tr, y_tr, features, p, data_id, data_name = [], [], dict(), 10, 19, 'page_blocks'
    with open(os.path.join(root_path, '%02d_%s/raw_%s' % (data_id, data_name, data_name))) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            items = [_.lstrip().rstrip() for _ in items if len(_) != 0]
            values = np.asarray([float(_) for _ in items[:-1]], dtype=float)
            x_tr.append(values / np.linalg.norm(values))
            y_tr.append(-1 if items[-1] == '1' else 1)
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %05d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        if not os.path.exists(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _)):
            f_tr = open(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            f_te = open(root_path + '%02d_%s/%s_te_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'1 qid:1 ' + r' '.join(items))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'1 qid:1 ' + r' '.join(items))
                f_te.write('\n')
            f_te.close()
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_pendigits(num_trials):
    """
    https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
    # of classes: 10 [0,1,2,...,9]
    # of data: 10,992
    # of features: 16
    # select [0] as the positive class
    """
    np.random.seed(17)
    x_tr, y_tr, features, p, data_id, data_name = [], [], dict(), 16, 20, 'pendigits'
    with open(os.path.join(root_path, '%02d_%s/raw_%s' % (data_id, data_name, data_name))) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(',')
            items = [_.lstrip().rstrip() for _ in items if len(_) != 0]
            values = np.asarray([float(_) for _ in items[:-1]], dtype=float)
            x_tr.append(values / np.linalg.norm(values))
            y_tr.append(1 if items[-1] == '0' else -1)
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %05d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        if not os.path.exists(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _)):
            f_tr = open(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            f_te = open(root_path + '%02d_%s/%s_te_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'1 qid:1 ' + r' '.join(items))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'1 qid:1 ' + r' '.join(items))
                f_te.write('\n')
            f_te.close()
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_optdigits(num_trials):
    """
    https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
    # of classes: 10 [0,1,2,...,9]
    # of data: 5,620
    # of features: 64 (Numerical)
    # select [0] as the positive class
    """
    np.random.seed(17)
    x_tr, y_tr, features, p, data_id, data_name = [], [], dict(), 64, 21, 'optdigits'
    with open(os.path.join(root_path, '%02d_%s/raw_%s' % (data_id, data_name, data_name))) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(',')
            items = [_.lstrip().rstrip() for _ in items if len(_) != 0]
            values = np.asarray([float(_) for _ in items[:-1]], dtype=float)
            x_tr.append(values / np.linalg.norm(values))
            y_tr.append(1 if items[-1] == '0' else -1)
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %05d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        if not os.path.exists(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _)):
            f_tr = open(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            f_te = open(root_path + '%02d_%s/%s_te_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'1 qid:1 ' + r' '.join(items))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'1 qid:1 ' + r' '.join(items))
                f_te.write('\n')
            f_te.close()
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_letter(num_trials):
    """
    https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    # of classes: 26 [A,B,C,...,Z]
    # of data: 20,000
    # of features: 16 (Numerical)
    # select [A] as the positive class
    """
    np.random.seed(17)
    x_tr, y_tr, features, p, data_id, data_name = [], [], dict(), 16, 22, 'letter'
    with open(os.path.join(root_path, '%02d_%s/raw_%s' % (data_id, data_name, data_name))) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(',')
            items = [_.lstrip().rstrip() for _ in items if len(_) != 0]
            values = np.asarray([float(_) for _ in items[1:]], dtype=float)
            x_tr.append(values / np.linalg.norm(values))
            y_tr.append(1 if items[0] == 'A' else -1)
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %05d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        if not os.path.exists(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _)):
            f_tr = open(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            f_te = open(root_path + '%02d_%s/%s_te_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'1 qid:1 ' + r' '.join(items))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'1 qid:1 ' + r' '.join(items))
                f_te.write('\n')
            f_te.close()
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_tic_tac(num_trials):
    """
    https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    # of classes: 2 [A,B,C,...,Z]
    # of data: 20,000
    # of features: 16 (Numerical)
    # select [A] as the positive class
    """
    np.random.seed(17)
    x_tr, y_tr, features, p, data_id, data_name = [], [], dict(), 16, 22, 'letter'
    with open(os.path.join(root_path, '%02d_%s/raw_%s' % (data_id, data_name, data_name))) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(',')
            items = [_.lstrip().rstrip() for _ in items if len(_) != 0]
            values = np.asarray([float(_) for _ in items[1:]], dtype=float)
            x_tr.append(values / np.linalg.norm(values))
            y_tr.append(1 if items[0] == 'A' else -1)
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %05d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        if not os.path.exists(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _)):
            f_tr = open(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            f_te = open(root_path + '%02d_%s/%s_te_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'1 qid:1 ' + r' '.join(items))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'1 qid:1 ' + r' '.join(items))
                f_te.write('\n')
            f_te.close()
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_yeast_cyt(num_trials):
    """
    https://archive.ics.uci.edu/ml/datasets/Yeast
    # of classes: 10 classes ['MIT', 'NUC', 'CYT', 'ME1', 'EXC', 'ME2', 'ME3', 'VAC', 'POX', 'ERL']
    # of data: 1,484
    # of features: 8 (Numerical)
    # select ['CYT'] as the positive class
    """
    np.random.seed(17)
    x_tr, y_tr, features, p, data_id, data_name = [], [], dict(), 8, 24, 'yeast_cyt'
    with open(os.path.join(root_path, '%02d_%s/raw_%s' % (data_id, data_name, data_name))) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(' ')
            items = [_.lstrip().rstrip() for _ in items if len(_) != 0]
            values = np.asarray([float(_) for _ in items[1:-1]], dtype=float)
            x_tr.append(values / np.linalg.norm(values))  # CYT ME1 ME2- ME3 NUC
            y_tr.append(1 if items[-1] == 'CYT' else -1)
            features[items[-1]] = ''
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %05d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        if not os.path.exists(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _)):
            f_tr = open(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            f_te = open(root_path + '%02d_%s/%s_te_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'1 qid:1 ' + r' '.join(items))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'1 qid:1 ' + r' '.join(items))
                f_te.write('\n')
            f_te.close()
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_spectf(num_trials):
    """
    https://archive.ics.uci.edu/ml/datasets/Yeast
    # of classes: 2 classes
    # of data: 266
    # of features: 44 (Numerical)
    # select ['CYT'] as the positive class
    """
    np.random.seed(17)
    x_tr, y_tr, features, p, data_id, data_name = [], [], dict(), 44, 25, 'spectf'
    with open(os.path.join(root_path, '%02d_%s/raw_%s' % (data_id, data_name, data_name))) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(',')
            items = [_.lstrip().rstrip() for _ in items if len(_) != 0]
            values = [float(_) for _ in items[1:]]
            x_tr.append(values / np.linalg.norm(values))
            y_tr.append(1 if items[0] == '0' else -1)
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %05d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        if not os.path.exists(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _)):
            f_tr = open(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            f_te = open(root_path + '%02d_%s/%s_te_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'1 qid:1 ' + r' '.join(items))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'1 qid:1 ' + r' '.join(items))
                f_te.write('\n')
            f_te.close()
        assert data['n'] == (n_tr + n_te)
    return data


def data_preprocess_ionosphere(num_trials):
    """
    https://archive.ics.uci.edu/ml/datasets/ionosphere
    # of classes: 2 classes
    # of data: 351
    # of features: 34 (Numerical)
    # select 1 as the positive class
    """
    np.random.seed(17)
    x_tr, y_tr, features, p, data_id, data_name = [], [], dict(), 34, 2, 'ionosphere'
    with open(os.path.join(root_path, '%02d_%s/raw_%s' % (data_id, data_name, data_name))) as f:
        for each_line in f.readlines():
            items = each_line.lstrip().rstrip().split(',')
            items = [_.lstrip().rstrip() for _ in items if len(_) != 0]
            values = [float(_) for _ in items[:-1]]
            x_tr.append(values / np.linalg.norm(values))
            y_tr.append(1 if items[-1] == 'b' else -1)
    data = {'name': data_name,
            'data_path': os.path.join(root_path, '%02d_%s/' % (data_id, data_name)),
            'p': p,
            'n': len(y_tr),
            'num_trials': num_trials,
            'num_posi': len([_ for _ in y_tr if _ > 0]),
            'num_nega': len([_ for _ in y_tr if _ < 0]),
            'x_tr': np.asarray(x_tr, dtype=float),
            'y_tr': np.asarray(y_tr, dtype=float)}
    data['posi_ratio'] = float(data['num_posi']) / float(data['n'])
    print('n: %05d' % data['n'])
    print('p: %03d' % data['p'])
    print('num_posi: %03d' % data['num_posi'])
    print('num_nega: %03d' % data['num_nega'])
    print('posi_ratio: %.5f' % data['posi_ratio'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 5. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        if not os.path.exists(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _)):
            f_tr = open(root_path + '%02d_%s/%s_tr_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            f_te = open(root_path + '%02d_%s/%s_te_%03d.dat' % (data_id, data_name, data_name, _), 'w')
            for index in tr_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_tr.write(r'1 qid:1 ' + r' '.join(items))
                f_tr.write('\n')
            f_tr.close()
            for index in te_indices:
                values = data['x_tr'][index]
                if data['y_tr'][index] == 1:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'2 qid:1 ' + r' '.join(items))
                else:
                    items = [r'%d:%.6f' % (ind, _) for ind, _ in zip(range(1, len(values) + 1), values)]
                    f_te.write(r'1 qid:1 ' + r' '.join(items))
                f_te.write('\n')
            f_te.close()
        assert data['n'] == (n_tr + n_te)
    return data


def main():
    data_preprocess_mushrooms(num_trials=100)
    exit()
    data_preprocess_ijcnn1(num_trials=100)
    data_preprocess_splice(num_trials=100)
    data_preprocess_breast_cancer(num_trials=100)
    data_preprocess_australian(num_trials=100)
    data_preprocess_ionosphere(num_trials=200)
    data_preprocess_pima(num_trials=200)
    data_preprocess_yeast_cyt(num_trials=200)
    data_preprocess_tic_tac(num_trials=200)
    data_preprocess_letter(num_trials=200)
    data_preprocess_optdigits(num_trials=200)
    data_preprocess_page_blocks(num_trials=200)
    data_preprocess_pendigits(num_trials=200)
    data_preprocess_page_blocks(num_trials=200)
    data_preprocess_yeast(num_trials=200)
    data_preprocess_spambase(num_trials=200)
    data_preprocess_covtype(num_trials=200)
    data_preprocess_w8a(num_trials=200)
    data_preprocess_a9a(num_trials=200)
    data_preprocess_fourclass(num_trials=200)
    data_preprocess_german(num_trials=200)
    data_preprocess_svmguide3(num_trials=200)
    data_preprocess_australian(num_trials=200)
    data_preprocess_breast_cancer(num_trials=200)
    data_preprocess_splice(num_trials=200)
    data_preprocess_ijcnn1(num_trials=200)
    data_preprocess_mushrooms(num_trials=200)


if __name__ == '__main__':
    main()
