# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import time
import numpy as np
from itertools import product
from sklearn.metrics import roc_auc_score


def get_x_tr_1(n, p):
    x_tr = np.random.normal(loc=0., scale=1., size=(n, p))
    y_tr = np.ones(n)
    y_tr[n // 2:] *= -1
    return x_tr, y_tr


def get_w_3d(x1, x2, x3):
    a = x1 - x2
    b = x1 - x3
    w = np.zeros(3)
    w[0] = a[1] * b[2] - a[2] * b[1]
    w[1] = a[2] * b[0] - a[0] * b[2]
    w[2] = a[0] * b[1] - a[1] * b[0]
    return w


def get_w_4d(x1, x2, x3, x4):
    a = x1 - x2
    b = x1 - x3
    c = x1 - x4
    mat = np.stack((a, b, c))
    w = np.zeros(4)
    w[0] = np.linalg.det(mat[:, [1, 2, 3]])
    w[1] = -np.linalg.det(mat[:, [0, 2, 3]])
    w[2] = np.linalg.det(mat[:, [0, 1, 3]])
    w[3] = -np.linalg.det(mat[:, [0, 1, 2]])
    return w


def get_x_tr_2():
    x_tr = np.random.normal(loc=0., scale=1., size=(4, 3))
    x1, x2, x3, x4 = x_tr[0], x_tr[1], x_tr[2], x_tr[3]
    w1 = get_w_3d(x1, x2, x3)
    w2 = get_w_3d(x1, x2, x4)
    w3 = get_w_3d(x1, x3, x4)
    w4 = get_w_3d(x2, x3, x4)
    y_tr = np.asarray([1., 1., -1., -1.])
    return x_tr, y_tr, w1, w2, w3, w4


def get_x_tr_3():
    n, p = 7, 3
    np.random.seed(1)
    x_tr = np.random.normal(loc=0., scale=1., size=(n, p))
    w_list = []
    eps1, eps2, eps3 = 1e-6, 1e-6, 1e-6
    for x1, x2, x3 in product(x_tr, x_tr, x_tr):
        w1 = np.array(get_w_3d(x1, x2, x3))
        w1[0] = w1[0] - eps1
        w1[1] = w1[1] + eps2
        w1[2] = w1[2] + eps3
        w2 = np.array(get_w_3d(x1, x2, x3))
        w2[0] = w2[0] + eps1
        w2[1] = w2[1] - eps2
        w2[2] = w2[2] + eps3
        w3 = np.array(get_w_3d(x1, x2, x3))
        w3[0] = w3[0] + eps1
        w3[1] = w3[1] + eps2
        w3[2] = w3[2] - eps3
        w4 = np.array(get_w_3d(x1, x2, x3))
        w4[0] = w4[0] + eps1
        w4[1] = w4[1] - eps2
        w4[2] = w4[2] - eps3
        w5 = np.array(get_w_3d(x1, x2, x3))
        w5[0] = w5[0] - eps1
        w5[1] = w5[1] + eps2
        w5[2] = w5[2] - eps3
        w6 = np.array(get_w_3d(x1, x2, x3))
        w6[0] = w6[0] - eps1
        w6[1] = w6[1] - eps2
        w6[2] = w6[2] + eps3
        w_list.append(w1)
        w_list.append(w2)
        w_list.append(w3)
        w_list.append(w4)
        w_list.append(w5)
        w_list.append(w6)
    y_tr = np.ones(n)
    y_tr[n // 2:] *= -1.
    list_indices = dict()
    for w in w_list:
        indices = np.argsort(np.dot(x_tr, w))
        list_indices['_'.join([str(_) for _ in indices])] = ''
        indices = np.argsort(np.dot(x_tr, -w))
        list_indices['_'.join([str(_) for _ in indices])] = ''
    print(len(list_indices))
    return x_tr, y_tr, w_list


def test_high_dimension_algo():
    np.random.seed(0)
    n, p = 6, 3
    timeout = time.time() + 6.
    all_indices = dict()
    x_tr, y_tr, w1_, w2_, w3_, w4_ = get_x_tr_2()
    eps = 1e-8
    list_indices = dict()
    for w in [w1_, w2_, w3_, w4_, -w1_, -w2_, -w3_, -w4_]:
        print('-----')
        w1 = np.array(w)
        w1[0] = w1[0] - eps
        w1[1] = w1[1] + eps
        w1[2] = w1[2] + eps
        print(w1, np.dot(x_tr, w1), np.argsort(np.dot(x_tr, w1)))
        indices = np.argsort(np.dot(x_tr, w1))
        list_indices['_'.join([str(_) for _ in indices])] = ''
        w2 = np.array(w)
        w2[0] = w2[0] + eps
        w2[1] = w2[1] - eps
        w2[2] = w2[2] + eps
        print(w2, np.dot(x_tr, w2), np.argsort(np.dot(x_tr, w2)))
        indices = np.argsort(np.dot(x_tr, w2))
        list_indices['_'.join([str(_) for _ in indices])] = ''
        w3 = np.array(w)
        w3[0] = w3[0] + eps
        w3[1] = w3[1] + eps
        w3[2] = w3[2] - eps
        print(w3, np.dot(x_tr, w3), np.argsort(np.dot(x_tr, w3)))
        indices = np.argsort(np.dot(x_tr, w3))
        list_indices['_'.join([str(_) for _ in indices])] = ''
        w4 = np.array(w)
        w4[0] = w4[0] + eps
        w4[1] = w4[1] - eps
        w4[2] = w4[2] - eps
        print(w4, np.dot(x_tr, w4), np.argsort(np.dot(x_tr, w4)))
        indices = np.argsort(np.dot(x_tr, w4))
        list_indices['_'.join([str(_) for _ in indices])] = ''
        w5 = np.array(w)
        w5[0] = w5[0] - eps
        w5[1] = w5[1] + eps
        w5[2] = w5[2] - eps
        print(w5, np.dot(x_tr, w5), np.argsort(np.dot(x_tr, w5)))
        indices = np.argsort(np.dot(x_tr, w5))
        list_indices['_'.join([str(_) for _ in indices])] = ''
        w6 = np.array(w)
        w6[0] = w6[0] - eps
        w6[1] = w6[1] - eps
        w6[2] = w6[2] + eps
        print(w6, np.dot(x_tr, w6), np.argsort(np.dot(x_tr, w6)))
        indices = np.argsort(np.dot(x_tr, w6))
        list_indices['_'.join([str(_) for _ in indices])] = ''
    for item in sorted(list_indices):
        print(item)
    print(len(list_indices))
    exit()
    while True:
        w = np.random.normal(loc=0., scale=1., size=p)
        scores = np.dot(x_tr, w)
        indices = np.argsort(scores)
        roc_auc_score(y_true=y_tr, y_score=scores)
        str_ = '_'.join([str(_) for _ in indices])
        all_indices[str_] = ''
        if time.time() > timeout:
            break
    print(len(all_indices))


def test_1():
    n, p = 7, 3
    np.random.seed(1)
    x_tr = np.random.normal(loc=0., scale=1., size=(n, p))
    y_tr = np.ones(n)
    y_tr[n // 2:] *= -1
    induces_dict = dict()
    w_list = []
    for i in range(1000000):
        w = np.random.normal(loc=0., scale=1., size=p)
        indices = np.argsort(np.dot(x_tr, w))
        str_ = '_'.join([str(_) for _ in indices])
        if str_ not in induces_dict:
            w_list.append(w)
            induces_dict[str_] = ''
    vals = []
    for w in sorted([list(_) for _ in w_list]):
        w = np.asarray(w)
        print(w, roc_auc_score(y_true=y_tr, y_score=np.dot(x_tr, w)))
        vals.append(roc_auc_score(y_true=y_tr, y_score=np.dot(x_tr, w)))
    plt.plot(sorted(vals))
    plt.show()


def test_2():
    x_tr, y_tr, w_list = get_x_tr_3()
    vals = []
    slopes = [np.dot(_, np.ones(3)) / (np.linalg.norm(_) * np.sqrt(3.)) for _ in w_list if np.dot(_, np.ones(3)) > 0.]
    w_list = np.asarray(w_list)
    for w in w_list[np.argsort(slopes)]:
        w = np.asarray(w)
        print(w, roc_auc_score(y_true=y_tr, y_score=np.dot(x_tr, w)))
        vals.append(roc_auc_score(y_true=y_tr, y_score=np.dot(x_tr, w)))
    plt.plot(vals)
    plt.show()


def good_case1(n, p, seed):
    np.random.seed(seed)
    x_tr = np.random.normal(loc=0., scale=1., size=(n, p))
    mid_points = []
    for (ind1, x1), (ind2, x2) in product(enumerate(x_tr), enumerate(x_tr)):
        if ind1 > ind2:
            mid_points.append((x1 + x2) / 2.)
    list_w = []
    for (ind1, x1), (ind2, x2), (ind3, x3) in product(enumerate(mid_points),
                                                      enumerate(mid_points), enumerate(mid_points)):
        if ind1 != ind2 and ind1 != ind3 and ind2 != ind3:
            list_w.append(get_w_3d(x1, x2, x3))
    induces_dict = dict()
    for w in list_w:
        indices = np.argsort(np.dot(x_tr, w))
        str_ = '_'.join([str(_) for _ in indices])
        if str_ not in induces_dict:
            induces_dict[str_] = ''
    print(seed, len(list_w), len(induces_dict))


def good_case2(n, p):
    np.random.seed(5)
    x_tr = np.random.normal(loc=0., scale=1., size=(n, p))
    mid_points = []
    for (ind1, x1), (ind2, x2), (ind3, x3) in product(enumerate(x_tr), enumerate(x_tr), enumerate(x_tr)):
        if ind1 > ind2 > ind3:
            mid_points.append((x1 + x2 + x3) / 3.)
    list_w = []
    for (ind1, x1), (ind2, x2), (ind3, x3) in product(enumerate(mid_points),
                                                      enumerate(mid_points), enumerate(mid_points)):
        if ind1 != ind2 and ind1 != ind3 and ind2 != ind3:
            list_w.append(get_w_3d(x1, x2, x3))
    print(len(list_w))
    induces_dict = dict()
    for w in list_w:
        indices = np.argsort(np.dot(x_tr, w))
        str_ = '_'.join([str(_) for _ in indices])
        if str_ not in induces_dict:
            induces_dict[str_] = ''
    print(len(induces_dict))


def good_case3(n, p, seed):
    np.random.seed(seed)
    x_tr = np.random.normal(loc=0., scale=1., size=(n, p))
    list_w = []
    for (ind1, x1), (ind2, x2), (ind3, x3) in product(enumerate(x_tr), enumerate(x_tr), enumerate(x_tr)):
        if ind1 != ind2 and ind1 != ind3 and ind2 != ind3:
            eps_plus = 1.01
            eps_minus = 0.99
            list_w.append(get_w_3d(x1 * eps_plus, x2 * eps_plus, x3 * eps_minus))
            list_w.append(get_w_3d(x1 * eps_plus, x2 * eps_minus, x3 * eps_plus))
            list_w.append(get_w_3d(x1 * eps_plus, x2 * eps_minus, x3 * eps_minus))
            list_w.append(get_w_3d(x1 * eps_minus, x2 * eps_plus, x3 * eps_plus))
            list_w.append(get_w_3d(x1 * eps_minus, x2 * eps_minus, x3 * eps_plus))
            list_w.append(get_w_3d(x1 * eps_minus, x2 * eps_plus, x3 * eps_minus))
    print(len(list_w))
    induces_dict = dict()
    for w in list_w:
        indices = np.argsort(np.dot(x_tr, w))
        str_ = '_'.join([str(_) for _ in indices])
        if str_ not in induces_dict:
            induces_dict[str_] = ''
    print(len(induces_dict))


def test_dim_4(n, p, seed):
    np.random.seed(seed)
    x_tr = np.random.normal(loc=0., scale=1., size=(n, p))
    mid_points = []
    for (ind1, x1), (ind2, x2) in product(enumerate(x_tr), enumerate(x_tr)):
        if ind1 > ind2:
            mid_points.append((x1 + x2) / 2.)
    list_w = []
    for (ind1, x1), (ind2, x2), (ind3, x3) in product(enumerate(mid_points),
                                                      enumerate(mid_points), enumerate(mid_points)):
        if ind1 != ind2 and ind1 != ind3 and ind2 != ind3:
            list_w.append(get_w_3d(x1, x2, x3))
    induces_dict = dict()
    for w in list_w:
        indices = np.argsort(np.dot(x_tr, w))
        str_ = '_'.join([str(_) for _ in indices])
        if str_ not in induces_dict:
            induces_dict[str_] = ''
    print(seed, len(list_w), len(induces_dict))


def calculate_number_of_hyperplanes(n=6):
    results = []
    initial_arr = [6, 22, 52, 100, 172]
    if n == 6:
        # print(initial_arr)
        return initial_arr[-1]

    for i in range(6, n):
        for j in range(1, 5):
            initial_arr[j] = initial_arr[j - 1] + initial_arr[j]
        results.append(initial_arr[-1])
    # print(initial_arr)
    return results


def test():
    import matplotlib.pyplot as plt

    plt.plot(calculate_number_of_hyperplanes(1000000), label='true')
    plt.plot([_ * _ * _ * _ * 0.26 for _ in range(6, 1000000)], label='predicted upper')
    plt.plot([_ * _ * _ * _ * 0.24 for _ in range(6, 1000000)], label='predicted lower')
    plt.plot([_ * _ * _ * _ * 0.25 for _ in range(6, 1000000)], label='predicted true')
    plt.legend()
    plt.show()
    exit()

    for i in range(50):
        good_case1(n=15, p=3, seed=i)
    exit()
    np.random.seed(0)
    x = np.random.normal(loc=0., scale=1., size=(10, 3))
    index = 1
    for i, j, k in product(range(10), range(10), range(10)):
        if i != j and j != k and i != k:
            print(index, i, j, k, get_w_3d(x[i], x[j], x[k]))
            index += 1
            # 0 1 2 [-2.66643526  2.10804758  0.9314267 ]
            # 0 2 1 [ 2.66643526 -2.10804758 -0.9314267 ]
            # 1 0 2 [ 2.66643526 -2.10804758 -0.9314267 ]
            # 1 2 0 [-2.66643526  2.10804758  0.9314267 ]
            # 2 0 1 [-2.66643526  2.10804758  0.9314267 ]
            # 2 1 0 [ 2.66643526 -2.10804758 -0.9314267 ]
            # [ 2.66643526 -2.10804758 -0.9314267 ]


def test_single(para):
    (seed, n), p = para, 4
    np.random.seed(seed)
    x_tr = np.random.normal(loc=0., scale=1., size=(n, p))
    mid_points = []
    for (ind1, x1), (ind2, x2) in product(enumerate(x_tr), enumerate(x_tr)):
        if ind1 > ind2:
            mid_points.append((x1 + x2) / 2.)
    induces_dict = dict()
    for (ind1, x1), (ind2, x2), (ind3, x3), (ind4, x4) in product(enumerate(mid_points),
                                                                  enumerate(mid_points),
                                                                  enumerate(mid_points),
                                                                  enumerate(mid_points)):
        if ind1 != ind2 and ind2 != ind3 and ind1 != ind3 and ind1 != ind4 and ind2 != ind4 and ind3 != ind4:
            w = get_w_4d(x1, x2, x3, x4)
            if np.sum(np.abs(w)) == 0.0:
                continue
            if len(np.unique(np.dot(x_tr, w))) != len(x_tr):
                continue
            indices = np.argsort(np.dot(x_tr, w))
            str_ = '_'.join([str(_) for _ in indices])
            if str_ not in induces_dict:
                induces_dict[str_] = ''
    print(n, len(induces_dict))
    return len(induces_dict)


if __name__ == '__main__':

    for n in range(5, 50):
        plt.plot([_ * (n - _) for _ in range(1, n)])
        plt.plot([n for _ in range(1, n)])
        plt.show()

    if False:
        n, p = 19, 3
        seed = 1
        np.random.seed(seed)
        x_tr = np.random.normal(loc=0., scale=1., size=(n, p))
        mid_points = []
        for (ind1, x1), (ind2, x2) in product(enumerate(x_tr), enumerate(x_tr)):
            if ind1 > ind2:
                mid_points.append((x1 + x2) / 2.)
        induces_dict = dict()
        for (ind1, x1), (ind2, x2), (ind3, x3) in product(enumerate(mid_points),
                                                          enumerate(mid_points),
                                                          enumerate(mid_points)):
            if ind1 != ind2 and ind2 != ind3 and ind1 != ind3:
                w = get_w_3d(x1, x2, x3)
                if np.sum(np.abs(w)) == 0.0:
                    continue
                if len(np.unique(np.dot(x_tr, w))) != len(x_tr):
                    continue
                indices = np.argsort(np.dot(x_tr, w))
                str_ = '_'.join([str(_) for _ in indices])
                if str_ not in induces_dict:
                    induces_dict[str_] = ''
        print(n, len(induces_dict))
        exit()
    import matplotlib.pyplot as plt

    plt.plot(calculate_number_of_hyperplanes(50), label='true')
    plt.plot([_ * _ * _ * _ * 0.25 for _ in range(6, 50)], label='n^4 / 4')
    plt.plot([_ * _ * _ * 6 for _ in range(6, 50)], label='n^3 * d!')
    plt.plot([_ * _ * _ * 8 for _ in range(6, 50)], label='n^3 * 2^d')
    plt.legend()
    plt.show()
    exit()
    import sys
    import multiprocessing

    pool = multiprocessing.Pool(processes=int(sys.argv[1]))
    results_pool = pool.map(test_single, [(_, int(sys.argv[2])) for _ in range(500)])
    print('maximal: ', max(results_pool))
    pool.close()
    pool.join()
