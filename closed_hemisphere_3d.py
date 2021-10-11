# -*- coding: utf-8 -*-
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import functools
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils.extmath import softmax
import multiprocessing
import pickle as pkl


def project_3d_2d(u, points: list):
    """
    Given a normal vector, project the points into this plane
    :param u:
    :param points: list of points
    :return:
    """
    u_norm = np.linalg.norm(u) ** 2.
    assert u_norm != 0.0  # normal vector should be nonzero.
    u1, u2, u3 = u[0], u[1], u[2]
    e11 = np.asarray([u2, -u1, 0])
    e22 = np.asarray([u1 * u3, u2 * u3, -(u1 ** 2. + u2 ** 2.)])
    e1 = e11 / np.linalg.norm(e11)
    e2 = e22 / np.linalg.norm(e22)
    points_2d = []
    for point in points:
        point_2d = [np.dot(e1, point), np.dot(e2, point)]
        points_2d.append(point_2d)
        print(np.linalg.norm(point_2d))
    return points_2d


def project_2d_3d(u, points_2d: list):
    u_norm = np.linalg.norm(u) ** 2.
    assert u_norm != 0.0  # normal vector should be nonzero.
    u1, u2, u3 = u[0], u[1], u[2]
    e11 = np.asarray([u2, -u1, 0])
    e22 = np.asarray([u1 * u3, u2 * u3, -(u1 ** 2. + u2 ** 2.)])
    e1 = e11 / np.linalg.norm(e11)
    e2 = e22 / np.linalg.norm(e22)
    points_3d = []
    for point_2d in points_2d:
        points_3d.append(point_2d[0] * e1 + point_2d[1] * e2)
    return points_3d


def sorting_rays(list_rays: list, u0, r0):
    def compare(r1, r2):
        u1, sign1, _ = r1
        p1 = (u1[1], -u1[0]) if sign1 == "nega" else (-u1[1], u1[0])
        u2, sign2, _ = r2
        p2 = (u2[1], -u2[0]) if sign2 == "nega" else (-u2[1], u2[0])
        dot_p1 = np.dot(p1, u0)
        dot_p2 = np.dot(p2, u0)
        if dot_p1 >= 0. and dot_p2 < 0:
            return -1
        elif dot_p1 < 0. and dot_p2 >= 0:
            return 1
        elif dot_p1 == 0. and dot_p2 == 0.:
            if r1[0][0] == r0[0][0] and r1[0][1] == r0[0][1] and r1[1] == r0[1]:
                return -1
            else:
                return 1
        else:
            if (r1[1] == "nega" and np.dot(p2, u1) > 0.) or \
                    (r1[1] == "posi" and np.dot(p2, u2) < 0.):
                return -1
            else:
                return 1

    list_rays.sort(key=functools.cmp_to_key(compare))
    return list_rays


def project_points(u, points):
    assert np.linalg.norm(u) != 0.0
    u_norm = np.linalg.norm(u) ** 2.
    c_prime = []
    for p in points:
        c_prime.append(p - (np.dot(p, u) / u_norm) * u)
    return c_prime


def check_projection(u, points, eps=1e-15):
    c_prime = project_points(u, points)  # project all points to plane
    for i in range(10):
        p = np.random.normal(loc=0.0, scale=2.0, size=3)
        p[2] = -(p[0] * u[0] + p[1] * u[1]) / u[2]
        # should be the same
        print(np.sum(np.array(np.dot(points, p)) >= eps, axis=0),
              np.sum(np.array(np.dot(c_prime, p)) >= eps, axis=0))
        assert np.sum(np.array(np.dot(points, p)) >= eps, axis=0) == \
               np.sum(np.array(np.dot(c_prime, p)) >= eps, axis=0)


def check_original_projected(points, u, cp):
    for i in range(1000):
        x = np.random.normal(0., 1., 3)
        x[2] = -(u[0] * x[0] + u[1] * x[1]) / u[2]
        val1 = np.sum(np.array(np.dot(points, x)) > 0., axis=0)
        val2 = np.sum(np.array(np.dot(cp, x)) > 0., axis=0)
        print(val1, val2)


def open_hemisphere_2d(points, sp, u, cp):
    assert len(points) == len(cp)
    points_2d = project_3d_2d(u, cp)
    rays = []
    for ind, _ in enumerate(points_2d):
        # ignore zeros
        if np.linalg.norm(_) <= 1e-15:
            continue
        rays.append([_, "nega", sp[ind]])
        rays.append([_, "posi", sp[ind]])
    u0 = rays[0][0]
    r0 = rays[0]
    rays = sorting_rays(list_rays=rays, u0=u0, r0=r0)
    p0 = rays[0][0]
    ap_val = np.sum(np.array(np.dot(points_2d, p0)) > 0., axis=0)
    max_a_val, opt_x = ap_val, np.asarray(p0)
    s2 = len(rays)
    ap_vals = [ap_val]
    for ii in range(s2):
        u1, sign1, sp1 = rays[ii]
        pi = (u1[1], -u1[0]) if sign1 == "nega" else (-u1[1], u1[0])
        u2, sign2, sp2 = rays[(ii + 1) % s2]
        pii = (u2[1], -u2[0]) if sign2 == "nega" else (-u2[1], u2[0])
        aq_val = ap_val + (1. if sign1 == "nega" and sp1 == 1 else 0.) - (1. if sign1 == "posi" and sp1 == 0 else 0.)
        if aq_val > max_a_val:
            max_a_val, opt_x = aq_val, np.asarray(pi) + np.asarray(pii)
        ap_val = aq_val + (1. if sign2 == "nega" and sp2 == 0 else 0.) - (1. if sign2 == "posi" and sp2 == 1 else 0.)
        ap_vals.append(ap_val)
        if ap_val > max_a_val:
            max_a_val, opt_x = aq_val, np.asarray(pii)
    import matplotlib.pyplot as plt
    plt.plot(ap_vals)
    # plt.plot(np.diff(ap_vals))
    plt.show()
    return opt_x, max_a_val, points_2d


def open_hemisphere_3d(points: list):
    eps = 1.e-15
    """
    each point in points set are unit vector in 3-dimension.
    :param points:
    :return:
    """
    for point in points:
        assert len(point) == 3
        assert np.abs(np.linalg.norm(point) - 1.) <= eps
    opt_val, opt_x = -1.0, np.zeros(3)
    # all nonzero points
    for u in points:
        # project all points to plane defined by its normal u
        # notice that c_prime may contain zero vectors.
        cp = project_points(u, points)
        sp = [1. for ind, _ in enumerate(points)]
        yu1, a1_val, points_2d = open_hemisphere_2d(points, sp, u, cp)
        if opt_val < a1_val:
            opt_x, opt_val = yu1, a1_val
            print(opt_val, opt_val / 2500., opt_x,
                  f"expected: {np.sum(np.array(np.dot(points_2d, yu1)) > 0., axis=0)}")
        sp = [1. if np.dot(_, u) <= 0.0 else 0. for _ in points]
        yu2, a2_val, points_2d = open_hemisphere_2d(points, sp, u, cp)
        if opt_val < a2_val:
            opt_x, opt_val = yu2, a2_val
            print(opt_val, opt_val / 2500., opt_x,
                  f"expected: {np.sum(np.array(np.dot(points_2d, yu2)) > 0., axis=0)}")
    return opt_x, opt_val


def test_logistic(x_tr, posi_indices, nega_indices):
    sub_tr_x = list(x_tr[posi_indices[:50], :])
    sub_tr_x.extend(list(x_tr[nega_indices[:50], :]))
    sub_tr_y = list(np.ones(50))
    sub_tr_y.extend(list(-np.ones(50)))
    lr = LogisticRegression(  # without ell_2 regularization.
        penalty='none', dual=False, tol=1e-8, C=1.0, fit_intercept=True,
        intercept_scaling=1, class_weight='balanced', solver='lbfgs', max_iter=10000, multi_class='ovr', verbose=0,
        warm_start=False, n_jobs=1, l1_ratio=None)
    lr.fit(X=np.asarray(sub_tr_x), y=sub_tr_y)
    print(roc_auc_score(y_true=sub_tr_y, y_score=lr.decision_function(X=np.asarray(sub_tr_x))))


def auc_opt_3d(x_tr, y_tr):
    x_tr = np.asarray(x_tr, dtype=np.float64)
    np.random.seed(17)
    assert 3 == x_tr.shape[1]
    posi_indices = [ind for ind, _ in enumerate(y_tr) if _ > 0.]
    nega_indices = [ind for ind, _ in enumerate(y_tr) if _ < 0.]
    set_k = []
    for i in posi_indices[:50]:
        for j in nega_indices[:50]:
            point = x_tr[i] - x_tr[j]
            if np.linalg.norm(point) <= 1e-15:
                # ignore the co-linear pair
                continue
            point = point / np.linalg.norm(point)
            set_k.append(point)
    w, ax = open_hemisphere_3d(points=set_k)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_tr[posi_indices, 0], x_tr[posi_indices, 1], x_tr[posi_indices, 2], 'r')
    ax.scatter3D(x_tr[nega_indices, 0], x_tr[nega_indices, 1], x_tr[nega_indices, 2], 'b')
    plt.show()


def main():
    data = pkl.load(open('t_sne_3d_pima.pkl', 'rb'))
    x_tr, y_tr = data[('original', 20)]['embeddings'], data[('original', 20)]['y_tr']
    auc_opt_3d(x_tr=x_tr, y_tr=y_tr)


main()
