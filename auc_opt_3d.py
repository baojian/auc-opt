# -*- coding: utf-8 -*-
import functools
import numpy as np
from numpy.linalg import norm


def change_coordinates(normal, points):
    """
    Change coordinates from 3d to 2d
    :param normal: the normal vector
    :param points: the points on this plane
    :return:
    """
    assert norm(normal) != 0.
    u = np.asarray([normal[1], -normal[0], 0.])
    u = u / norm(u)
    v = np.cross(u, normal)
    v = v / norm(v)
    coordinates = np.asarray([u, v], dtype=np.float64).T
    mapped_points = np.dot(points, coordinates)
    return mapped_points


def change_coordinates_back(normal, w):
    """
    Change coordinates from 2d to 3d
    :param normal: the normal vector
    :param w: a 2d point
    :return:
    """
    # project 2d points back to 3d.
    uu = np.asarray([normal[1], -normal[0], 0.])
    uu = uu / norm(uu)
    vv = np.cross(uu, normal)
    vv = vv / norm(vv)
    w_3d = w[0] * uu + w[1] * vv
    return w_3d


def sorting_rays(list_rays: list, u0, r0):
    def compare(r1, r2):
        u1, sign1, _, p1, dot_p1 = r1
        u2, sign2, _, p2, dot_p2 = r2
        if dot_p1 >= 0. and dot_p2 < 0:
            return -1
        elif dot_p1 < 0. and dot_p2 >= 0:
            return 1
        elif dot_p1 == 0. and dot_p2 == 0.:
            if r1[0][0] == u0[0] and r1[0][1] == u0[1] and r1[1] == r0[1]:
                return -1
            else:
                return 1
        else:
            pu = np.dot(p2, u1)
            if (r1[1] == 0 and pu > 0.) or (r1[1] == 1 and pu < 0.):
                return -1
            else:
                return 1

    list_rays.sort(key=functools.cmp_to_key(compare))
    return list_rays


def auc_opt_2d(points, sp, u, cp, precision_eps=1e-15):
    points_2d = change_coordinates(normal=u, points=cp)
    rays = []
    u0 = points_2d[0]
    # 0 for negative and 1 for positive
    r0 = [u0, 0, sp[0], (u0[1], -u0[0]), np.dot((u0[1], -u0[0]), u0)]
    for ind, _ in enumerate(points_2d):
        if norm(_) <= precision_eps:  # ignore zeros
            continue
        rays.append([_, 0, sp[ind], (_[1], -_[0]), np.dot((_[1], -_[0]), u0)])
        rays.append([_, 1, sp[ind], (-_[1], _[0]), np.dot((-_[1], _[0]), u0)])
    rays = sorting_rays(list_rays=rays, u0=u0, r0=r0)
    p0 = rays[0][0]
    ap_val = np.sum(np.array(np.dot(points_2d, p0)) > 0., axis=0)
    max_a_val, opt_x = ap_val, np.asarray(p0)
    for ii in range(len(rays)):
        u1, sign1, sp1, pi, _ = rays[ii]
        u2, sign2, sp2, pii, _ = rays[(ii + 1) % len(rays)]
        posi_val = (1. if sign1 == 0 and sp1 == 1 else 0.)
        nega_val = (1. if sign1 == 1 and sp1 == 0 else 0.)
        aq_val = ap_val + posi_val - nega_val
        if aq_val > max_a_val:
            max_a_val, opt_x = aq_val, np.asarray(pi) + np.asarray(pii)
        posi_val = (1. if sign2 == 0 and sp2 == 0 else 0.)
        nega_val = (1. if sign2 == 1 and sp2 == 1 else 0.)
        ap_val = aq_val + posi_val - nega_val
        if ap_val > max_a_val:
            max_a_val, opt_x = ap_val, np.asarray(pii)
    opt_x = change_coordinates_back(normal=u, w=opt_x)
    opt_auc = np.sum(np.array(np.dot(points, opt_x)) > 0., axis=0) / len(points)
    return opt_x, opt_auc


def open_hemisphere_3d(points: list, precision_eps=1e-15):
    """
    each point in points set are unit vector in 3-dimension.
    :param points:
    :param precision_eps
    :return:
    """
    for point in points:
        assert len(point) == 3
        assert np.abs(norm(point) - 1.) <= precision_eps
    opt_auc, opt_x = -1.0, np.zeros(3)
    # all nonzero points
    for ind, u in enumerate(points):
        # project all points to plane defined by its normal
        # u notice that c_prime may contain zero vectors.
        u_norm = norm(u) ** 2.
        cp = [p - (np.dot(p, u) / u_norm) * u for p in points]
        sp = [1.] * len(points)
        yu1, auc_val = auc_opt_2d(points, sp, u, cp)
        if opt_auc < auc_val:
            opt_x, opt_auc = yu1, auc_val
        sp = [1. if np.dot(_, u) <= 0.0 else 0. for _ in points]
        yu2, auc_val = auc_opt_2d(points, sp, u, cp)
        if opt_auc < auc_val:
            opt_x, opt_auc = yu2, auc_val
            epsilon = np.infty
            delta = -1.
            for point in points:
                val1 = np.dot(yu2, point)
                val2 = np.dot(u, point)
                if val1 > 0. and val2 < 0.:
                    if epsilon > val1:
                        epsilon = val1
                    if delta < (1. + np.abs(val2)):
                        delta = 1. + np.abs(val2)
            alpha = epsilon / delta
            opt_x1 = yu2 + alpha * u
            auc_val1 = np.sum(np.array(np.dot(points, opt_x1)) > 0., axis=0) / len(points)
            if auc_val1 > auc_val:
                opt_x, opt_auc = opt_x1, auc_val1
        auc = np.sum(np.array(np.dot(points, opt_x)) > 0., axis=0) / (len(points) * 1.)
        print(f"index: {ind} points: {len(points)} opt-x: {opt_x} AUC-val: {auc}")
    return opt_x, opt_auc


def opt_auc_3d_algo(x_tr, y_tr, precision_eps=1e-15):
    assert 3 == x_tr.shape[1]
    x_tr = np.asarray(x_tr, dtype=np.float64)
    posi_indices = [ind for ind, _ in enumerate(y_tr) if _ > 0.]
    nega_indices = [ind for ind, _ in enumerate(y_tr) if _ < 0.]
    set_k = []
    for i in posi_indices:
        for j in nega_indices:
            point = x_tr[i] - x_tr[j]
            if norm(point) <= precision_eps:
                # ignore the co-linear pair
                continue
            point = point / norm(point)
            set_k.append(point)
    w, opt_auc = open_hemisphere_3d(points=set_k)
    return w, opt_auc


def opt_auc_3d_algo_reg(x_tr, y_tr, precision_eps=1e-15):
    assert 3 == x_tr.shape[1]
    x_tr = np.asarray(x_tr, dtype=np.float64)
    posi_indices = [ind for ind, _ in enumerate(y_tr) if _ > 0.]
    nega_indices = [ind for ind, _ in enumerate(y_tr) if _ < 0.]
    set_k = []
    for i in posi_indices:
        for j in nega_indices:
            point = x_tr[i] - x_tr[j]
            if norm(point) <= precision_eps:
                # ignore the co-linear pair
                continue
            point = point / norm(point)
            set_k.append(point)
    w1, opt_auc = open_hemisphere_3d(points=set_k)
    w2, opt_auc = open_hemisphere_3d(points=set_k[::-1])
    return w1, w2
