import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize


def triangle_normal(pa, pb, pc):
    n = np.cross(pc - pa, pb - pa)
    return n / norm(n)


def make_2d_coordinates():
    np.cross()
    pass


def proj_3d_2d(normal, points):
    proj_points = []
    normal_square = np.linalg.norm(normal) ** 2.
    for point in points:
        proj_points.append(point - (np.dot(point, normal) / normal_square) * normal)
    return proj_points


def proj_2d_3d(p0, u, v, mapped_point):
    aa, bb = mapped_point
    x = np.dot([p0[0], u[0], v[0]], [1, aa, bb])
    y = np.dot([p0[1], u[1], v[1]], [1, aa, bb])
    z = np.dot([p0[2], u[2], v[2]], [1, aa, bb])
    return np.asarray([x, y, z], dtype=np.float64)


def test1():
    np.random.seed(17)
    normal = np.random.normal(0, 1., 3)
    points = np.random.normal(0., 1., size=(100, 3))
    points = np.asarray([_ / np.linalg.norm(_) for _ in points])
    proj_points = proj_3d_2d(normal, points)
    u = np.asarray([normal[1], -normal[0], 0])
    u = u / norm(u)
    v = np.cross(u, normal)
    v = v / norm(v)
    mapped_points = [(np.dot(p, u), np.dot(p, v)) for p in proj_points]
    w = proj_points[20]
    w_proj = mapped_points[20]
    for aa, bb in zip(np.dot(proj_points, w), np.dot(mapped_points, w_proj)):
        if np.dot(aa, bb) > 0:
            print(aa, bb, "yes")
        else:
            print(aa, bb, "no")
    w = np.asarray([-mapped_points[2][1], mapped_points[2][0]])
    w_back = np.asarray(w[0] * u + w[1] * v)
    for x, y in zip(np.dot(mapped_points, w), np.dot(proj_points, w_back)):
        print(np.dot(x, y))
    print(np.sum(np.array(np.dot(mapped_points, w)) > 0., axis=0))
    print(np.sum(np.array(np.dot(mapped_points, w)), axis=0))

    print(np.sum(np.array(np.dot(proj_points, w_back)) > 0., axis=0))
    print(np.sum(np.array(np.dot(points, w_back)) > 0., axis=0))

    print(np.sum(np.array(np.dot(proj_points, w_back)), axis=0))
    print(np.sum(np.array(np.dot(points, w_back)), axis=0))


def test2():
    pa = np.random.normal(0., 1., 3)
    pb = np.random.normal(0., 1., 3)
    pc = np.random.normal(0., 1., 3)
    normal = triangle_normal(pa, pb, pc)
    print(np.dot(pa, normal))
    print(np.dot(pb, normal))
    print(np.dot(pc, normal))
    print(np.dot(pa - pb, normal))
    print(np.dot(pb - pc, normal))
    print(np.dot(pc - pa, normal))
    np.random.seed(17)
    normal = np.random.normal(0, 1., 3)
    points = np.random.normal(0., 1., size=(100, 3))
    proj_points = []
    for _ in points:
        proj_points.append(_ - (np.dot(_, normal) / np.linalg.norm(normal) * normal))

    for i in range(100):
        xx = np.random.normal(0., 1., 3)
        xx[2] = - (xx[0] * normal[0] + xx[1] * normal[1]) / normal[2]
        for aa, bb in zip(points, proj_points):
            print(np.abs(np.dot(aa, xx) - np.dot(bb, xx)))


test1()
