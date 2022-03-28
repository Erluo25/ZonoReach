import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.linalg import expm

class UnitBox:
    def __init__(self):
        pass

    def maximum(self, dir):
        result = []
        for val in dir[0]:
            if val > 0:
                result.append(1)
            else:
                result.append(-1)
        return np.array([result])


class Zonotope:

    def __init__(self, center=None, g_mat=None):
        self.center = center
        self.g_mat = g_mat
        self.g_num = self.g_mat.shape[1]
        self.domain = UnitBox()
        self.init_bounds = np.array([[-1, 1] for _ in range(0, self.g_mat.shape[1])])
        self.bounds_in_all_dim = self.get_bounds()

    '''
    The purpose of this function is to calculate the upper/lower bound of the zonotope 
    Return: the max norm value.
    Reference: https://github.com/stanleybak/quickzonoreach/blob/master/quickzonoreach/zono.py
    '''

    def get_bounds(self):
        mat_t = self.g_mat
        size = self.center.size

        # pos_1_gens may need to be updated if matrix size changed due to assignment
        neg1_gens = np.array([[i[0]] for i in self.init_bounds], dtype=float)  # At this time all -1
        pos1_gens = np.array([[i[1]] for i in self.init_bounds], dtype=float)  # At this time all 1

        pos_mat = np.clip(mat_t, 0, np.inf)
        neg_mat = np.clip(mat_t, -np.inf, 0)

        pos_pos = np.dot(pos_mat, pos1_gens)
        neg_neg = np.dot(neg_mat, neg1_gens)

        pos_neg = np.dot(neg_mat, pos1_gens)
        neg_pos = np.dot(pos_mat, neg1_gens)

        rv = np.zeros((size, 2), dtype=float)
        dim = self.center.shape[0]
        rv[:, 0] = np.reshape((self.center + pos_neg + neg_pos), dim)
        rv[:, 1] = np.reshape((self.center + pos_pos + neg_neg), dim)

        return rv

    def zono_max_norm(self):
        rv = self.get_bounds()
        max_val = max(np.linalg.norm(rv[:, 0], ord=np.Inf), np.linalg.norm(rv[:, 1], ord=np.Inf))
        return max_val

    def zonomax(self, direction):
        result = self.center
        for i in range(0, self.g_num):
            col = self.g_mat[:, [i]]
            if direction @ col > 0:
                result = result + col
            else:
                result = result - col
        return result

    def domainmax(self, direction):
        domain_dir = direction @ self.g_mat
        domain_pts = self.domain.maximum(domain_dir)
        result = self.g_mat @ domain_pts.T + self.center
        return result

    def plot(self, num_dir=100, color='k'):
        sys_dim = self.g_mat.shape[0]
        '''
        if sys_dim > 2:
            #Need to do some projections
            temp_b_mat = np.zeros((sys_dim, 2))
            temp_b_mat[0][0] = 1
            temp_b_mat[1][1] = 1
            temp_b_mat = temp_b_mat @ temp_b_mat.T
            self.center = (temp_b_mat @ self.center)
            self.g_mat = (temp_b_mat @ self.g_mat)
            self.g_num = self.g_mat.shape[1]
        '''

        verts = []
        for theta in np.linspace(0, 2 * np.pi, num_dir):
            vx = np.cos(theta)
            vy = np.sin(theta)
            if sys_dim > 2:
                dir = np.zeros((1, sys_dim))
                dir[0][0] = vx
                dir[0][1] = vy
            else:
                dir = np.array([[vx, vy]])
            vert = self.domainmax(dir)
            assert np.allclose(vert, self.zonomax(direction=dir))
            verts.append(vert)
        xs = [pt[0] for pt in verts]
        ys = [pt[1] for pt in verts]
        plt.plot(xs, ys, color)

    def minkowski_sum(self, zono2):
        center = self.center + zono2.center
        g_mat = np.hstack((self.g_mat, zono2.g_mat))
        return Zonotope(center=center, g_mat=g_mat)

    '''
    def copy(self):
        return Zonotope(center=self.center, g_mat=self.g_mat)
    '''