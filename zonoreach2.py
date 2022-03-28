import numpy as np
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

    def __init__(self, center, g_mat):
        self.center = center
        self.g_mat = g_mat
        self.g_num = self.g_mat.shape[1]
        self.domain = UnitBox()

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

    """This plot only works for 2-D range sapce"""
    """
    def plot(self):
        verts = []
        for theta in np.linspace(0, 2*np.pi, 50):
            vx = np.cos(theta)
            vy = np.sin(theta)
            dir = np.array([[vx, vy]])
            vert = self.domainmax(dir)#self.zonomax(direction=dir)
            verts.append(vert)
        xs = [pt[0] for pt in verts]
        ys = [pt[1] for pt in verts]
        plt.plot(xs, ys, "black")    
    """

    def plot(self):
        verts = []
        for theta in np.linspace(0, 2*np.pi, 50):
            vx = np.cos(theta)
            vy = np.sin(theta)
            dir = np.array([[vx, vy]])
            vert = self.domainmax(dir)#self.zonomax(direction=dir)
            verts.append(vert)
        xs = [pt[0] for pt in verts]
        ys = [pt[1] for pt in verts]
        plt.plot(xs, ys, "black")


def move_zono_forward(zono:Zonotope, de_mat, step_size, step_num):
    #Rise the de_mat to transformation matrix
    mat = expm(de_mat * step_size)
    for i in range(0, step_num):
        zono.center = mat @ zono.center
        zono.g_mat = mat @ zono.g_mat
        zono.plot()



def main():
    # Some inputs
    step_size = 2 * np.pi / 8
    step_num = 10

    # Let's define the zonotope X: [-5, -4], Y:[0, 1]
    center = np.array([[-4.5], [0.5]])
    g_mat = np.array([[0.5, 0.0], [0.0, 0.5]])
    init = Zonotope(center, g_mat)
    init.plot()

    # Let's define out de matrix x ' = y, y' = -x
    de_mat = np.array([[0.0, 1.0], [-1.0, 0.0]])
    move_zono_forward(init, de_mat, step_size, step_num)

    plt.xlim([-6,6])
    plt.ylim([-6,6])
    plt.show()


if __name__ == '__main__':
    main()