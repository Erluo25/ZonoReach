import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.linalg import expm
from zonotope import *
from reachabilitySolver import *

def main():

    test1()


def test1():
    step_size = np.pi / 4
    max_steps = 3
    # Want to perform the the follow DE
    mat_a = np.array([[0.0, 1.0], [-1.0, 0.0]])

    # Init of x: x in [-5, -4], y in [0, 1]
    g_mat = np.array([[0.5, 0], [0, 0.5]])
    center = np.array([[-4.5], [0.5]])
    init = Zonotope(center, g_mat)
    # init.plot()
    input_mat = np.array([[1, 0], [1, 1]])
    input_box = np.array([[-0.5, 0.5], [-1, 0]])
    prob = Reachable_Problem(step_size=step_size, step_num=max_steps,
                             transform_mat=mat_a, init_zono=init,
                             input_mat=input_mat, input_box=input_box)
    prob.solve(discrete_time=True)
    #prob.plot_result()
    #plt.xlim([-16, 16])
    #plt.ylim([-16, 16])
    #plt.show()

def test2():
    step_size = 0.02
    max_steps = 100
    # Want to perform the the follow DE
    mat_a = np.array([[-1.0, -4.0], [4, -1.0]])

    # Init of x: x in [0.9, 1.1], y in [-0.1, 0.1]
    g_mat = np.array([[0.1, 0], [0, 0.1]])
    center = np.array([[1.0], [0.0]])
    init = Zonotope(center, g_mat)
    # init.plot()
    input_mat = np.array([[1, 0], [1, 0]])
    input_box = np.array([[-0.05, 0.05], [0.0, 0.0]])
    prob = Reachable_Problem(step_size=step_size, step_num=max_steps,
                             transform_mat=mat_a, init_zono=init,
                             input_mat=input_mat, input_box=input_box)
    prob.solve()
    prob.plot_result()
    plt.xlim([-0.8, 1.2])
    plt.ylim([-0.6, 1.0])
    plt.show()


def test3():
    step_size = 0.005
    max_steps = 10
    # Want to perform the the follow DE
    mat_a = np.array([[-1.0, -4.0, 0.0, 0.0, 0.0],
                      [4.0, -1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, -3.0, 1.0, 0.0],
                      [0.0, 0.0, -1.0, -3.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, -2.0]])

    # Init of x: x in [0.9, 1.1], y in [-0.1, 0.1]
    g_mat = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    center = np.array([[1.0], [0.0], [0.0], [0.0], [0.0]])
    init = Zonotope(center, g_mat)
    # init.plot()
    input_mat = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])
    input_box = np.array([[-0.01, 0.01], [0.0, 0.0]])
    prob = Reachable_Problem(step_size=step_size, step_num=max_steps,
                             transform_mat=mat_a, init_zono=init,
                             input_mat=input_mat, input_box=input_box)
    prob.solve()
    #print(len(prob.return_list))
    prob.plot_result()
    #plt.xlim([-1, 1.5])
    #plt.ylim([-1, 1])
    plt.show()


if __name__ == '__main__':
    main()

