import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.linalg import expm
from zonotope import *

class Reachable_Problem():
    '''
    Params:
        time_step r, step_size N: time T = r * N / N = T / r
        transform_mat: the square matrix A representing the linear DEs system, size should be n * n
        init_zono: the initial zonotope, mainly composed of center and generators.
        input_mat:[[<--k-->], <--n-->]
            There are n sublist in the input_mat representing the input situation for each row of the DE system.
            Inside each sublist, there're k elements representing the scalar of each input u_i.
            If the row doesn't contain the some input u_i, it's slot will be 0.
        inputbox: [u1,u2, ....] k*2
                # x' = y + u1, y' = -x + + u1 + u2
                # u1 in [-0.5, 0.5], u2 in [-1, 0]
    '''
    def __init__(self, step_size, step_num, transform_mat,
                 init_zono: Zonotope, input_mat, input_box):
        assert step_size > 0 and step_num > 0, "step size and step_num need to be bigger than 0"
        assert (transform_mat.shape[0] == transform_mat.shape[1]), "transform matrix need to be n*n"
        self.step_size = step_size
        self.step_num = step_num
        self.a_mat = transform_mat
        self.e_r_A_mat = expm(self.a_mat * self.step_size)
        self.init_zono = init_zono
        assert (input_mat.shape[0] == self.a_mat.shape[0]) and (input_mat.shape[1] == input_box.shape[0])
        self.input_mat = input_mat
        self.input_box = input_box
        self.input_num = input_box.shape[0]
        self.dim = transform_mat.shape[0]
        self.return_list = []


    def solve(self, discrete_time=False):

        '''
        The entry point of solving the reachability problem.
        Will full fill the return_list.

        Steps:
        1. Prepare alpha value
        2. Get the box extension with alpha
        3. Prepare p0 then do minkowski sum with U --- this is the omega_0 in the newer paper
        4. Go into the loop at each round i from (0, step_num - 1)



        1. Prepare alpha and beta's value
        2. Get the box extension with alpha and beta
        3. Prepare initial P0, Q0, R0
        4. Go into the loop at each round i
            4.1 Apply e^{At} to the Q_(i-1) assign it to P_i
            4.2 Update Q_i with P_i + box(alpha + beta)
            4.3 Add Q_i to the return list

        :return: no exact return value. The result will be stored in self.return_list
        '''

        # First need to do some initializations
        self.return_list = []
        r = self.step_size
        max_norm_A = np.linalg.norm(self.a_mat, ord=np.Inf)
        # Calculate alpha and beta value
        e_r_norm_A = math.exp(r * max_norm_A)
        alpha = (e_r_norm_A - 1 - (r * max_norm_A)) * self.init_zono.zono_max_norm()
        # Get the zonotope U.
        miu = self.input_box_to_zono_miu()

        if discrete_time:
            omega_0 = self.init_zono
        else:
            p_0 = self.get_p0()
            q_0 = self.bloat(zono=p_0, radius=alpha)
            omega_0 = q_0.minkowski_sum(miu)
        self.return_list.append(omega_0)
        temp_x = omega_0
        temp_v = miu
        s_list = []
        for _ in range(0, self.step_num):
            temp_x = self.forward_zono_one_step(temp_x)
            if (len(s_list) == 0):
                s_list.append(temp_v)
            else:
                s_list.append(s_list[-1].minkowski_sum(temp_v))
            temp_v = self.forward_zono_one_step(temp_v)
            self.return_list.append(temp_x.minkowski_sum(s_list[-1]))

    def get_miu(self):

        '''
        Get the small miu value based on input
        '''

        min_inputs = self.input_box[:, 0]
        max_inputs = self.input_box[:, 1]
        neg_ins = np.absolute(np.dot(self.input_mat, min_inputs))
        pos_ins = np.absolute(np.dot(self.input_mat, max_inputs))
        rv = max(np.max(neg_ins), np.max(pos_ins))
        return rv


    def input_box_to_zono_miu(self):
        '''
        The input box will have size k * 2
        k represent the number of inputs u1, u2, ..., uk
        and 2 means each one has a range

        1. We represent the input_box into a zonotope.
        2. For each point inside the box / zonotope (which is a k * 1 vector)
                This function transform a vector in the input domain [u1, u2, ..., uk] telling possible u1 to uk value,
                into the system domain expressing something like
                [[u1 + u3], [0.5 * u2 + 3 * u4 - uk], ... <-- n terms corresponding each row of the system --> ]
                by using the b_mat matrix.
        3. After that, apply the vector to the matrix A^-1 (e^Ar - I) to form a point in U

        The main idea is apply 2~3 to each point inside the box which is equivalent to apply 2~3 to the whole zonotope
        '''
        input_box = self.input_box
        a_inv_mat = np.linalg.inv(self.a_mat)
        trans_mat = a_inv_mat @ ((self.e_r_A_mat - np.identity(self.dim)) @ self.input_mat)

        # Now calculate the center and g_mat in the input domain
        center = (np.sum(input_box, axis=1) / 2).reshape(self.input_num, 1)
        g_mat_vec = input_box[:, 1].reshape(self.input_num, 1) - center
        temp_i = np.identity(n=self.input_num)
        g_mat = temp_i * g_mat_vec

        # Now apply trans_mat to transform set of points in the input domain to the system domain
        center = trans_mat @ center
        g_mat = trans_mat @ g_mat
        zono_miu = Zonotope(center=center, g_mat=g_mat)
        return zono_miu

    def get_p0(self):
        '''
        Use the given initial zonotope, create a new Zonotope which extend it to P_0
        :return: A new zonotope representing P_0
        '''
        mat = self.e_r_A_mat
        init_zono = self.init_zono
        center = init_zono.center
        g_mat = init_zono.g_mat
        new_center = (center + np.dot(mat, center)) / 2
        new_g_mat_pos = (g_mat + np.dot(mat, g_mat)) / 2
        new_g_mat_c = (center - np.dot(mat, center)) / 2
        new_g_mat_neg = (g_mat - np.dot(mat, g_mat)) / 2
        new_g_mat = np.hstack((new_g_mat_pos, new_g_mat_c, new_g_mat_neg))
        return Zonotope(center=new_center, g_mat=new_g_mat)

    def bloat(self, zono: Zonotope, radius):
        '''
        Given a zonotope and a raidus, this function will created a NEW zonotope which is the \
        bloated given zonotope by a box of radius r.
        :param zono: The given zonotope to be bloated
        :param radius: The radius of the box to bloat
        :return: A new bloated zonotope
        '''
        g_mat = zono.g_mat
        dim = g_mat.shape[0]

        bloat_gen_mat = radius * np.identity(dim)
        if radius > 1e-9:
            new_g_mat = np.hstack((g_mat, bloat_gen_mat))
        else:
            new_g_mat = g_mat.copy()

        # Return a copy of the modified zonotope.
        return Zonotope(center=zono.center, g_mat=new_g_mat)

    def forward_zono_one_step(self, zono:Zonotope):
        '''
        Forward the given zonotope using e^(r*A) == self.e_r_A_mat without bloating
        :param zono: The zonotope going to be forward.
        :return: A new zonotope representing the forwarded zonotope by one step
        '''
        new_center = np.dot(self.e_r_A_mat, zono.center)
        new_g_mat = np.dot(self.e_r_A_mat, zono.g_mat)
        return Zonotope(center=new_center, g_mat=new_g_mat)

    def plot_result(self):
        self.init_zono.plot(color='r-o')
        #print("There are ", len(self.return_list), " zonotopes in the return list")
        if len(self.return_list) > 0:
            for zono in self.return_list:
                zono.plot(color='b-o')
        else:
            print("There's no zonotope in the return list to plot ")

