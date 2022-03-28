import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.linalg import expm
import kamenev as kamenev


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
        for theta in np.linspace(0, 2*np.pi, num_dir):
            vx = np.cos(theta)
            vy = np.sin(theta)
            if sys_dim > 2:
                dir = np.zeros((1, sys_dim))
                dir[0][0] = vx
                dir[0][1] = vy
                #print(dir)
            else:
                dir = np.array([[vx, vy]])
            vert = self.domainmax(dir)
            assert np.allclose(vert, self.zonomax(direction=dir))
            verts.append(vert)
        xs = [pt[0] for pt in verts]
        ys = [pt[1] for pt in verts]
        plt.plot(xs, ys, color)

    '''
    
    def copy(self):
        return Zonotope(center=self.center, g_mat=self.g_mat)
    '''

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
        self.dim = transform_mat.shape[0]
        self.return_list = []

    '''
    Get the miu value based on input
    '''
    def get_miu(self):
        min_inputs = self.input_box[:, 0]
        max_inputs = self.input_box[:, 1]
        neg_ins = np.absolute(np.dot(self.input_mat, min_inputs))
        pos_ins = np.absolute(np.dot(self.input_mat, max_inputs))
        rv = max(np.max(neg_ins), np.max(pos_ins))
        return rv


    def solve(self, discrete_time=False):

        '''
        The entry point of solving the reachability problem.
        Will full fill the return_list.

        Steps:
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
        u = self.get_miu()
        beta = ((e_r_norm_A - 1) / max_norm_A) * u

        print("r is: ", r)
        print("max_norm_A is: ", max_norm_A)
        print("e^{r||A||} is: ", e_r_norm_A)
        print("sup_(x in Z) ||x||", self.init_zono.zono_max_norm())
        print("alpha: ", alpha)
        print("miu ", u)
        print("beta: ", beta)        

        # Now we need to get the P_0

        if discrete_time:
            q_0 = self.init_zono
        else:
            p_0 = self.get_p0()
            q_0 = self.bloat(zono=p_0, radius=(alpha + beta))

        self.return_list.append(q_0)
        for i in range(1, self.step_num):
            prev_q = self.return_list[-1]
            temp_p = self.forward_zono_one_step(zono=prev_q)

            temp_q = self.bloat(zono=temp_p, radius=beta)

            self.return_list.append(temp_q)

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
                zono.plot(color='b')
        else:
            print("There's no zonotope in the return list to plot ")

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
    prob.plot_result()
    plt.xlim([-16, 16])
    plt.ylim([-16, 16])
    plt.show()

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

