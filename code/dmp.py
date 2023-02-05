import numpy as np


def estimate_derivatives(t, x):
    # time step
    t_delta = t[1:] - t[:-1]
    t_delta = np.concatenate([t_delta[[0]], t_delta])[:, np.newaxis]
    # velocity
    v0 = -1.5*x[[0], :] + 2*x[[1], :] - 0.5*x[[2], :]
    vt = -0.5*x[:-2, :] + 0.5*x[2:, :]
    vT = 0.5*x[[-3], :] - 2*x[[-2], :] + 1.5*x[[-1], :]
    v = np.concatenate([v0, vt, vT]) / t_delta
    # acceleration
    a0 = 2*x[[0], :] - 5*x[[1], :] + 4*x[[2], :] - 1*x[[3], :]
    at = 1*x[:-2, :] - 2*x[1:-1, :] + 1*x[2:, :]
    aT = -1*x[[-4], :] + 4*x[[-3], :] - 5*x[[-2], :] + 2*x[[-1], :]
    a = np.concatenate([a0, at, aT]) / t_delta**2
    return x, v, a


tableau_C = np.array([
    0, 1/4, 3/8, 12/13, 1, 1/2  # sub-steps coefficients for each ki
])
tableau_A = np.array([
    [        0,          0,          0,         0,      0],  # k1
    [      1/4,          0,          0,         0,      0],  # k2
    [     3/32,       9/32,          0,         0,      0],  # k3
    [1932/2197, -7200/2197,  7296/2197,         0,      0],  # k4
    [  439/216,         -8,   3680/513, -845/4104,      0],  # k5
    [    -8/27,          2, -3544/2565, 1859/4104, -11/40]   # k6
])
tableau_B = np.array([
    [  16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],  # order 5
    [  25/216, 0,  1408/2565,   2197/4104,  -1/5,    0]   # order 4
])


# noinspection PyPep8Naming
def RK45(f, t0, y0, h_init=1e-3, e_tol=1e-6, h_min=1e-10, T=float('inf'), final_cond=None):

    h = h_init
    tt = [t0]
    yy = np.zeros(shape=(1, y0.shape[0]))
    yy[0, :] = y0

    s = tableau_C.shape[0]
    K = np.zeros(shape=(s, y0.shape[0]))
    keep_going = True
    can_continue = True
    while keep_going:
        t = tt[-1]
        y = yy[-1, :]
        j = 0
        while True:
            # compute coefficients
            for i in range(s):
                k_t = t + h * tableau_C[i]
                k_y = y + h * tableau_A[i, :] @ K[:-1, :]
                K[i, :] = f(k_t, k_y)

            # compute next step
            y_now_higher = y + h * tableau_B[0, :] @ K
            y_now_lower = y + h * tableau_B[1, :] @ K

            # compute truncation error and new step size
            # trunc_error = np.linalg.norm(y_now_higher - y_now_lower)
            # h_new = 0.9 * h * (e_tol / trunc_error) ** (1/5)
            # h = h_new
            # if the error is lower than the tolerance, move to the next step
            # if not, recompute this step with the new step size (h_new)
            # if trunc_error <= e_tol:
            #     break

            trunc_error = np.linalg.norm(y_now_higher - y_now_lower) / np.linalg.norm(y_now_higher)
            if trunc_error <= e_tol:
                h *= 1.1
                break
            else:
                h *= 0.9

            j += 1
            if j > 1000:
                print("Too many loops for h!")
                can_continue = False
                break

            if h < h_min:
                print("Reached too small h!")
                can_continue = False
                break

        # save the state
        t_now = t + h
        y_now = y_now_higher[np.newaxis, :]
        tt.append(t_now)
        yy = np.concatenate([yy, y_now])

        # check the termination condition
        time_condition = t_now < T
        goal_condition = final_cond is None or not final_cond(t_now, y_now[0, :])
        keep_going = (time_condition and goal_condition) and can_continue

    return np.array(tt), yy


# noinspection PyPep8Naming
def EUL(f, t0, y0, h_init=1e-3, e_tol=1e-6, h_min=1e-10, T=float('inf'), final_cond=None):
    h = h_init
    tt = [t0]
    yy = np.zeros(shape=(1, y0.shape[0]))
    yy[0, :] = y0

    keep_going = True
    while keep_going:
        # this state
        t = tt[-1]
        y = yy[-1, :]

        # evolution using Euler
        y_now = y + h * f(t, y)

        # save the state
        t_now = t + h
        y_now = y_now[np.newaxis, :]
        tt.append(t_now)
        yy = np.concatenate([yy, y_now])

        # check the termination condition
        time_condition = t_now < T
        goal_condition = final_cond is None or not final_cond(t_now, y_now[0, :])
        keep_going = time_condition and goal_condition

    return np.array(tt), yy


class Behaviour(object):
    
    def __init__(self, time_span, position, velocity=None, acceleration=None):
        self.time = time_span
        self.T = time_span[-1]
        
        self.p = position
        self.p0 = position[0, :]
        self.pT = position[-1, :]
        
        if velocity is not None and acceleration is not None:
            self.v = velocity
            self.a = acceleration
        else:
            _, self.v, self.a = estimate_derivatives(self.time, self.p)


class DMP(object):
    
    # TODO allow different K values
    def __init__(self, n_dim, K, n_basis, alpha=4):
        '''
        Initialize the DMP object. The user should be able to set
        * the dimension of the system,
        * elastic term (damping term is automatically set to have critical damping)
        * number (and, optionally, parameters) of basis functions
        * decay parameter of the canonical system
        '''
        self.n_dim = n_dim
        self.elastic_constant = K
        self.damping_constant = 2 * np.sqrt(self.elastic_constant)
        self.n_basis = n_basis
        self.alpha_cs = alpha
        self.T = 1

        # default parameters
        self.behaviour = None
        self.weights = np.zeros([self.n_dim, self.n_basis+1])  # weights of the dynamic
        self.obstacles = lambda x, v: 0  # no obstacles

    def exp_basis(self, s):
        i = np.arange(0, self.n_basis + 1)
        c = np.exp(-self.alpha_cs * i * self.T / self.n_basis)
        h = np.zeros(self.n_basis + 1)
        h[:-1] = 1 / (c[1:] - c[:-1]) ** 2
        h[-1] = h[-2]
        return np.exp(-h * (s - c) ** 2)

    # TODO allow for custom basis functions
    def compute_basis_vector(self, s):
        basis = self.exp_basis(s)
        Psi = basis / np.sum(basis, axis=1, keepdims=True) * s  # fucking bitch s
        return Psi

    def compute_perturbation(self, s):
        # weights = [dim]x[N+1]
        # basis   = [N+1]x[1]
        Psi = self.compute_basis_vector(s)

        #   [dim]x[1] / [1] * [1]
        f = self.weights @ Psi.T
        if f.shape[1] == 1:
            f = f[:, 0]
        return f

    def set_obstacles(self, *obstacles):
        self.obstacles = lambda x, v: sum([obs.p(x, v) for obs in obstacles])

    def learn_trajectory(self, desired_behavior: Behaviour):
        '''
        This method should compute the set of weights given a desired behavior.

        desired_behavior is a [time]*[dim] matrix
        '''
        #############################################
        # Step 1 : extract the desired forcing term #
        #############################################

        # prep
        self.behaviour = desired_behavior
        self.T = self.behaviour.T

        K = self.elastic_constant * np.eye(self.n_dim)
        D = self.damping_constant * np.eye(self.n_dim)

        # basically, we use this regressor
        # f(s) = W*Psi(s)
        # Psi(s) = [ phi_i(s) / SUM_i(phi_i(s)) ]

        s0 = 1
        tau = 1
        s = s0 * np.exp(-self.alpha_cs / tau * self.behaviour.time)[:, np.newaxis]

        # compute desired perturbations
        x0, g = self.behaviour.p0, self.behaviour.pT
        x, v, a = self.behaviour.p, self.behaviour.v, self.behaviour.a
        f = (a + v @ D) @ np.linalg.inv(K) - (g - x) + (g - x0) * s

        # compute the basis vector
        Psi = self.compute_basis_vector(s)

        ##############################################################################
        # Step 2 : compute the set of weights starting from the desired forcing term #
        ##############################################################################
        F, P = f.T, Psi.T
        P_pinv = P.T @ np.linalg.inv(P @ P.T)
        weights = F @ P_pinv
        
        self.weights = weights
        return weights

    def execute_trajectory(self, x0, xgoal, t_delta=None, tau=1, tol=1e-3, use_euler=False):
        '''
        This method should return an execution of the dynamical system.
        The system should evolve until convergence (within a given tolerance) to the goal is achieved.
        '''

        def dynamics(t, z, g, K, D):
            v = z[0:self.n_dim]
            x = z[self.n_dim:2 * self.n_dim]
            s = z[-1:]
            f = self.compute_perturbation(s[:, np.newaxis])
            dyn = np.concatenate([
                K @ (g(t) - x) - D @ v - K @ (g(t) - x0) * s + K @ f + self.obstacles(x, v),
                v,
                -self.alpha_cs * s
            ]) / tau
            return dyn

        return self._execute(dynamics, x0, xgoal, t_delta, tol, use_euler)

    def execute_trajectory_scaled(self, x0, xgoal, t_delta=None, tau=1, tol=1e-3, use_euler=False):
        '''
        This method should return an execution of the dynamical system.
        The system should evolve until convergence (within a given tolerance) to the goal is achieved.
        '''

        scalability = self.get_scalability_function()

        def dynamics(t, z, g, K, D):
            v = z[0:self.n_dim]
            x = z[self.n_dim:2 * self.n_dim]
            s = z[-1:]

            # change the system
            S = scalability(self.behaviour.p0, self.behaviour.pT, x0, xgoal(t))
            S_inv = np.linalg.inv(S)
            K = S @ K @ S_inv
            D = S @ D @ S_inv
            f = S @ self.compute_perturbation(s[:, np.newaxis])

            dyn = np.concatenate([
                K @ (g(t) - x) - D @ v - K @ (g(t) - x0) * s + K @ f + self.obstacles(x, v),
                v,
                -self.alpha_cs * s
            ]) / tau
            return dyn

        return self._execute(dynamics, x0, xgoal, t_delta, tol, use_euler)

    def _execute(self, dynamics, x0, xgoal, t_delta=None, tol=1e-3, use_euler=False):
        '''
        This method should return an execution of the dynamical system.
        The system should evolve until convergence (within a given tolerance) to the goal is achieved.
        '''

        D = self.damping_constant * np.eye(self.n_dim)
        K = self.elastic_constant * np.eye(self.n_dim)

        v0 = np.zeros(shape=(self.n_dim,))
        s0 = np.ones(shape=(1,))
        t0 = 0

        z0 = np.concatenate([v0, x0, s0])
        dyn = lambda t, z: dynamics(t, z, xgoal, K, D)  # time is not important here
        cond = lambda t, z: np.linalg.norm(z[self.n_dim:2 * self.n_dim] - xgoal(t)) <= tol  # ||x - g|| <= tol

        if use_euler:
            solver = EUL
        else:
            solver = RK45
        time_span, z_span = solver(dyn, t0, z0, h_init=t_delta, final_cond=cond)
        return time_span, z_span

    def get_scalability_function(self):
        if self.n_dim == 2:
            return self._scalability2d
        elif self.n_dim == 3:
            return self._scalability3d
        else:
            raise Exception("Only 2D and 3D scalability is implemented")

    def _scalability2d(self, x0, g, x0_new, g_new):

        # versors
        v = g - x0
        v_new = g_new - x0_new

        # 2D rotation matrix around the Z axis
        th = np.arccos(v @ v_new / (np.linalg.norm(v) * np.linalg.norm(v_new)))

        R = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th), np.cos(th)]
        ])
        l = np.linalg.norm(v_new) / np.linalg.norm(v)
        S = l * R
        return S

    def _scalability3d(self, x0, g, x0_new, g_new):

        # versors
        a = (g - x0) / np.linalg.norm(g - x0)
        b = (g_new - x0_new) / np.linalg.norm(g_new - x0_new)

        # 3D rotation matrix that sends  a  to  b  around the axis between them
        c = np.dot(a, b)
        s = np.linalg.norm(np.cross(a, b))
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        # transform the rotation on the canonical axis
        u = a
        v = (b - (a @ b) * a) / np.linalg.norm(b - (a @ b) * a)
        w = np.cross(u, v)
        F = np.zeros((3, 3))
        F[:, 0] = u
        F[:, 1] = v
        F[:, 2] = w
        R = np.linalg.inv(F) @ R @ F

        # create the final matrix
        l = np.linalg.norm(g - x0) / np.linalg.norm(g_new - x0_new)
        S = l * R
        return S
