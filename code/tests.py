import numpy as np
import matplotlib.pyplot as plt
from dmp import DMP, estimate_derivatives, Behaviour


# PLOTTING

def plot_evolutions(trjs, legend=None, savepath=None):
    plt.figure()
    for t, x, style in trjs:
        plt.plot(t, x, style)
    if legend is not None:
        plt.legend(legend)
    plt.xlabel("t")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, format='pdf', dpi=1200)


def plot_trajectories(trjs, legend=None, extra_plot_handles=None, show_endpoints=True, savepath=None):
    plt.figure()

    # choose between 2D or 3D
    n_dims = trjs[0][1].shape[1]
    if n_dims == 2:
        ax = plt.gca()
        plt.axis('equal')
    elif n_dims == 3:
        ax = plt.axes(projection='3d')
        ax.view_init(15, 30)
        plt.axis('auto')
    else:
        raise Exception("Cannot plot in more than 3 dimensions!")

    # for obstacles or anything else
    if extra_plot_handles is not None:
        extra_plot_handles()

    # for the trajectories
    if n_dims == 3:
        for t, x, style in trjs:
            ax.plot3D(x[:, 0], x[:, 1], x[:, 2], style)
            if show_endpoints:
                ax.plot3D(x[0, 0], x[0, 1], x[0, 2], 'ok')
                ax.plot3D(x[-1, 0], x[-1, 1], x[-1, 2], 'xk')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        for t, x, style in trjs:
            ax.plot(x[:, 0], x[:, 1], style)
            if show_endpoints:
                ax.plot(x[0, 0], x[0, 1], 'ok')
                ax.plot(x[-1, 0], x[-1, 1], 'xk')
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # show the legend if necessary
    if legend is not None:
        plt.legend(legend)
    
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, format='pdf', dpi=1200)


# noinspection PyPep8Naming
def plot_obstacle_point(o, show_3D=False):
    if show_3D:
        plt.gca().plot3D(o[0], o[1], o[2], 'or')
    else:
        plt.plot(o[0], o[1], 'or')


def plot_moving_goal(time_span, goal):
    goal_span = np.array([goal(t) for t in time_span])
    if goal_span.shape[1] == 3:
        plt.gca().plot3D(goal_span[:, 0], goal_span[:, 1], goal_span[:, 2], ":k")
    else:
        plt.plot(goal_span[:, 0], goal_span[:, 1], ":k")


# TESTS

def test_default_trajectory():
    #################################################################
    # try the learning process on the un-perturbed scenario (W = 0) #
    # so to find out the errors in the estimation                   #
    #################################################################

    # create the DMP
    n_dim = 2
    dmp = DMP(n_dim=n_dim, K=150, n_basis=50, alpha=4)

    # look at the default trajectory
    x0 = np.zeros(shape=(n_dim,))
    xgoal = lambda t: np.ones(shape=(n_dim,))
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-3, tol=1e-5)

    # learn the weights using the default trajectory and re-compute the trajectory
    db = Behaviour(t_dmp, trj_dmp[:, 2:4])
    dmp.learn_trajectory(db)
    t_dmp_new, trj_dmp_new = dmp.execute_trajectory(x0, xgoal, t_delta=1e-3, tol=1e-5)

    # compare the trajectories
    plot_trajectories([
        (t_dmp, trj_dmp[:, 2:4], "--b"),
        (t_dmp_new, trj_dmp_new[:, 2:4], "r")
    ],
        savepath='../plots/trj_default.pdf'
    )
    plot_evolutions([
        (t_dmp, trj_dmp[:, 2], "--b"),
        (t_dmp, trj_dmp[:, 3], "--c"),
        (t_dmp_new, trj_dmp_new[:, 2], "r"),
        (t_dmp_new, trj_dmp_new[:, 3], "m")
    ],
        legend=["x_default", "y_default", "x_dmp", "y_dmp"],
        savepath='../plots/evol_default.pdf'
    )


def test_simple_trajectory():

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,              # x
        np.sin(time_span) ** 2  # y
    ]).T
    trj_desired_vel = np.array([
        time_span,                              # x
        2*np.sin(time_span)*np.cos(time_span)   # y
    ]).T
    trj_desired_acc = np.array([
        time_span,                                      # x
        2*np.cos(time_span)**2-2*np.sin(time_span)**2   # y
    ]).T
    db = Behaviour(time_span, trj_desired, trj_desired_vel, trj_desired_acc)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)


    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = np.array([0.0, 0.0])
    xgoal = lambda t: np.array([3.6, 0.2])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-1)

    plot_trajectories([
        (time_span, trj_desired[:, 0:2], "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r")
    ],
        savepath='../plots/trj_simple.pdf'
    )
    plot_evolutions([
        (t_dmp, trj_dmp[:, 2], "c"),
        (t_dmp, trj_dmp[:, 3], "--c"),
        (t_dmp, trj_dmp[:, 0], "m"),
        (t_dmp, trj_dmp[:, 1], "--m"),
        (t_dmp, trj_dmp[:, 4], "y")
    ],
        legend=["$x(t)$", "$y(t)$", r"$\dot{x}(t)$", r"$\dot{y}(t)$", "$s(t)$"],
        savepath='../plots/evol_simple.pdf'
    )


def test_trajectory_robustness(random_start=True, random_goal=True):
    
    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 5, 100)
    trj_desired = np.array([
        (time_span ** 2) * np.cos(time_span),      # x
        time_span * np.sin(time_span)  # y
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    weights = dmp.learn_trajectory(db)
    dmp.weights = weights

    # Many executions
    trajectories = []
    for i in range(25):
        x0 = trj_desired[0, :] + (np.random.random(2) * 1.0 - 0.5) * random_start
        new_goal = trj_desired[-1, :] + (np.random.random(2) * 1.0 - 0.5) * random_goal
        xgoal = lambda t: new_goal
        t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=0.5)
        trajectories.append((t_dmp, trj_dmp[:, 2:4], ""))

    trajectories.append((time_span, trj_desired, "--b"))
    
    plot_trajectories(trajectories, savepath='../plots/trj_robustness.pdf')


def test_simple_3D_trajectory():

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        np.cos(time_span),  # x
        np.sin(time_span),  # y
        time_span           # z
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = trj_desired[0, :]
    xgoal = lambda t: trj_desired[-1, :] + np.array([0.0, 0.2, 1])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r")
    ],
        savepath='../plots/trj_simple_3D.pdf'
    )


def test_discontinuous_3D_trajectory():

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 4, 200)
    # a square rotated by 45 degrees over the x-axis
    trj_desired = np.zeros(shape=(len(time_span), 3))
    t1 = time_span < 1
    trj_desired[t1] = \
        np.array([
            1 - time_span[t1],     # x
            np.ones(time_span[t1].shape),  # y
            time_span[t1]      # z
        ]).T
    t2 = np.logical_and(1 <= time_span, time_span < 2)
    trj_desired[t2] = \
        np.array([
            np.zeros(time_span[t2].shape),  # x
            1 - (time_span[t2] - 1),        # y
            np.ones(time_span[t2].shape)    # z
        ]).T
    t3 = np.logical_and(2 <= time_span, time_span < 3)
    trj_desired[t3] = \
        np.array([
            time_span[t3] - 2,  # x
            np.zeros(time_span[t3].shape),   # y
            1 - (time_span[t3] - 2)   # z
        ]).T
    t4 = 3 <= time_span
    trj_desired[t4] = \
        np.array([
            np.ones(time_span[t4].shape),   # x
            time_span[t4] - 3,               # y
            np.zeros(time_span[t4].shape),   # z
        ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = trj_desired[0, :]
    xgoal = lambda t: trj_desired[-1, :] + np.array([0.0, -0.1, 0.1])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r")
    ],
        savepath='../plots/trj_discontinuous_3D.pdf'
    )


def test_ugly_trajectory():
    
    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,              # x
        np.sin(time_span) ** 2  # y
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = np.array([0.0, 0.0])
    xgoal = lambda t: np.array([-6, 8])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)

    plot_trajectories([
        (time_span, trj_desired[:, 0:2], "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r")
    ],
        savepath='../plots/trj_ugly_shape.pdf'
    )


def test_ugly_3D_trajectory():

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        np.cos(time_span),  # x
        np.sin(time_span),  # y
        time_span           # z
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = np.array([1.0, 0.0, 0.0])
    xgoal = lambda t: np.array([1.0, 0.0, 2.6])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-2)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r")
    ],
        savepath='../plots/trj_ugly_shape_3D.pdf'
    )


# TEST SCALABILITY

def test_simple_scalable_trajectory():

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,              # x
        np.sin(time_span) ** 2  # y
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution and application of affine transformation
    x0 = np.array([0.0, 0.0])
    xgoal = lambda t: np.array([-6, 8])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=0.02)
    t_dmp_scaled, trj_dmp_scaled = dmp.execute_trajectory_scaled(x0, xgoal, t_delta=1e-2, tol=0.02)

    # plotting
    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r"),
        (t_dmp_scaled, trj_dmp_scaled[:, 2:4], "-.g")
    ],
        savepath='../plots/trj_scalable.pdf'
    )


def test_simple_3D_scalable_trajectory():

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        np.cos(time_span),  # x
        np.sin(time_span),  # y
        time_span           # z
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution
    x0 = np.array([1.0, 0.0, 0.0])
    xgoal = lambda t: np.array([1.0, 0.0, -2*np.pi])
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=0.015)
    t_dmp_scaled, trj_dmp_scaled = dmp.execute_trajectory_scaled(x0, xgoal, t_delta=1e-2, tol=0.015)

    # plotting
    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r"),
        (t_dmp_scaled, trj_dmp_scaled[:, 3:6], "-.g")
    ],
        savepath='../plots/trj_scalable_3D.pdf'
    )


# TEST MOVING GOAL

def test_simple_scalable_trajectory_moving_goal():

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,              # x
        np.sin(time_span) ** 2  # y
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    ###################
    # EXECUTION PHASE #
    ###################

    # Execution

    def goal_circular(t):
        t = min(1, t / np.pi)  # stop a little before 1rad, only reached when  t = pi  (end of the trajectory)
        return 2*np.pi*np.array([np.cos(t), np.sin(t)])

    x0 = np.array([0.0, 0.0])
    xgoal = goal_circular
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=0.1)
    t_dmp, trj_dmp_scaled = dmp.execute_trajectory_scaled(x0, xgoal, t_delta=1e-2, tol=0.1)

    def extra_handles():
        plot_moving_goal(t_dmp, xgoal)

    plot_trajectories([
        (time_span, trj_desired[:, 0:2], "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r"),
        (t_dmp, trj_dmp_scaled[:, 2:4], "-.g")
    ],
        extra_plot_handles=extra_handles,
        savepath='../plots/trj_scalable_moving_goal.pdf'
    )


def test_simple_3D_scalable_trajectory_moving_goal():

    ##################
    # LEARNING PHASE #
    ##################

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        np.cos(time_span),  # x
        np.sin(time_span),  # y
        time_span           # z
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=250, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    ###################
    # EXECUTION PHASE #
    ###################

    def goal_linear(t):
        ti = 0.5*np.pi
        tf = 1.5*np.pi
        s = np.clip(0, (t - ti)/(tf - ti), 1)
        return np.array([1.0, 0.0, -2*np.pi]) - 8.0*np.array([0, 0, 1]) * s

    x0 = trj_desired[0, :]
    xgoal = goal_linear
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=0.01)
    t_dmp_scaled, trj_dmp_scaled = dmp.execute_trajectory_scaled(x0, xgoal, t_delta=1e-2, tol=0.01)

    def extra_handles():
        plot_moving_goal(time_span, goal_linear)

    # plotting
    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r"),
        (t_dmp_scaled, trj_dmp_scaled[:, 3:6], "-.g")
    ],
        extra_plot_handles=extra_handles,
        savepath='../plots/trj_scalable_moving_goal_3D.pdf'
    )


# TEST OBSTACLES

class Obstacle(object):
    
    def __init__(self):
        self.type = None
        self.center = None
        self.p = None
        self.U = None
    
    @staticmethod
    def point_static(o, beta):
        # The point static potential is formulated as
        #   Us(x) = eta / 2 * (1/rx - 1/r0) ** 2 if rx <= r0 else 0
        # with distance function of a circle
        #   r(x) = ||x - o||
        # We then must compute the force using
        #   p(x,v) = -∇x Us(x)
        r = lambda x: np.linalg.norm(x - o)
        p = lambda x, v: beta / r(x)**(beta+2) * (x - o)
        U = lambda x, v: 1 / r(x)**beta
        
        obs = Obstacle()
        obs.type = "point_static"
        obs.center = o
        obs.p = p
        obs.U = U
        obs.beta = beta
        return obs
    
    @staticmethod
    def point_radial_static(o, eta, r0):
        # The point static potential is formulated as
        #   Us(x) = eta / 2 * (1/rx - 1/r0) ** 2 if rx <= r0 else 0
        # with distance function of a circle
        #   r(x) = ||x - o||
        # We then must compute the force using
        #   p(x,v) = -∇x Us(x)
        r = lambda x: np.linalg.norm(x - o)
        p = lambda x, v: eta * (1 / r(x) - 1 / r0) / r(x)**3 * (x - o) * (r(x) <= r0)
        U = lambda x, v: eta/2 * (1 / r(x) - 1 / r0) ** 2 * (r(x) <= r0)
        
        obs = Obstacle()
        obs.type = "point_radial_static"
        obs.center = o
        obs.p = p
        obs.U = U
        obs.eta = eta
        obs.r0 = r0
        return obs
    
    @staticmethod
    def point_dynamic(o, lambd, beta):
        # The point static potential is formulated as
        #   Ud(x) = lambda * (-cos(th)) ** beta * ||v||/r(x) if th in (pi/2, pi] else 0
        # with distance function of a circle
        #   r(x) = ||x - o||
        # and
        #   cos(th) = (v°x) / (||v||*r(x))
        # We then must compute the force using
        #   p(x,v) = -∇x Us(x)
        norm = lambda a: np.linalg.norm(a)
        r = lambda x: norm(x - o)
        grad_r = lambda x: (x - o) / r(x)
        cos = lambda x, v: np.dot(v, x - o) / (norm(v) * r(x))
        grad_cos = lambda x, v: (r(x) * v - np.dot(v, x - o) * grad_r(x)) / (norm(v) * r(x) ** 2)
        theta_cond = lambda c: np.pi/2 < np.arccos(c) <= np.pi
    
        def potential(x, v):
            if norm(v) == 0:
                p = 0
            else:
                p = lambd * (-cos(x, v)) ** (beta - 1) * norm(v) / r(x) \
                    * (beta * grad_cos(x, v) - cos(x, v) / r(x) * grad_r(x)) * theta_cond(cos(x, v))
            return p
    
        p = lambda x, v: potential(x, v)
        U = lambda x, v: lambd * (-cos(x, v)) ** beta * norm(v) / r(x) * theta_cond(cos(x, v))
        
        obs = Obstacle()
        obs.type = "point_dynamic"
        obs.center = o
        obs.p = p
        obs.U = U
        obs.lambd = lambd
        obs.beta = beta
        return obs


def test_ugly_obstacles():

    # Desired behavior, shaped as a (T x n_dim) array
    radius = 2
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        -radius * (1 - 2*np.cos(time_span) + np.cos(2*time_span)),  # x
        radius * (2*np.sin(time_span) - np.sin(2*time_span))        # y
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    # Add obstacle
    obs1 = Obstacle.point_radial_static(trj_desired[75, :] + np.array([-0.5, 0.5]), eta=750, r0=1.5)
    obs2 = Obstacle.point_dynamic(trj_desired[40, :], lambd=2, beta=2)
    dmp.set_obstacles(obs1, obs2)

    # Execution
    x0 = trj_desired[0, :]
    xgoal = lambda t: trj_desired[-1, :]
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-1, tol=0.1, use_euler=True)

    def extra_handles():
        plot_obstacle_point(obs1.center)
        plot_obstacle_point(obs2.center)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r")
    ],
        extra_plot_handles=extra_handles,
        savepath='../plots/trj_ugly_obstacles.pdf'
    )


def test_obstacles():

    # Desired behavior, shaped as a (T x n_dim) array
    radius = 2
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        -radius * (1 - 2*np.cos(time_span) + np.cos(2*time_span)),  # x
        radius * (2*np.sin(time_span) - np.sin(2*time_span))        # y
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    # Add obstacle
    obs1 = Obstacle.point_radial_static(trj_desired[75, :] + np.array([-0.5, 0.5]), eta=750, r0=1.5)
    obs2 = Obstacle.point_dynamic(trj_desired[40, :], lambd=2, beta=2)
    dmp.set_obstacles(obs1, obs2)

    # Execution
    x0 = trj_desired[0, :]
    xgoal = lambda t: trj_desired[-1, :]
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-1, tol=0.1)

    def extra_handles():
        plot_obstacle_point(obs1.center)
        plot_obstacle_point(obs2.center)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r")
    ],
        extra_plot_handles=extra_handles,
        savepath='../plots/trj_obstacles.pdf'
    )


# TEST FANCY STUFF

def test_fancy_1():

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        np.cos(time_span),  # x
        np.sin(time_span),  # y
        time_span           # z
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=250, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    # Add obstacles
    obs1 = Obstacle.point_dynamic(trj_desired[40, :], lambd=2, beta=2)
    obs2 = Obstacle.point_static(trj_desired[75, :], beta=4)
    dmp.set_obstacles(obs1, obs2)

    # Execution
    def goal_1(t):
        if t < np.pi:
            return trj_desired[-1, :]
        elif np.pi <= t < 1.5*np.pi:
            return 1.2*trj_desired[-1, :]
        else:
            return trj_desired[-1, :]

    def goal_2(t):
        t1 = 1.5*np.pi
        tf = 2*np.pi
        if t < t1:
            return trj_desired[-1, :]
        else:
            return trj_desired[-1, :] - 1.5*(t - t1)/(tf - t1) * np.array([0, 0, 1])

    goal = goal_2

    x0 = trj_desired[0, :]
    xgoal = goal
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-1)

    def extra_handles():
        plot_obstacle_point(obs1.center, show_3D=True)
        plot_obstacle_point(obs2.center, show_3D=True)
        plot_moving_goal(t_dmp, xgoal)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r")
    ],
        extra_plot_handles=extra_handles,
        savepath='../plots/trj_fancy_1.pdf'
    )


def test_fancy_2():

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi, 200)
    trj_desired = np.array([
        2*np.cos(time_span),  # x
        np.sin(time_span),    # y
        np.sqrt(time_span)    # z
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=250, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    # Add obstacles
    obs1 = Obstacle.point_static(trj_desired[75, :], beta=4)
    obs2 = Obstacle.point_dynamic(trj_desired[40, :], lambd=2, beta=2)
    dmp.set_obstacles(obs1, obs2)

    # Execution
    def goal_1(t):
        if t < np.pi:
            return trj_desired[-1, :]
        elif np.pi <= t < 1.5*np.pi:
            return 1.2*trj_desired[-1, :]
        else:
            return trj_desired[-1, :]

    def goal_2(t):
        t1 = 1.5*np.pi
        tf = 2*np.pi
        if t < t1:
            return trj_desired[-1, :]
        else:
            return trj_desired[-1, :] + 0.75*(t - t1)/(tf - t1) * np.array([0, 0, 1])

    def goal_3(t):
        return trj_desired[-1, :]

    goal = goal_3

    x0 = trj_desired[0, :]
    xgoal = goal
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=1e-1)

    def extra_handles():
        plot_obstacle_point(obs1.center, show_3D=True)
        plot_obstacle_point(obs2.center, show_3D=True)
        plot_moving_goal(t_dmp, xgoal)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 3:6], "r")
    ],
        extra_plot_handles=extra_handles,
        savepath='../plots/trj_fancy_2.pdf'
    )


def test_fancy_3():

    # Desired behavior, shaped as a (T x n_dim) array
    time_span = np.linspace(0, 2*np.pi-0.15, 300)
    trj_desired = np.array([
        16*np.sin(time_span)**3,  # x
        13*np.cos(time_span) - 5*np.cos(2*time_span) - 2*np.cos(3*time_span) - np.cos(4*time_span)   # y
    ]).T
    db = Behaviour(time_span, trj_desired)

    # Learning
    dmp = DMP(trj_desired.shape[1], K=150, n_basis=50, alpha=4)
    dmp.learn_trajectory(db)

    # Add obstacles
    obs1 = Obstacle.point_static(trj_desired[75, :], beta=100)
    obs2 = Obstacle.point_radial_static(trj_desired[150, :] + np.array([0, 1.5]), eta=1500, r0=10.0)
    obs3 = Obstacle.point_dynamic(trj_desired[220, :], lambd=50, beta=2)
    dmp.set_obstacles(obs1, obs2, obs3)

    def goal(t):
        t1 = 2.5
        x1 = trj_desired[-1, :]
        p1 = np.array([0, -5])
        
        t2 = 4.0
        x2 = x1 + p1
        p2 = lambda i: 2 * np.array([np.cos(i), -np.sin(i)])
        
        t3 = 5.0
        
        if t < t1:
            g = x1 + p1 * np.clip(0, (t - t1) / (t2 - t1), 1)
        else:
            g = x2 + p2(np.clip(0, (t - t2) / (t3 - t2), 2.5))
        
        return g

    # Execution
    x0 = trj_desired[0, :]
    xgoal = goal
    t_dmp, trj_dmp = dmp.execute_trajectory(x0, xgoal, t_delta=1e-2, tol=0.1)

    def extra_handles():
        plot_obstacle_point(obs1.center, show_3D=False)
        plot_obstacle_point(obs2.center, show_3D=False)
        plot_obstacle_point(obs3.center, show_3D=False)
        plot_moving_goal(t_dmp, xgoal)

    plot_trajectories([
        (time_span, trj_desired, "--b"),
        (t_dmp, trj_dmp[:, 2:4], "r")
    ],
        extra_plot_handles=extra_handles,
        savepath='../plots/trj_fancy_3.pdf'
    )


# OTHER TESTS

def test_estimate_derivatives():
    time_span = np.linspace(0, np.pi, 100)
    trj_desired = np.array([
        time_span,  # x
        np.exp(time_span)  # y
    ]).T

    # real
    x_real = trj_desired[:, 0:2]
    v_real = np.array([
        time_span,          # x
        np.exp(time_span)   # y
    ]).T
    a_real = np.array([
        time_span,          # x
        np.exp(time_span)   # y
    ]).T

    # estimates
    x_est, v_est, a_est = estimate_derivatives(trj_desired[:, 0:2], time_span)

    # compute errors
    x_err = abs(x_real - x_est)
    v_err = abs(v_real - v_est)
    a_err = abs(a_real - a_est)

    plt.figure()
    plt.semilogy(time_span, x_err[:, 1], 'r')
    plt.semilogy(time_span, v_err[:, 1], 'b')
    plt.semilogy(time_span, a_err[:, 1], 'g')
    plt.legend(["x", "v", "a"])
    plt.title("ERROR [log]")
    plt.show()

    plt.figure()
    plt.plot(time_span, x_est[:, 1], 'r')
    plt.plot(time_span, v_est[:, 1], 'b')
    plt.plot(time_span, a_est[:, 1], 'g')
    plt.legend(["x", "v", "a"])
    plt.title("ESTIMATE")
    plt.show()

    plt.figure()
    plt.plot(time_span, x_real[:, 1], 'r')
    plt.plot(time_span, v_real[:, 1], 'b')
    plt.plot(time_span, a_real[:, 1], 'g')
    plt.legend(["x", "v", "a"])
    plt.title("REAL")
    plt.show()
    exit()


def test_RK45():
    from dmp import RK45

    def dynamics(t, s):
        K = 150
        x1 = s[0]
        x2 = s[1]
        dyn = np.array([
            x2,
            -2*np.sqrt(K)*x2 - K*x1
        ])
        return dyn

    s0 = np.ones(shape=(2,))
    s0[1] = 0
    t0 = 0
    dyn = lambda t, z: dynamics(t, z)

    time_span, z_span = RK45(dyn, t0, s0, h_init=10, T=3)

    plt.figure()
    plt.plot(time_span, z_span, 'r')
    plt.show()


# FIXME deprecated
def test_potential():

    # obs, obs_U = obstacle_point_radial_static(o, 4, 1.5)

    xs = 1000
    x = np.linspace(-2, 2, xs)

    limit = 50
    o = np.array([0])
    beta = [0.25, 0.5, 1, 1.5]

    plt.figure()
    for b in beta:
        obs = Obstacle.point_radial_static(o, 1, b)
        zv = np.array([obs.U(xt, 0) for xt in x])
        zv[zv > limit] = limit
        plt.plot(x, zv)

    plt.ylim((-0.5, limit/2))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend([str(b) for b in beta])
    plt.show()


def test_potential_3D_static():
    from matplotlib import cm

    o = np.array([0, 0])
    v = np.array([1, 1])
    obs = Obstacle.point_static(o, 4)

    xs, ys = 200, 200
    x = np.linspace(-5, 5, xs)
    y = np.linspace(-5, 5, ys)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()[:, np.newaxis]
    yv = yv.ravel()[:, np.newaxis]
    X = np.concatenate([xv, yv], axis=1)
    potential = np.array([obs.U(X[i, :], v) for i in range(X.shape[0])])
    xv = xv.reshape((xs, ys))
    yv = yv.reshape((xs, ys))
    zv = potential.reshape((xs, ys))

    limit = 1
    zv[zv > limit] = limit

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(45, -45)
    # ax.view_init(0, 0)
    ax.plot_surface(xv, yv, zv, cmap=cm.plasma, linewidth=0.01, antialiased=True)
    ax.set_zlim((-0.1, limit))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    plt.savefig("../plots/potential_3D_static.pdf", format='pdf', dpi=1200)


def test_potential_3D_radial_static():
    from matplotlib import cm
    
    o = np.array([0, 0])
    v = np.array([1, 1])
    obs = Obstacle.point_radial_static(o, 4, 1.5)
    
    xs, ys = 200, 200
    x = np.linspace(-5, 5, xs)
    y = np.linspace(-5, 5, ys)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()[:, np.newaxis]
    yv = yv.ravel()[:, np.newaxis]
    X = np.concatenate([xv, yv], axis=1)
    potential = np.array([obs.U(X[i, :], v) for i in range(X.shape[0])])
    xv = xv.reshape((xs, ys))
    yv = yv.reshape((xs, ys))
    zv = potential.reshape((xs, ys))
    
    limit = 1
    zv[zv > limit] = limit
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(45, -45)
    # ax.view_init(0, 0)
    ax.plot_surface(xv, yv, zv, cmap=cm.plasma, linewidth=0.01, antialiased=True)
    ax.set_zlim((-0.1, limit))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    plt.savefig("../plots/potential_3D_radial_static.pdf", format='pdf', dpi=1200)


def test_potential_3D_dynamic():
    from matplotlib import cm
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.art3d import Text3D
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return min(zs)
    
    o = np.array([0, 0])
    v = np.array([1, 1])
    obs = Obstacle.point_dynamic(o, 1, 2)
    
    xs, ys = 200, 200
    x = np.linspace(-7, 2, xs)
    y = np.linspace(-7, 2, ys)
    xv, yv = np.meshgrid(x, y)
    xv = xv.ravel()[:, np.newaxis]
    yv = yv.ravel()[:, np.newaxis]
    X = np.concatenate([xv, yv], axis=1)
    potential = np.array([obs.U(X[i, :], v) for i in range(X.shape[0])])
    xv = xv.reshape((xs, ys))
    yv = yv.reshape((xs, ys))
    zv = potential.reshape((xs, ys))
    
    limit = 1
    zv[zv > limit] = limit
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(45, -45)
    # ax.view_init(0, 0)
    ax.plot_surface(xv, yv, zv, cmap=cm.plasma, linewidth=0.01, antialiased=True)
    ax.add_artist(Arrow3D([-6, -4], [-6, -4], [1, 1], mutation_scale=20, lw=3, arrowstyle="-|>", color="k"))
    ax.text(-5.25, -5.25, 1.05, "v", color="k")
    ax.set_zlim((-0.1, limit))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    plt.savefig("../plots/potential_3D_dynamic.pdf", format='pdf', dpi=1200)


if __name__ == "__main__":
    # # VARIOUS TESTS
    # test_estimate_derivatives()
    # test_RK45()
    # test_potential_3D_static()
    # test_potential_3D_radial_static()
    # test_potential_3D_dynamic()
    #
    # # TRAJECTORIES
    test_default_trajectory()
    test_simple_trajectory()
    # test_simple_3D_trajectory()
    # test_discontinuous_3D_trajectory()
    # test_trajectory_robustness(random_start=False)
    # test_trajectory_robustness(random_goal=False)
    # test_ugly_trajectory()
    # test_ugly_3D_trajectory()
    # test_simple_scalable_trajectory()
    # test_simple_3D_scalable_trajectory()
    # test_simple_scalable_trajectory_moving_goal()
    # test_simple_3D_scalable_trajectory_moving_goal()
    # test_ugly_obstacles()
    # test_obstacles()
    #
    # test_fancy_1()
    # test_fancy_2()
    # test_fancy_3()
