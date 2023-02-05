# README #

Repository for the Modeling Week 2022 at University of Verona.

## (Quick) DMP Overview ##

### Formulation ###

The term _Dynamic Movement Primitives_ (DMPs) refers to a framework for trajectory learning based on a second order ODE of spring-mass-damper type:
$$ \begin{cases}
    \tau \dot{\mathbf{v}} = \mathbf{K} (\mathbf{g} - \mathbf{x}) - \mathbf{D} \mathbf{v} - \mathbf{K} ( \mathbf{g} - \mathbf{x}_0 ) s + \mathbf{K} \mathbf{f}(s) \\
    \tau \dot{\mathbf{x}} = \mathbf{v}
\end{cases} , $$
where
* $\mathbf{x} \in \mathbb{R}^d$ is the position of the system;
* $\mathbf{v} \in \mathbb{R}^d$ is the velocity of the system;
* $\mathbf{g} \in \mathbb{R}^d$ is the goal position (i.e. the desired final position);
* $\mathbf{x_0} \in \mathbb{R}^d$ is the initial position;
* $\mathbf{f} \in \mathbb{R}^d$ is a non-linear forcing term;
* parameter $s \in \mathbb{R}$ is a re-parametrization of time, governed by the _Canonical System_
$$ \tau \dot{s} = -\alpha s, \qquad \alpha > 0. $$
* $\mathbf{K, D} \in \mathbb{R}^{n \times n}$ are diagonal matrices
$$
\mathbf{K} =
\begin{bmatrix}
K_1 & 0   & \cdots & 0 \\
0   & K_2 & \ddots & \vdots \\
\vdots & \ddots & \ddots & 0 \\
0 & \cdots & 0 & K_n
\end{bmatrix}, 
\qquad
\mathbf{D} =
\begin{bmatrix}
D_1 & 0   & \cdots & 0 \\
0   & D_2 & \ddots & \vdots \\
\vdots & \ddots & \ddots & 0 \\
0 & \cdots & 0 & D_n
\end{bmatrix}, 
$$
representing the elastic and damping terms.
To ensure the critical damping of the system, we fix $D_i = 2\sqrt{K_i}$.
Usually, these matrices are set to be a multiple of the identity matrix,
$$ \mathbf{K} = K \, \mathbf{I}_n ,\quad \mathbf{D} = D \, \mathbf{I}_n , $$
with $D = 2\sqrt{K}$.

Forcing term $\mathbf{f}$ is written in terms of _basis functions_. Each component $f_j (s)$ is written as
$$ f_j(s) = \frac{\sum_{i=0}^N \omega_i \psi_i(s)}{\sum_{i=0}^N \psi_i(s)} s , $$
where $\omega_i\in\mathbb{R}$ and $\{\psi_i(s)\}_{i=0}^N$ is a set of basis functions.
In the literature, _Ragial Gaussian basis functions_ are used: given a set of centers $\{c_i\}_{i=0}^N$ and a set of positive widths $\{h_i\}_{i=1}^N$, we have
$$ \psi_i(s) = \exp( -h_i (s - c_i)^2 ). $$

### Learning and Execution ###

During the _learning phase_, a trajectory $\mathbf{x}(t)$ is recorded. This permits to evaluate the forcing term $\mathbf{f}(s)$.
Then, the set of weights $\omega_i$ is computed using linear regression:

$$ {\omega}^\star = \arg \min_{\omega} \left\Vert {\sum_{i=0}^N \omega_i \psi_i(s) \over \sum_{i=0}^N \psi_i(s) }s - \tilde{f}(s) \right\Vert , $$

where $\tilde{f}(s)$ is the forcing term obtained from the desired trajectory

$$ \tilde{f}(s) = {\ddot{x}(s) + D \dot{x}(s) \over K} - (g - x(s)) + (g - x_0) s .$$

We remark that this learning process must be performed for each Cartesian component of the space $\mathbb{R}^n$.

Then, during the _execution phase_, the dynamical system can be integrated using the weights to generate the forcing term, but possibly changing the initial and goal position.
This will result in a trajectory of similar shape to the learned one, but adapted to the new points.
Moreover, the goal position can change during the execution and convergence to it is still guaranteed.

## Task for the Modelling Week ##
At the end of the week, you should be able to create a DMP object.
You should be able to learn a trajectory, and execute it, with the possibility to change starting and goal position.
In the `code/` folder you can find a draft of the expected result.
Some class methods have already been defined, you may add all the methods you want.
At the end of `code/dmp.py` a demo is already implemented.
It is expected that the demo works by the end of the week.

If you manage to implement the DMP framework before the end of the week, you may try tackle some improvements such as:

* give the user the possibility to choose the family of basis functions
* implement one or more of the methods for obstacle avoidace
* implement the affine-invariance of DMPs

## References ##

* IJSPEERT, Auke; NAKANISHI, Jun; SCHAAL, Stefan. Learning attractor landscapes for learning motor primitives. Advances in neural information processing systems, 2002, 15.
* PARK, Dae-Hyung, et al. Movement reproduction and obstacle avoidance with dynamic movement primitives and potential fields. In: Humanoids 2008-8th IEEE-RAS International Conference on Humanoid Robots. IEEE, 2008. p. 91-98.
* HOFFMANN, Heiko, et al. Biologically-inspired dynamical systems for movement generation: Automatic real-time goal adaptation and obstacle avoidance. In: 2009 IEEE International Conference on Robotics and Automation. IEEE, 2009. p. 2587-2592.
* GINESI, Michele, et al. Dynamic movement primitives: Volumetric obstacle avoidance. In: 2019 19th international conference on advanced robotics (ICAR). IEEE, 2019. p. 234-239.
* GINESI, Michele, et al. Dynamic movement primitives: Volumetric obstacle avoidance using dynamic potential functions. Journal of Intelligent & Robotic Systems, 2021, 101.4: 1-20.
* GINESI, Michele; SANSONETTO, Nicola; FIORINI, Paolo. Overcoming some drawbacks of dynamic movement primitives. Robotics and Autonomous Systems, 2021, 144: 103844.



```python

```
