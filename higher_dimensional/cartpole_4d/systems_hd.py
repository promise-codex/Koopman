"""Higher-dimensional benchmark systems for the region-of-attraction study.

Same interface as the 2D systems used earlier: each plant exposes f(x, u) that
returns dx/dt, the target equilibrium x_eq and u_eq, state labels, and a
sampling box (per-state half-width) used both for training data and for Monte
Carlo ground truth. In four or more dimensions the state space is too large to
grid, so the ground truth is a large random sample inside this box rather than a
regular lattice.
"""
import numpy as np


class CartPole:
    """Inverted pendulum on a cart. State x = [p, p_dot, theta, theta_dot].

    theta is measured from upright, so theta = 0 is the unstable target, and a
    horizontal force u drives the cart. The equations are the standard
    frictionless point-mass model from the Lagrangian:

        p_ddot     = (u + m l sin(theta) theta_dot^2 - m g sin(theta) cos(theta))
                     / (M + m sin(theta)^2)
        theta_ddot = (g sin(theta) - cos(theta) p_ddot) / l

    with M the cart mass, m the pole mass, l the pole length to the point mass,
    and g gravity. Holding the upright is a four-dimensional balancing task.
    """
    name = "cartpole"
    labels = ["x (m)", "x_dot (m/s)", "theta (rad)", "omega (rad/s)"]
    sample_box = np.array([2.0, 2.0, 1.0, 2.0])

    def __init__(self, M=1.0, m=0.1, l=0.5, g=9.81):
        self.M, self.m, self.l, self.g = M, m, l, g
        self.x_eq = np.array([0.0, 0.0, 0.0, 0.0])
        self.u_eq = 0.0

    def f(self, x, u):
        p, v, th, om = x
        s, c = np.sin(th), np.cos(th)
        den = self.M + self.m * s * s
        pacc = (u + self.m * self.l * s * om * om - self.m * self.g * s * c) / den
        thacc = (self.g * s - c * pacc) / self.l
        return np.array([v, pacc, om, thacc])


SYSTEMS = {"cartpole": CartPole}


def get_system(name):
    return SYSTEMS[name]()


if __name__ == "__main__":
    # self-test: linearize at the target and confirm it is unstable and controllable
    import control as ct
    plant = CartPole()
    n = len(plant.x_eq)
    eps = 1e-6
    A = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = eps
        A[:, i] = (plant.f(plant.x_eq + e, 0.0) - plant.f(plant.x_eq - e, 0.0)) / (2 * eps)
    B = ((plant.f(plant.x_eq, eps) - plant.f(plant.x_eq, -eps)) / (2 * eps)).reshape(n, 1)
    np.set_printoptions(precision=4, suppress=True)
    w = np.linalg.eigvals(A)
    print("continuous A =\n", A)
    print("continuous B =", B.ravel())
    print("eigenvalues:", np.round(w, 4))
    print("unstable (a real part > 0):", bool(np.any(w.real > 1e-6)))
    rank = np.linalg.matrix_rank(ct.ctrb(A, B))
    print(f"controllability rank: {rank} (need {n})")
