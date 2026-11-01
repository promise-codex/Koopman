"""Design a controller from the learned planar-quadrotor Koopman model.

Same recipe as the cart-pole study, generalised to two inputs: linearize the
learned model at hover with an autograd Jacobian, design a discrete LQR on that
local pair (now with a two by two input weight), and apply the state feedback to
the true nonlinear plant.
"""
import numpy as np
import torch
import control as ct

from koopman import KoopmanControl
from systems_quad import get_system

# LQR weights: positions and tilt weighted most, velocities lightly damped.
Q_QUAD = (4.0, 4.0, 20.0, 1.0, 1.0, 1.0)
R_QUAD = (0.2, 0.2)


def load_koopman(name):
    d = np.load(f"results_quad/koopman_{name}.npz")
    model = KoopmanControl(n_state=int(d["n_state"]), n_extra=int(d["n_extra"]),
                           hidden=int(d["hidden"]), n_input=int(d["n_input"]), mu=d["mu"], sd=d["sd"])
    model.load_state_dict(torch.load(f"results_quad/koopman_{name}.pt"))
    model.eval()
    return model, d["A"], d["B"], d["sd"], d["mu"]


def local_linear_from_koopman(model, A, B, sd, x_eq):
    """Linearize x_{k+1} = decode(A lift(x) + B u) at the hover point."""
    n = len(x_eq)
    xeq_t = torch.tensor(x_eq, dtype=torch.float32)
    J = torch.autograd.functional.jacobian(lambda x: model.lift(x), xeq_t).detach().numpy()
    Ddec = np.zeros((n, A.shape[0]))
    Ddec[:n, :n] = np.diag(sd)
    return Ddec @ A @ J, Ddec @ B          # B is (z_dim, 2) so B_loc is (6, 2)


def true_discrete(plant, Ts):
    """Exact finite-difference linearization of the true plant, for comparison."""
    n = len(plant.x_eq)
    du = plant.n_input
    eps = 1e-6
    Ac = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = eps
        Ac[:, i] = (plant.f(plant.x_eq + e, plant.u_eq) - plant.f(plant.x_eq - e, plant.u_eq)) / (2 * eps)
    Bc = np.zeros((n, du))
    for j in range(du):
        e = np.zeros(du); e[j] = eps
        Bc[:, j] = (plant.f(plant.x_eq, plant.u_eq + e) - plant.f(plant.x_eq, plant.u_eq - e)) / (2 * eps)
    sysd = ct.c2d(ct.ss(Ac, Bc, np.eye(n), np.zeros((n, du))), Ts, method="zoh")
    return np.asarray(sysd.A), np.asarray(sysd.B)


def lqr(Am, Bm, q, r):
    R = np.diag(r) if np.ndim(r) > 0 else np.array([[float(r)]])
    K, S, _ = ct.dlqr(Am, Bm, np.diag(q), R)
    return np.asarray(K), np.asarray(S)


if __name__ == "__main__":
    Ts = 0.05
    plant = get_system("quadrotor")
    model, A, B, sd, mu = load_koopman(plant.name)
    A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
    Ad, Bd = true_discrete(plant, Ts)
    np.set_printoptions(precision=3, suppress=True)
    print("learned local A_loc eig:", np.round(np.linalg.eigvals(A_loc), 3))
    print("true    discrete Ad eig:", np.round(np.linalg.eigvals(Ad), 3))
    K_learn, P = lqr(A_loc, B_loc, Q_QUAD, R_QUAD)
    K_true, _ = lqr(Ad, Bd, Q_QUAD, R_QUAD)
    print("learned K =\n", np.round(K_learn, 2))
    print("true    K =\n", np.round(K_true, 2))
    Acl = Ad - Bd @ K_learn
    print("closed-loop (true plant, learned K) spectral radius:", np.round(np.abs(np.linalg.eigvals(Acl)).max(), 3))
