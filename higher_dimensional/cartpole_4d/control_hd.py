"""Design a controller from the learned higher-dimensional Koopman model.

Same idea as the 2D study: linearize the learned model at the target with an
autograd Jacobian, design a discrete LQR on that local pair, and apply the state
feedback to the true nonlinear plant. Dimension general.
"""
import numpy as np
import torch
import control as ct

from koopman import KoopmanControl
from sim import simulate_digital_loop
from systems_hd import get_system


def load_koopman(name):
    d = np.load(f"results_hd/koopman_{name}.npz")
    model = KoopmanControl(n_state=int(d["n_state"]), n_extra=int(d["n_extra"]),
                           hidden=int(d["hidden"]), mu=d["mu"], sd=d["sd"])
    model.load_state_dict(torch.load(f"results_hd/koopman_{name}.pt"))
    model.eval()
    return model, d["A"], d["B"], d["sd"], d["mu"]


def local_linear_from_koopman(model, A, B, sd, x_eq):
    """Linearize x_{k+1} = decode(A lift(x) + B u) at (x_eq, u_eq)."""
    n = len(x_eq)
    xeq_t = torch.tensor(x_eq, dtype=torch.float32)
    J = torch.autograd.functional.jacobian(lambda x: model.lift(x), xeq_t).detach().numpy()
    Ddec = np.zeros((n, A.shape[0]))
    Ddec[:n, :n] = np.diag(sd)
    return Ddec @ A @ J, Ddec @ B


def true_discrete(plant, Ts):
    """Exact finite-difference linearization of the true plant, for comparison."""
    n = len(plant.x_eq)
    eps = 1e-6
    Ac = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = eps
        Ac[:, i] = (plant.f(plant.x_eq + e, 0.0) - plant.f(plant.x_eq - e, 0.0)) / (2 * eps)
    Bc = ((plant.f(plant.x_eq, eps) - plant.f(plant.x_eq, -eps)) / (2 * eps)).reshape(n, 1)
    sysd = ct.c2d(ct.ss(Ac, Bc, np.eye(n), np.zeros((n, 1))), Ts, method="zoh")
    return np.asarray(sysd.A), np.asarray(sysd.B)


def lqr(Am, Bm, q, r):
    K, S, _ = ct.dlqr(Am, Bm, np.diag(q), np.array([[r]]))
    return np.asarray(K), np.asarray(S)


Q_CARTPOLE = (1.0, 1.0, 10.0, 1.0)
R_CARTPOLE = 0.1

if __name__ == "__main__":
    Ts = 0.05
    plant = get_system("cartpole")
    model, A, B, sd, mu = load_koopman(plant.name)
    A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
    Ad, Bd = true_discrete(plant, Ts)
    np.set_printoptions(precision=3, suppress=True)
    print("learned local A_loc eig:", np.round(np.linalg.eigvals(A_loc), 3))
    print("true    discrete Ad eig:", np.round(np.linalg.eigvals(Ad), 3))

    K_learn, P = lqr(A_loc, B_loc, Q_CARTPOLE, R_CARTPOLE)
    K_true, _ = lqr(Ad, Bd, Q_CARTPOLE, R_CARTPOLE)
    print("learned K:", np.round(K_learn, 2))
    print("true    K:", np.round(K_true, 2))

    def make_ctrl(K):
        Kf = np.asarray(K).reshape(-1)
        return lambda x: plant.u_eq - float(Kf @ (np.asarray(x) - plant.x_eq))

    starts = [[0.3, 0, 0.1, 0], [0.5, 0, 0.2, 0], [-0.8, 0, 0.3, 0],
              [1.0, 0, 0.3, 0.5], [0.0, 0, 0.4, 0]]
    for umax in (15.0, 20.0, 30.0):
        print(f"\n--- umax = {umax} ---")
        for s in starts:
            x0 = np.array(s, dtype=float)
            _, Xk, _ = simulate_digital_loop(plant, make_ctrl(K_learn), x0, Ts, 160, u_max=umax)
            _, Xt, _ = simulate_digital_loop(plant, make_ctrl(K_true), x0, Ts, 160, u_max=umax)
            ek = np.linalg.norm(Xk[-1] - plant.x_eq)
            et = np.linalg.norm(Xt[-1] - plant.x_eq)
            print(f"  x0={np.array(s)}  learned {'OK ' if ek < 5e-2 else 'MISS'} ({ek:.1e})"
                  f"   true {'OK ' if et < 5e-2 else 'MISS'} ({et:.1e})")
