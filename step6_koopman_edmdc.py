"""Step 6: classical Koopman with control (EDMDc).

Uses a fixed set of nonlinear observables chosen by hand (powers of the state,
plus sin/cos for the pendulum), stacks every observed one-step transition, and
solves a single least-squares problem for the best linear model
z_{k+1} = A z_k + B u_k in that lifted space.

EDMDc is the baseline for the neural model in step 7. It needs only numpy, and
it confirms that the data and the predict-and-compare pipeline are correct. The
neural model's task is to learn better observables than the fixed ones used here.

Run:  python step6_koopman_edmdc.py --system duffing
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from data import make_dataset
from systems import get_system


def psi(X):
    """Fixed nonlinear observables. The first two are the raw state, so decoding
    is just reading off the first two numbers."""
    x, v = X[..., 0], X[..., 1]
    return np.stack([x, v, x**2, x * v, v**2, x**3, np.sin(x), np.cos(x)], axis=-1)


def fit_edmdc(X, U):
    Xk = X[:, :-1, :].reshape(-1, 2)
    Xk1 = X[:, 1:, :].reshape(-1, 2)
    Uk = U.reshape(-1, 1)
    Z, Zp = psi(Xk), psi(Xk1)
    m = Z.shape[1]
    G = np.hstack([Z, Uk])
    K = Zp.T @ np.linalg.pinv(G.T)   # solve Zp' = K G'
    return K[:, :m], K[:, m:]         # A, B


def rollout(A, B, x0, u_seq):
    z = psi(x0)
    xs = [z[:, :2]]
    for k in range(u_seq.shape[1]):
        z = z @ A.T + u_seq[:, k, :] @ B.T
        xs.append(z[:, :2])
    return np.stack(xs, axis=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="duffing", choices=["duffing", "pendulum"])
    p.add_argument("--Ts", type=float, default=0.05)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    plant = get_system(args.system)
    print("generating trajectories (runs the nonlinear simulator)...")
    X, U = make_dataset(plant, n_traj=400, horizon=30, Ts=args.Ts, seed=0)
    Xv, Uv = make_dataset(plant, n_traj=100, horizon=30, Ts=args.Ts, seed=999)

    A, B = fit_edmdc(X, U)
    Xpred = rollout(A, B, Xv[:, 0, :], Uv)
    rmse = np.sqrt(((Xpred - Xv) ** 2).mean())
    naive = np.sqrt(((Xv[:, :1, :] - Xv) ** 2).mean())
    print(f"lifted dim = {A.shape[0]}  (2 state + 6 hand-picked observables)")
    print(f"EDMDc multi-step RMSE = {rmse:.4f}   (naive 'no change' baseline = {naive:.4f})")
    print(f"the lifted model is {naive / rmse:.1f}x better than predicting no motion")

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    np.savez(f"results/koopman_edmdc_{plant.name}.npz", A=A, B=B, Ts=args.Ts)

    x_true, x_pred = Xv[0], Xpred[0]
    t = np.arange(X.shape[1]) * args.Ts
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for d, ax in enumerate(axes):
        ax.plot(t, x_true[:, d], "-", lw=2, label="true (nonlinear sim)")
        ax.plot(t, x_pred[:, d], "--", lw=2, label="EDMDc linear prediction")
        ax.set_ylabel(plant.labels[d]); ax.legend(fontsize=8)
    axes[0].set_title(f"{plant.name}: classic Koopman (EDMDc) prediction vs truth")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    out = f"figures/step6_{plant.name}_edmdc_prediction.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    if not args.no_show:
        plt.show()
