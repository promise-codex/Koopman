"""Step 4: discrete LQR, closed around the true nonlinear plant.

Run:  python step4_lqr.py --system duffing --Ts 0.05
"""
import argparse
import os

import control as ct
import matplotlib.pyplot as plt
import numpy as np

from sim import simulate_digital_loop
from step3_linearize import linearize_and_discretize
from systems import get_system

p = argparse.ArgumentParser()
p.add_argument("--system", default="duffing", choices=["duffing", "pendulum"])
p.add_argument("--Ts", type=float, default=0.05)
p.add_argument("--umax", type=float, default=None, help="actuator limit (None = unlimited)")
p.add_argument("--no-show", action="store_true")
args = p.parse_args()

np.set_printoptions(precision=4, suppress=True)
plant = get_system(args.system)
A, B, Ad, Bd = linearize_and_discretize(plant, args.Ts)

# Cost weights: penalize position more than velocity, and effort a little
Q = np.diag([10.0, 1.0])
R = np.array([[0.1]])
K, S, cl_eigs = ct.dlqr(Ad, Bd, Q, R)
K = np.asarray(K)
print(f"LQR gain K = {K}")
print(f"Closed-loop eigenvalues (linear model): {cl_eigs}  |.| = {np.abs(cl_eigs)}")


def controller(x):
    return plant.u_eq - (K @ (x - plant.x_eq)).item()


offsets = [[0.2, 0.0], [-0.5, 0.3], [1.0, 0.0], [-1.5, -1.0]]
n_steps = int(10.0 / args.Ts)

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
summary = {}
for d in offsets:
    x0 = plant.x_eq + np.array(d)
    t, X, U = simulate_digital_loop(plant, controller, x0, args.Ts, n_steps, u_max=args.umax)
    err = np.linalg.norm(X[-1] - plant.x_eq)
    tag = f"x0={np.round(x0, 2)}"
    print(f"{tag:28s} final error = {err:.2e}  max|u| = {np.abs(U).max():.2f}"
          f"  -> {'CONVERGED' if err < 1e-3 else 'did NOT converge'}")
    axes[0].plot(t, X[:, 0], label=tag)
    axes[1].plot(t, X[:, 1])
    axes[2].step(t[:-1], U, where="post")
    summary[tag] = dict(t=t, X=X, U=U)

axes[0].axhline(plant.x_eq[0], color="k", ls="--", lw=0.8)
axes[0].set(ylabel=plant.labels[0], title=f"{plant.name}: digital LQR on the nonlinear plant (Ts={args.Ts}s)")
axes[0].legend(fontsize=8)
axes[1].set(ylabel=plant.labels[1])
axes[2].set(ylabel="u (held for Ts)", xlabel="time (s)")
fig.tight_layout()
out = f"figures/step4_{plant.name}_lqr.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")

# Save the baseline for later comparison against the learned controller
np.savez(f"results/baseline_lqr_{plant.name}.npz", K=K, Ad=Ad, Bd=Bd, Q=Q, R=R, Ts=args.Ts,
         **{f"run{i}_{k}": v for i, s in enumerate(summary.values()) for k, v in s.items()})
print(f"saved results/baseline_lqr_{plant.name}.npz")
if not args.no_show:
    plt.show()
