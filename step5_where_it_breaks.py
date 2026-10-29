"""Step 5: map where the LQR baseline stops working (the region of attraction).

Runs the baseline controller from a grid of initial states and records which
ones return to the target. This map is the ground truth that the failure
predictor (step 9) later reproduces from the learned model alone.

Run:  python step5_where_it_breaks.py --system duffing --umax 0.5
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
p.add_argument("--umax", type=float, default=0.5)
p.add_argument("--grid", type=int, default=25)
p.add_argument("--no-show", action="store_true")
args = p.parse_args()

plant = get_system(args.system)
_, _, Ad, Bd = linearize_and_discretize(plant, args.Ts)
K = np.asarray(ct.dlqr(Ad, Bd, np.diag([10.0, 1.0]), np.array([[0.1]]))[0])
controller = lambda x: plant.u_eq - (K @ (x - plant.x_eq)).item()

span = 3.0
p_vals = plant.x_eq[0] + np.linspace(-span, span, args.grid)
v_vals = plant.x_eq[1] + np.linspace(-span, span, args.grid)
ok = np.zeros((args.grid, args.grid))
n_steps = int(8.0 / args.Ts)
for i, v in enumerate(v_vals):
    for j, pos in enumerate(p_vals):
        _, X, _ = simulate_digital_loop(plant, controller, [pos, v], args.Ts, n_steps, u_max=args.umax)
        ok[i, j] = np.linalg.norm(X[-1] - plant.x_eq) < 1e-2
    print(f"row {i + 1}/{args.grid} done", end="\r")

print(f"\n{ok.mean() * 100:.0f}% of tested initial conditions converged (umax={args.umax})")
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
np.savez(f"results/roa_lqr_{plant.name}.npz", p=p_vals, v=v_vals, ok=ok, umax=args.umax, Ts=args.Ts)

plt.figure(figsize=(6, 5))
plt.pcolormesh(p_vals, v_vals, ok, cmap="RdYlGn", shading="auto", vmin=0, vmax=1)
plt.plot(*plant.x_eq, "k*", ms=12)
plt.xlabel(plant.labels[0]); plt.ylabel(plant.labels[1])
plt.title(f"{plant.name}: where LQR works (green) / fails (red)\nTs={args.Ts}s, |u|<={args.umax}")
plt.tight_layout()
out = f"figures/step5_{plant.name}_roa.png"
plt.savefig(out, dpi=130)
print(f"saved {out}")
if not args.no_show:
    plt.show()
