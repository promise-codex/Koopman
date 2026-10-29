"""Step 8: design a controller from the learned model and close the loop.

The learned model is x_{k+1} = decode(A . lift(x) + B u). The procedure:
  1. Linearize the learned model at the target. Its slope at the equilibrium is
         A_loc = Ddec . A . J,   B_loc = Ddec . B
     where J = d(lift)/dx at x_eq (from autograd) and Ddec reads the state back
     out of the lifted vector. No plant equations are used, only the network and
     the A, B learned in step 7.
  2. Design a discrete LQR on (A_loc, B_loc).
  3. Run u = u_eq - K (x - x_eq) on the true nonlinear plant.
  4. Compare against the baseline controller from the exact linearization
     (step 4), trajectory by trajectory and by region of attraction.

The learned model is linearized rather than running an LQR on the full lifted
(A, B) directly. The latter does not stabilize this unstable equilibrium: the
global fit is accurate for open-loop prediction but not sharp enough locally for
feedback. Where the data-driven controller loses ground to the baseline is what
the step 9 predictor estimates in advance.

Run:  python step8_koopman_control.py --system duffing --umax 0.5
      python step8_koopman_control.py --system pendulum --umax 5
"""
import argparse
import os

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from koopman import KoopmanControl
from sim import simulate_digital_loop
from step3_linearize import linearize_and_discretize
from systems import get_system


def load_koopman(name):
    d = np.load(f"results/koopman_{name}.npz")
    model = KoopmanControl(n_state=int(d["n_state"]), n_extra=int(d["n_extra"]),
                           mu=d["mu"], sd=d["sd"])
    model.load_state_dict(torch.load(f"results/koopman_{name}.pt"))
    model.eval()
    return model, d["A"], d["B"], d["sd"]


def local_linear_from_koopman(model, A, B, sd, x_eq):
    """Linearize x_{k+1} = decode(A lift(x) + B u) at (x_eq, u_eq)."""
    n = len(x_eq)
    xeq_t = torch.tensor(x_eq, dtype=torch.float32)
    J = torch.autograd.functional.jacobian(lambda x: model.lift(x), xeq_t).detach().numpy()
    Ddec = np.zeros((n, A.shape[0]))
    Ddec[:n, :n] = np.diag(sd)          # decode slope: normalized state -> real state
    return Ddec @ A @ J, Ddec @ B


def lqr(Amat, Bmat, q=(10.0, 1.0), r=0.1):
    return np.asarray(ct.dlqr(Amat, Bmat, np.diag(q), np.array([[r]]))[0])


def region_of_attraction(plant, controller, Ts, umax, span=3.0, grid=21, t_end=8.0):
    pv = plant.x_eq[0] + np.linspace(-span, span, grid)
    vv = plant.x_eq[1] + np.linspace(-span, span, grid)
    ok = np.zeros((grid, grid))
    steps = int(t_end / Ts)
    for i, v in enumerate(vv):
        for j, pos in enumerate(pv):
            _, X, _ = simulate_digital_loop(plant, controller, [pos, v], Ts, steps, u_max=umax)
            ok[i, j] = np.linalg.norm(X[-1] - plant.x_eq) < 1e-2
    return pv, vv, ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="duffing", choices=["duffing", "pendulum"])
    p.add_argument("--Ts", type=float, default=0.05)
    p.add_argument("--umax", type=float, default=0.5, help="actuator cap (use ~5 for the pendulum)")
    p.add_argument("--grid", type=int, default=21, help="region-of-attraction resolution (finer = smoother boundary)")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    plant = get_system(args.system)
    model, A, B, sd = load_koopman(plant.name)
    n = len(plant.x_eq)

    # controller from the learned model
    A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
    K_koop = lqr(A_loc, B_loc)
    koop_controller = lambda x: plant.u_eq - float((K_koop @ (np.asarray(x) - plant.x_eq)).item())

    # baseline controller from the exact linearization
    _, _, Ad, Bd = linearize_and_discretize(plant, args.Ts)
    K_base = lqr(Ad, Bd)
    base_controller = lambda x: plant.u_eq - float((K_base @ (np.asarray(x) - plant.x_eq)).item())

    np.set_printoptions(precision=4, suppress=True)
    print(f"=== {plant.name} ===")
    print("local A from learned model:\n", A_loc)
    print("true Ad (exact linearization):\n", Ad)
    print(f"|eig| learned {np.round(np.abs(np.linalg.eigvals(A_loc)),3)}  "
          f"true {np.round(np.abs(np.linalg.eigvals(Ad)),3)}")
    print(f"LQR gain  learned K={K_koop[0]}   baseline K={K_base[0]}")

    os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)

    # trajectory comparison
    offsets = [[0.3, 0.0], [-0.6, 0.3], [0.9, 0.0]]
    steps = int(10.0 / args.Ts)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    print(f"\nclosed loop on the true nonlinear plant (|u| <= {args.umax}):")
    for d in offsets:
        x0 = plant.x_eq + np.array(d)
        t, Xk, _ = simulate_digital_loop(plant, koop_controller, x0, args.Ts, steps, u_max=args.umax)
        _, Xb, _ = simulate_digital_loop(plant, base_controller, x0, args.Ts, steps, u_max=args.umax)
        ek, eb = np.linalg.norm(Xk[-1] - plant.x_eq), np.linalg.norm(Xb[-1] - plant.x_eq)
        print(f"  x0={np.round(x0,2)}  learned {'OK ' if ek<1e-2 else 'MISS'} ({ek:.1e})"
              f"   baseline {'OK ' if eb<1e-2 else 'MISS'} ({eb:.1e})")
        c = axes[0].plot(t, Xk[:, 0], label=f"learned, x0={np.round(x0,2)}")[0]
        axes[0].plot(t, Xb[:, 0], "--", color=c.get_color(), alpha=0.6)
    axes[0].axhline(plant.x_eq[0], color="k", lw=0.7)
    axes[0].set(ylabel=plant.labels[0], xlabel="time (s)",
                title=f"{plant.name}: learned-model controller (solid) vs baseline (dashed)")
    axes[0].legend(fontsize=8)

    # region of attraction comparison (cached so reruns are instant)
    cache = f"results/koopman_control_{plant.name}.npz"
    pv = None
    if os.path.exists(cache):
        d = np.load(cache)
        if d["p"].size == args.grid and float(d["umax"]) == args.umax:
            pv, vv, ok_k, ok_b = d["p"], d["v"], d["roa_learned"], d["roa_baseline"]
    if pv is None:
        print("mapping region of attraction (this takes ~30-60 s)...")
        pv, vv, ok_k = region_of_attraction(plant, koop_controller, args.Ts, args.umax, grid=args.grid)
        _, _, ok_b = region_of_attraction(plant, base_controller, args.Ts, args.umax, grid=args.grid)
    print(f"region of attraction: learned {ok_k.mean()*100:.0f}%   baseline {ok_b.mean()*100:.0f}%")
    PV, VV = np.meshgrid(pv, vv)
    dis = ok_k.astype(bool) != ok_b.astype(bool)
    g_r = ListedColormap(["#d62728", "#2e8b57"])
    axes[1].pcolormesh(pv, vv, ok_k.astype(int), cmap=g_r, shading="auto", vmin=0, vmax=1)
    # overlay the exact-equation baseline recover/fail boundary (clean at any resolution)
    axes[1].contour(PV, VV, ok_b.astype(float), levels=[0.5], colors="white", linewidths=1.5)
    axes[1].plot([], [], color="white", lw=1.6, label="exact-equation baseline boundary")
    if args.grid <= 35 and dis.any():          # per-cell marks only legible on a coarse grid
        axes[1].scatter(PV[dis], VV[dis], s=45, facecolors="none", edgecolors="black",
                        linewidths=1.2, label="differs from baseline")
    axes[1].plot(*plant.x_eq, "w*", ms=13, mec="k")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set(xlabel=plant.labels[0], ylabel=plant.labels[1],
                title=f"learned controller: recovers (green) / fails (red);  white = exact-equation baseline\n"
                      f"region of attraction: {ok_k.mean()*100:.0f}% (learned) vs {ok_b.mean()*100:.0f}% (baseline)")
    fig.tight_layout()
    out = f"figures/step8_{plant.name}_koopman_control.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    np.savez(f"results/koopman_control_{plant.name}.npz",
             K=K_koop, A_loc=A_loc, B_loc=B_loc, roa_learned=ok_k, roa_baseline=ok_b,
             p=pv, v=vv, umax=args.umax)
    print(f"saved results/koopman_control_{plant.name}.npz")
    if not args.no_show:
        plt.show()
