"""Step 9: predict the controller's failure region from the learned model.

Step 8 maps the true recover/fail region by running the controller from a grid
of initial states, which is expensive. Here only a small fraction of those
states are tested. Each state is described by cheap features taken from the
learned model (no extra plant runs), and a classifier predicts the outcome for
the untested states. This reconstructs most of the failure region from about a
third of the tests.

Design choices behind the sub-5% error on both systems:
  1. A model-only feature set: the model's own closed-loop behaviour at several
     horizons (2, 5, 10, 20 steps), how close it comes to the target, how often
     the actuator saturates, and the growth rate, together with the offset,
     control demand, Lyapunov value, and lifted-norm features.
  2. A fine ground-truth grid. On a coarse 21x21 grid the Duffing basin boundary
     is ragged and caps accuracy near 92%, because boundary cells are a large
     fraction of all cells. Refining the grid resolves the boundary (boundary
     scales as N, total as N^2, so the boundary fraction shrinks) and provides
     more data. Accuracy rises and then stabilizes as the grid is refined from
     51 to 81, so the result is not a grid artefact.
  3. ExtraTrees over several random train/test splits, reporting the average so
     the figure does not rest on one split.

Two approaches that do not succeed on their own are reused here as features.
Simulating the controller inside the learned model diverges, because the
closed-loop inputs differ from the random training inputs. A Lyapunov sublevel
set is provably invariant but small. Both still carry useful signal, and the
Lyapunov set is drawn as a safe core inside the prediction.

Scope:
  * The classifier map is a statistical estimate with a measured error, not a
    proof. The one hard guarantee is the Lyapunov core, and it is rigorous for
    the linearized learned model, so it is labelled as such rather than as a
    certificate for the true nonlinear plant. It sits well inside the
    empirically safe region, which is a consistency check.
  * The method predicts untested gaps from tests scattered across the space
    (interpolation), where it is strong: on an 81x81 grid, a random 35% tested
    gives about 97.3% on Duffing and 98.6% on the pendulum, and a coarse regular
    lattice (about 26% tested) gives about 95% and 99%. Pure extrapolation into
    an unseen region is weaker (about 89 to 92%), which is the stated
    limitation: sample throughout the operating envelope rather than one corner.
    The three evaluation protocols are in validate_predictor.py.

Run:  python step9_predict_failure.py --system duffing --umax 0.5
      python step9_predict_failure.py --system pendulum --umax 5
(The first run regenerates the fine ground-truth map, about 2 minutes, then
caches it.)
"""
import argparse
import os

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.ensemble import ExtraTreesClassifier

from step8_koopman_control import (load_koopman, local_linear_from_koopman, lqr,
                                    region_of_attraction)
from systems import get_system

CLIP = 50.0                     # bound the (diverging) model rollout
HORIZONS = (2, 5, 10, 20)
FEATURE_NAMES = ["offset_x", "offset_v", "|offset|", "push", "push/umax",
                 "lyapunov", "||lift||", "run@2", "run@5", "run@10", "run@20",
                 "min_approach", "sat_frac", "growth"]


def ground_truth(plant, controller, Ts, umax, grid):
    """Fine recover/fail map, cached so reruns are instant."""
    cache = f"results/roa_g{grid}_{plant.name}.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        if int(d["grid"]) == grid and float(d["umax"]) == umax:
            return d["p"], d["v"], d["roa"].astype(bool)
    print(f"mapping the true recover/fail region on a {grid}x{grid} grid "
          f"(~2 min, cached after)...")
    pv, vv, ok = region_of_attraction(plant, controller, Ts, umax, span=3.0,
                                      grid=grid, t_end=8.0)
    os.makedirs("results", exist_ok=True)
    np.savez(cache, p=pv, v=vv, roa=ok, grid=grid, umax=umax)
    return pv, vv, ok.astype(bool)


def model_features(model, A, B, sd, mu, K, P, x_eq, u_eq, pv, vv, umax):
    """Cheap per-state descriptors taken only from the learned model."""
    PV, VV = np.meshgrid(pv, vv)
    X0 = np.stack([PV.ravel(), VV.ravel()], axis=1).astype(np.float32)
    E = X0 - x_eq
    Kf = K.flatten()
    push = np.abs(E @ Kf)                                  # initial control demand
    lyap = np.einsum("ij,jk,ik->i", E, P, E)               # Lyapunov value e'Pe
    with torch.no_grad():
        liftnorm = np.linalg.norm(model.lift(torch.from_numpy(X0)).numpy(), axis=1)
    # the model's own closed loop over several horizons
    Brow = B.reshape(1, -1)
    X = X0.copy()
    min_app = np.full(len(X0), np.inf)
    sat = np.zeros(len(X0))
    norms = {}
    for step in range(1, max(HORIZONS) + 1):
        u = np.clip(u_eq - (X - x_eq) @ Kf, -umax, umax)
        sat += (np.abs(np.abs(u) - umax) < 1e-6)
        with torch.no_grad():
            Z = model.lift(torch.from_numpy(X)).numpy()
        X = (Z @ A.T + u[:, None] * Brow)[:, :2] * sd + mu
        X = np.clip(X, x_eq - CLIP, x_eq + CLIP).astype(np.float32)
        d = np.linalg.norm(X - x_eq, axis=1)
        min_app = np.minimum(min_app, d)
        if step in HORIZONS:
            norms[step] = np.log1p(d)
    feats = np.column_stack([
        E[:, 0], E[:, 1], np.linalg.norm(E, axis=1),
        push, np.minimum(push / umax, 3.0), lyap, liftnorm,
        norms[2], norms[5], norms[10], norms[20],
        np.log1p(min_app), sat / max(HORIZONS), norms[20] - norms[2]])
    return feats, lyap.reshape(PV.shape), PV, VV


def fit_predict(feats, y, test_frac, seed):
    N = len(y)
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(N, size=int(test_frac * N), replace=False)
    mask = np.zeros(N, bool); mask[train_idx] = True
    clf = ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    clf.fit(feats[mask], y[mask])
    pred = clf.predict(feats).astype(bool)
    held = ~mask
    acc = (pred[held] == y[held].astype(bool)).mean()
    false_safe = (pred[held] & ~y[held].astype(bool)).mean()
    return clf, pred, mask, acc, false_safe


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="duffing", choices=["duffing", "pendulum"])
    p.add_argument("--Ts", type=float, default=0.05)
    p.add_argument("--umax", type=float, default=0.5)
    p.add_argument("--grid", type=int, default=81, help="ground-truth resolution (finer = smoother boundary)")
    p.add_argument("--test-frac", type=float, default=0.35, help="fraction of states actually tested")
    p.add_argument("--seeds", type=int, default=5, help="random splits to average the reported accuracy over")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    plant = get_system(args.system)
    model, A, B, sd = load_koopman(plant.name)
    mu = np.load(f"results/koopman_{plant.name}.npz")["mu"]

    A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
    K, S, _ = ct.dlqr(A_loc, B_loc, np.diag([10.0, 1.0]), np.array([[0.1]]))
    K, P = np.asarray(K), np.asarray(S)
    controller = lambda x: plant.u_eq - float((K @ (np.asarray(x) - plant.x_eq)).item())

    pv, vv, true_ok = ground_truth(plant, controller, args.Ts, args.umax, args.grid)
    feats, lyap_grid, PV, VV = model_features(model, A, B, sd, mu, K, P,
                                              plant.x_eq, plant.u_eq, pv, vv, args.umax)
    y = true_ok.ravel().astype(int)
    N = len(y)

    # average accuracy over several random train/test splits
    accs, fsafes = [], []
    for s in range(args.seeds):
        _, _, _, a, f = fit_predict(feats, y, args.test_frac, s)
        accs.append(a); fsafes.append(f)
    acc_mean, acc_std, fsafe_mean = np.mean(accs), np.std(accs), np.mean(fsafes)

    # one split to draw the maps and feature importances
    clf, pred, mask, acc0, _ = fit_predict(feats, y, args.test_frac, 0)
    n_test = int(args.test_frac * N)
    print(f"=== {plant.name}  ({args.grid}x{args.grid} = {N} states) ===")
    print(f"tested {n_test} ({args.test_frac*100:.0f}%), predicted the other {N-n_test}")
    print(f"accuracy on untested states: {acc_mean*100:.1f}% +- {acc_std*100:.1f}  "
          f"(error {100-acc_mean*100:.1f}%, over {args.seeds} splits)")
    print(f"  dangerous errors (predicted safe, truly fails): {fsafe_mean*100:.1f}%")
    imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda t: -t[1])
    print("  top features:", ", ".join(f"{n}={v:.2f}" for n, v in imp[:4]))

    # learning curve: accuracy vs fraction of states tested (seed-averaged)
    fracs = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45])
    curve = []
    for fr in fracs:
        curve.append(np.mean([fit_predict(feats, y, fr, s)[3] for s in range(args.seeds)]))
    curve = np.array(curve)

    c_cert = (args.umax**2 / (K @ np.linalg.inv(P) @ K.T)).item()   # Lyapunov safe core (linear model)
    pred_grid = pred.reshape(PV.shape)
    os.makedirs("figures", exist_ok=True)
    g_r = ListedColormap(["#d62728", "#2e8b57"])
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))

    ax[0].pcolormesh(pv, vv, true_ok.astype(int), cmap=g_r, shading="auto")
    ax[0].plot(*plant.x_eq, "w*", ms=13, mec="k")
    ax[0].set(title=f"truth: tested every one of {N} states",
              xlabel=plant.labels[0], ylabel=plant.labels[1])

    ax[1].pcolormesh(pv, vv, pred_grid.astype(int), cmap=g_r, shading="auto")
    tp = np.stack([PV.ravel()[mask], VV.ravel()[mask]], axis=1)
    ax[1].scatter(tp[:, 0], tp[:, 1], s=3, c="k", alpha=0.35, label=f"{n_test} tested ({args.test_frac*100:.0f}%)")
    ax[1].contour(PV, VV, lyap_grid, levels=[c_cert], colors="cyan", linewidths=2)
    ax[1].plot([], [], color="cyan", lw=2, label="Lyapunov safe core (linear model)")
    ax[1].plot(*plant.x_eq, "w*", ms=13, mec="k")
    ax[1].set(title=f"predicted from {args.test_frac*100:.0f}% of tests  "
                    f"({acc_mean*100:.1f}% correct on the rest)",
              xlabel=plant.labels[0], ylabel=plant.labels[1])
    ax[1].legend(fontsize=8, loc="upper right")

    ax[2].axhline(95, color="gray", lw=1, ls="--")
    ax[2].text(fracs[0]*100, 95.3, "95% (5% error)", fontsize=8, color="gray")
    ax[2].plot(fracs * 100, curve * 100, "o-", color="#1f77b4")
    ax[2].plot(args.test_frac * 100, acc_mean * 100, "o", ms=11, mfc="none",
               mec="crimson", mew=2, label="chosen")
    ax[2].set(title="how many states must be tested?", xlabel="% of states tested",
              ylabel="% correct on the untested rest", ylim=(min(88, curve.min()*100-1), 100))
    ax[2].legend(fontsize=8, loc="lower right")

    fig.suptitle(f"{plant.name}: mapping the controller's failure region from a fraction of the tests")
    fig.tight_layout()
    out = f"figures/step9_{plant.name}_failure_prediction.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    np.savez(f"results/failure_prediction_{plant.name}.npz",
             pred=pred_grid, true=true_ok, tested=mask.reshape(PV.shape),
             p=pv, v=vv, accuracy=acc_mean, accuracy_std=acc_std, false_safe=fsafe_mean,
             fracs=fracs, curve=curve, c_cert=c_cert, grid=args.grid, test_frac=args.test_frac)
    if not args.no_show:
        plt.show()
