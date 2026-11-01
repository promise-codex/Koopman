"""Predict the cart-pole (4D) failure region from model features, from a small
fraction of Monte-Carlo tests.

Ground truth is a large random sample of initial states, each labelled recover
or fail by the validated vectorised simulator. A classifier is trained on a
small fraction and predicts the rest. Because the state is four-dimensional, a
regular grid is infeasible (even 25 points per axis is 25^4 = 390,625 starts),
so the data-efficiency of predicting from a fraction is the point.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import ExtraTreesClassifier

from control_hd import load_koopman, local_linear_from_koopman, lqr, Q_CARTPOLE, R_CARTPOLE
from features_hd import model_features, feature_names
from roa_hd import simulate_batch, monte_carlo_roa
from systems_hd import get_system

Ts = 0.05
UMAX = 8.0
BOX = np.array([2.5, 3.0, 0.9, 3.0])
NSTEPS = 160
N = 25000

p = argparse.ArgumentParser()
p.add_argument("--frac", type=float, default=0.15, help="fraction of samples tested")
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--no-show", action="store_true")
args = p.parse_args()

plant = get_system("cartpole")
model, A, B, sd, mu = load_koopman(plant.name)
A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
K, P = lqr(A_loc, B_loc, Q_CARTPOLE, R_CARTPOLE)
Kf = K.reshape(-1)

os.makedirs("results_hd", exist_ok=True)
os.makedirs("figures_hd", exist_ok=True)
cache = "results_hd/roa_mc_cartpole.npz"
if os.path.exists(cache):
    d = np.load(cache)
    X0, y = d["X0"], d["y"]
else:
    print(f"simulating {N} Monte-Carlo starts for ground truth...")
    X0, rec = monte_carlo_roa(Kf, plant, Ts, NSTEPS, UMAX, N, BOX, seed=0)
    y = rec.astype(int)
    np.savez(cache, X0=X0, y=y)
print(f"ground truth: N={len(y)} starts, recover={y.mean()*100:.1f}%, fail={100-y.mean()*100:.1f}%")

feats = model_features(model, A, B, sd, mu, K, P, plant.x_eq, plant.u_eq, X0, UMAX)
names = feature_names(X0.shape[1])


def fit_eval(frac, seed):
    rng = np.random.default_rng(seed)
    Ntot = len(y)
    tr = rng.choice(Ntot, int(frac * Ntot), replace=False)
    mask = np.zeros(Ntot, bool); mask[tr] = True
    clf = ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    clf.fit(feats[mask], y[mask])
    pred = clf.predict(feats[~mask])
    acc = (pred == y[~mask]).mean()
    fs = ((pred == 1) & (y[~mask] == 0)).mean()      # predicted safe, truly fails
    return clf, mask, acc, fs


print("\nlearning curve (accuracy on untested samples, averaged over seeds):")
fracs = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
curve = []
for fr in fracs:
    accs, fss = [], []
    for s in range(args.seeds):
        _, _, a, f = fit_eval(fr, s)
        accs.append(a); fss.append(f)
    curve.append(np.mean(accs))
    print(f"  tested {fr*100:4.1f}% ({int(fr*N):5d} starts):  acc {np.mean(accs)*100:5.1f} +-{np.std(accs)*100:3.1f}   false-safe {np.mean(fss)*100:4.1f}")
curve = np.array(curve)

# headline split + importances + a trained classifier for the slice figure
clf, mask, acc0, fs0 = fit_eval(args.frac, 0)
imp = sorted(zip(names, clf.feature_importances_), key=lambda t: -t[1])
print(f"\nheadline: tested {int(args.frac*N)} of {N} ({args.frac*100:.0f}%), "
      f"accuracy on the rest {acc0*100:.1f}%, false-safe {fs0*100:.1f}%")
print("top features:", ", ".join(f"{n}={v:.2f}" for n, v in imp[:5]))

# ---- 2D slice visualisation of the 4D result (x = x_dot = 0 plane) ----
G = 121
th = np.linspace(-BOX[2], BOX[2], G)
om = np.linspace(-BOX[3], BOX[3], G)
TH, OM = np.meshgrid(th, om)
slice_X0 = np.column_stack([np.zeros(TH.size), np.zeros(TH.size), TH.ravel(), OM.ravel()])
slice_true = (np.linalg.norm(
    simulate_batch(Kf, slice_X0, plant.x_eq, plant.u_eq, Ts, NSTEPS, UMAX) - plant.x_eq, axis=1) < 0.1)
slice_feats = model_features(model, A, B, sd, mu, K, P, plant.x_eq, plant.u_eq, slice_X0, UMAX)
slice_pred = clf.predict(slice_feats).astype(bool)
slice_acc = (slice_pred == slice_true).mean()

g_r = ListedColormap(["#d62728", "#2e8b57"])
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
ax[0].pcolormesh(th, om, slice_true.reshape(G, G).astype(int), cmap=g_r, shading="auto")
ax[0].plot(0, 0, "w*", ms=13, mec="k")
ax[0].set(title=f"true recover (green) / fail (red)\nslice x=0, x_dot=0",
          xlabel="theta (rad)", ylabel="omega (rad/s)")
ax[1].pcolormesh(th, om, slice_pred.reshape(G, G).astype(int), cmap=g_r, shading="auto")
ax[1].plot(0, 0, "w*", ms=13, mec="k")
ax[1].set(title=f"predicted from {args.frac*100:.0f}% of 4D tests\nslice agreement {slice_acc*100:.1f}%",
          xlabel="theta (rad)", ylabel="omega (rad/s)")
ax[2].axhline(95, color="gray", lw=1, ls="--")
ax[2].text(fracs[0]*100, 95.3, "95% (5% error)", fontsize=8, color="gray")
ax[2].plot(np.array(fracs) * 100, curve * 100, "o-", color="#1f77b4")
ax[2].set(title="how many 4D tests are needed?", xlabel="% of samples tested",
          ylabel="% correct on the untested rest", ylim=(min(85, curve.min()*100 - 1), 100))
fig.suptitle("Cart-pole (4D): mapping the controller's failure region from a fraction of the tests")
fig.tight_layout()
out = "figures_hd/cartpole_failure_prediction.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
np.savez("results_hd/failure_prediction_cartpole.npz",
         fracs=fracs, curve=curve, acc=acc0, false_safe=fs0, recover_frac=y.mean())
if not args.no_show:
    pass
