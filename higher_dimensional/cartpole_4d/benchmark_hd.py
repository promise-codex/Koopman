"""Benchmark the predictor against existing region-of-attraction approaches.

All methods see the same cart-pole ground truth and the same test budget. The
question is whether the model-feature representation beats the alternatives:
  - model features + ExtraTrees        (this work)
  - raw state + ExtraTrees             (same classifier, raw coordinates)
  - raw state + SVM (RBF)              (the HAL-style classifier on the state)
  - raw state + Gaussian process       (the published GP-classification approach)
  - quadratic Lyapunov sublevel set    (classical estimate from the local LQR)

Reported on the held-out states: accuracy and the safety-critical false-safe
rate, at a fixed test budget, averaged over seeds.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler

from control_hd import load_koopman, local_linear_from_koopman, lqr, Q_CARTPOLE, R_CARTPOLE
from features_hd import model_features
from systems_hd import get_system

BUDGET = 3750          # 15% of 25000
GP_CAP = 1500          # GP training is O(n^3); cap its training set
SEEDS = 5
GP_SEEDS = 3

d = np.load("results_hd/roa_mc_cartpole.npz")
X0, y = d["X0"], d["y"]
plant = get_system("cartpole")
model, A, B, sd, mu = load_koopman(plant.name)
A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
K, P = lqr(A_loc, B_loc, Q_CARTPOLE, R_CARTPOLE)
feats = model_features(model, A, B, sd, mu, K, P, plant.x_eq, plant.u_eq, X0, 8.0)
V = np.einsum("ij,jk,ik->i", X0 - plant.x_eq, P, X0 - plant.x_eq)   # quadratic Lyapunov value
N = len(y)


def split(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    return perm[:BUDGET], perm[BUDGET:]


def scores(pred, ytrue):
    acc = (pred == ytrue).mean()
    fs = ((pred == 1) & (ytrue == 0)).sum() / max((pred == 1).sum(), 1)
    return acc * 100, fs * 100


def run_clf(make, Xrep, seeds, standardize=False, cap=None):
    accs, fss = [], []
    for s in range(seeds):
        tr, te = split(s)
        if cap is not None and len(tr) > cap:
            tr = tr[:cap]
        Xtr, Xte = Xrep[tr], Xrep[te]
        if standardize:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = make(s).fit(Xtr, y[tr])
        a, f = scores(clf.predict(Xte), y[te])
        accs.append(a); fss.append(f)
    return np.mean(accs), np.std(accs), np.mean(fss)


def run_lyapunov(seeds, purity=0.99):
    accs, fss = [], []
    for s in range(seeds):
        tr, te = split(s)
        order = tr[np.argsort(V[tr])]
        rec = (y[order] == 1).astype(float)
        run_purity = np.cumsum(rec) / np.arange(1, len(order) + 1)
        ok = np.where(run_purity >= purity)[0]
        c = V[order[ok[-1]]] if len(ok) else -1.0
        pred = (V[te] <= c).astype(int)
        a, f = scores(pred, y[te])
        accs.append(a); fss.append(f)
    return np.mean(accs), np.std(accs), np.mean(fss)


print(f"cart-pole 4D, test budget {BUDGET} ({BUDGET/N*100:.0f}%), recover rate {y.mean()*100:.1f}%\n")
print(f"{'method':38s} {'accuracy':>14s}  {'false-safe':>10s}")
rows = []

a, s_, f = run_clf(lambda s: ExtraTreesClassifier(n_estimators=400, random_state=s, n_jobs=-1), feats, SEEDS)
rows.append(("model features + ExtraTrees (this work)", a, s_, f))
a, s_, f = run_clf(lambda s: ExtraTreesClassifier(n_estimators=400, random_state=s, n_jobs=-1), X0, SEEDS)
rows.append(("raw state + ExtraTrees", a, s_, f))
a, s_, f = run_clf(lambda s: SVC(C=10.0, gamma="scale"), X0, SEEDS, standardize=True)
rows.append(("raw state + SVM (RBF)", a, s_, f))
a, s_, f = run_clf(lambda s: GaussianProcessClassifier(ConstantKernel(1.0) * RBF(1.0), n_restarts_optimizer=0, max_iter_predict=100),
                   X0, GP_SEEDS, standardize=True, cap=GP_CAP)
rows.append((f"raw state + Gaussian process (train<={GP_CAP})", a, s_, f))
a, s_, f = run_lyapunov(SEEDS)
rows.append(("quadratic Lyapunov sublevel set", a, s_, f))

for name, a, s_, f in rows:
    print(f"{name:38s}   {a:5.1f} +-{s_:3.1f} %   {f:6.1f} %")

# figure
import matplotlib.pyplot as plt
labels = ["model feats\n+ ExtraTrees\n(this work)", "raw state\n+ ExtraTrees", "raw state\n+ SVM",
          "raw state\n+ GP", "quadratic\nLyapunov"]
accs = [r[1] for r in rows]; errs = [r[2] for r in rows]; fss = [r[3] for r in rows]
x = np.arange(len(labels))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
colors = ["#2e8b57", "#1f77b4", "#1f77b4", "#1f77b4", "#7f7f7f"]
ax[0].bar(x, accs, yerr=errs, color=colors, capsize=3)
ax[0].set(title="accuracy on untested states", ylabel="% correct", ylim=(50, 100))
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
ax[1].bar(x, fss, color=colors)
ax[1].set(title="false-safe rate (lower is safer)", ylabel="%")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
fig.suptitle(f"Cart-pole: predictor vs existing ROA approaches (test budget {BUDGET})")
fig.tight_layout()
fig.savefig("figures_hd/cartpole_benchmark.png", dpi=130)
print("\nsaved figures_hd/cartpole_benchmark.png")
