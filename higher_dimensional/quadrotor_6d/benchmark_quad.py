"""Benchmark the 6D region-of-attraction predictor honestly.

Two questions, kept separate so the answer is not confounded:
  1. Does the model-feature representation beat the raw state? To answer this we
     hold the classifier fixed and swap only the representation (ExtraTrees, SVM
     and Gaussian process, each on raw state and on model features).
  2. What is the best predictor overall, and how does accuracy scale with the
     number of tested states?
The classical quadratic Lyapunov sublevel set is included as the control-theory
baseline. Six dimensions cannot be gridded, so every method predicts the
untested states of a large random sample.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from control_quad import load_koopman, local_linear_from_koopman, lqr, Q_QUAD, R_QUAD
from features_quad import model_features
from systems_quad import get_system

BUDGET = 6000
GP_CAP = 1500
SEEDS = 5
GP_SEEDS = 3

d = np.load("results_quad/roa_mc_quad.npz")
X0, y = d["X0"], d["y"]
plant = get_system("quadrotor")
model, A, B, sd, mu = load_koopman(plant.name)
A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
K, P = lqr(A_loc, B_loc, Q_QUAD, R_QUAD)
feats = model_features(model, A, B, sd, mu, K, P, plant.x_eq, plant.u_eq, X0, 8.0)
V = np.einsum("ij,jk,ik->i", X0 - plant.x_eq, P, X0 - plant.x_eq)
N = len(y)


def split(seed, budget=BUDGET):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    return perm[:budget], perm[budget:]


def scores(pred, yt):
    return (pred == yt).mean() * 100, ((pred == 1) & (yt == 0)).sum() / max((pred == 1).sum(), 1) * 100


def run_clf(make, Xrep, seeds, standardize=False, cap=None, budget=BUDGET):
    accs, fss = [], []
    for s in range(seeds):
        tr, te = split(s, budget)
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


def run_lyap(seeds, purity=0.99, budget=BUDGET):
    accs, fss = [], []
    for s in range(seeds):
        tr, te = split(s, budget)
        order = tr[np.argsort(V[tr])]
        run_purity = np.cumsum(y[order] == 1) / np.arange(1, len(order) + 1)
        ok = np.where(run_purity >= purity)[0]
        c = V[order[ok[-1]]] if len(ok) else -1.0
        a, f = scores((V[te] <= c).astype(int), y[te])
        accs.append(a); fss.append(f)
    return np.mean(accs), np.std(accs), np.mean(fss)


et = lambda s: ExtraTreesClassifier(n_estimators=400, random_state=s, n_jobs=-1)
sv = lambda s: SVC(C=10.0, gamma="scale")
gp = lambda s: GaussianProcessClassifier(ConstantKernel(1.0) * RBF(1.0), n_restarts_optimizer=0, max_iter_predict=100)

print(f"planar quadrotor 6D, budget {BUDGET} ({BUDGET/N*100:.0f}%), recover {y.mean()*100:.1f}%\n")
print("representation test (same classifier, raw state vs model features):")
groups = ["ExtraTrees", "SVM", "GP"]
raw_acc, feat_acc, raw_err, feat_err = [], [], [], []
for nm, make, kw in [("ExtraTrees", et, {}), ("SVM", sv, {"standardize": True}),
                     ("GP", gp, {"standardize": True, "cap": GP_CAP})]:
    ns = GP_SEEDS if nm == "GP" else SEEDS
    ar, er, _ = run_clf(make, X0, ns, **kw)
    af, ef, _ = run_clf(make, feats, ns, **kw)
    raw_acc.append(ar); feat_acc.append(af); raw_err.append(er); feat_err.append(ef)
    print(f"  {nm:11s}: raw {ar:5.1f} +-{er:3.1f}   model-feats {af:5.1f} +-{ef:3.1f}")
la, le, lf = run_lyap(SEEDS)
print(f"  quadratic Lyapunov sublevel set: {la:5.1f} +-{le:3.1f}   false-safe {lf:.1f}")

budgets = [200, 400, 800, 1600, 3200, 6400]
curves = {"raw state + SVM": [], "raw state + GP": [], "model feats + ExtraTrees": []}
print(f"\n{'budget':>7s}  {'raw SVM':>8s}  {'raw GP':>8s}  {'feats ET':>9s}")
for b in budgets:
    s_ = run_clf(sv, X0, SEEDS, standardize=True, budget=b)[0]
    g_ = run_clf(gp, X0, GP_SEEDS, standardize=True, cap=GP_CAP, budget=b)[0]
    e_ = run_clf(et, feats, SEEDS, budget=b)[0]
    curves["raw state + SVM"].append(s_); curves["raw state + GP"].append(g_)
    curves["model feats + ExtraTrees"].append(e_)
    print(f"{b:7d}  {s_:8.1f}  {g_:8.1f}  {e_:9.1f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
x = np.arange(len(groups)); w = 0.38
ax[0].bar(x - w / 2, raw_acc, w, yerr=raw_err, capsize=3, color="#1f77b4", label="raw state")
ax[0].bar(x + w / 2, feat_acc, w, yerr=feat_err, capsize=3, color="#2e8b57", label="model features")
ax[0].axhline(la, color="#7f7f7f", ls="--", lw=1.2, label=f"quadratic Lyapunov ({la:.0f}%)")
ax[0].set(title="representation makes little difference;\nthe classifier does", ylabel="% correct", ylim=(45, 100))
ax[0].set_xticks(x); ax[0].set_xticklabels(groups); ax[0].legend(fontsize=8)
colors = {"raw state + SVM": "#1f77b4", "raw state + GP": "#ff7f0e", "model feats + ExtraTrees": "#2e8b57"}
for name, c in curves.items():
    ax[1].plot(budgets, c, "o-", label=name, color=colors[name])
ax[1].set_xscale("log")
ax[1].axhline(90, color="gray", lw=1, ls="--")
ax[1].set(title="accuracy vs test budget (6D)", xlabel="number of states tested (log scale)",
          ylabel="% correct on untested states")
ax[1].legend(fontsize=8)
fig.suptitle("Planar quadrotor (6D): an honest benchmark of the ROA predictor")
fig.tight_layout()
fig.savefig("figures_quad/quad_benchmark.png", dpi=130)
print("\nsaved figures_quad/quad_benchmark.png")
