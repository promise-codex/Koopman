"""Accuracy versus test budget: does the model-feature representation help most
when few true-plant tests are available (the data-efficiency claim)?

Compares the model-feature ExtraTrees against the two strongest raw-state
baselines (SVM and Gaussian process) as the number of tested states shrinks.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from control_hd import load_koopman, local_linear_from_koopman, lqr, Q_CARTPOLE, R_CARTPOLE
from features_hd import model_features
from systems_hd import get_system

d = np.load("results_hd/roa_mc_cartpole.npz")
X0, y = d["X0"], d["y"]
plant = get_system("cartpole")
model, A, B, sd, mu = load_koopman(plant.name)
A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
K, P = lqr(A_loc, B_loc, Q_CARTPOLE, R_CARTPOLE)
feats = model_features(model, A, B, sd, mu, K, P, plant.x_eq, plant.u_eq, X0, 8.0)
N = len(y)

budgets = [100, 200, 400, 800, 1600, 3200]


def ev(make, Xrep, budget, seed, standardize=False, cap=None):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    tr, te = perm[:budget], perm[budget:]
    if cap and len(tr) > cap:
        tr = tr[:cap]
    Xtr, Xte = Xrep[tr], Xrep[te]
    if standardize:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = make(seed).fit(Xtr, y[tr])
    return (clf.predict(Xte) == y[te]).mean() * 100


et_make = lambda s: ExtraTreesClassifier(n_estimators=400, random_state=s, n_jobs=-1)
sv_make = lambda s: SVC(C=10.0, gamma="scale")
gp_make = lambda s: GaussianProcessClassifier(ConstantKernel(1.0) * RBF(1.0), n_restarts_optimizer=0)

curves = {"model feats + ExtraTrees": [], "raw state + SVM": [], "raw state + GP": []}
print(f"{'budget':>7s}  {'model-feats':>11s}  {'raw SVM':>8s}  {'raw GP':>8s}")
for b in budgets:
    et = np.mean([ev(et_make, feats, b, s) for s in range(5)])
    sv = np.mean([ev(sv_make, X0, b, s, standardize=True) for s in range(5)])
    gp = np.mean([ev(gp_make, X0, b, s, standardize=True, cap=1500) for s in range(3)])
    curves["model feats + ExtraTrees"].append(et)
    curves["raw state + SVM"].append(sv)
    curves["raw state + GP"].append(gp)
    print(f"{b:7d}  {et:11.1f}  {sv:8.1f}  {gp:8.1f}")

plt.figure(figsize=(7, 5))
colors = {"model feats + ExtraTrees": "#2e8b57", "raw state + SVM": "#1f77b4", "raw state + GP": "#ff7f0e"}
for name, c in curves.items():
    plt.plot(budgets, c, "o-", label=name, color=colors[name])
plt.xscale("log")
plt.axhline(95, color="gray", lw=1, ls="--")
plt.xlabel("number of states tested (log scale)")
plt.ylabel("% correct on untested states")
plt.title("Cart-pole: accuracy vs test budget")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures_hd/cartpole_budget.png", dpi=130)
print("\nsaved figures_hd/cartpole_budget.png")
