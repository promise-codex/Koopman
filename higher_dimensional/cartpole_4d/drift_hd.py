"""Model drift and transfer: does the predictor survive plant mismatch?

The learned Koopman model, the controller, and the model features are all fixed
at the nominal plant. The true pole mass and length are then drifted. For each
drift level we measure the true recover/fail labels of the nominal controller on
the drifted plant, and ask two things:
  - zero-shot: does the predictor trained on the nominal plant still predict the
    drifted outcomes (no new tests)?
  - few-shot: if we test a small number of states on the drifted plant and
    re-fit, does accuracy recover? This is the adaptive-control use case.
The features never change, because they come only from the nominal model.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

from control_hd import load_koopman, local_linear_from_koopman, lqr, Q_CARTPOLE, R_CARTPOLE
from features_hd import model_features
from roa_hd import simulate_batch, f_batch_cartpole
from systems_hd import get_system

Ts = 0.05
UMAX = 8.0
BOX = np.array([2.5, 3.0, 0.9, 3.0])
NSTEPS = 160

plant = get_system("cartpole")
model, A, B, sd, mu = load_koopman(plant.name)
A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
K, P = lqr(A_loc, B_loc, Q_CARTPOLE, R_CARTPOLE)
Kf = K.reshape(-1)

d = np.load("results_hd/roa_mc_cartpole.npz")
X0, y_nom = d["X0"], d["y"]
feats = model_features(model, A, B, sd, mu, K, P, plant.x_eq, plant.u_eq, X0, UMAX)
N = len(y_nom)

rng = np.random.default_rng(0)
tr_nom = rng.permutation(N)[:3750]
clf_nom = ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(feats[tr_nom], y_nom[tr_nom])


def fewshot(yd, nb, seed):
    rng2 = np.random.default_rng(100 + seed)
    pp = rng2.permutation(N)
    trb, teb = pp[:nb], pp[nb:]
    c = ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=-1).fit(feats[trb], yd[trb])
    pred = c.predict(feats[teb])
    acc = (pred == yd[teb]).mean() * 100
    fs = ((pred == 1) & (yd[teb] == 0)).sum() / max((pred == 1).sum(), 1) * 100
    return acc, fs


drifts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
rec, zero, few300, few1000, fs0, fsf = [], [], [], [], [], []
print(f"{'drift':>6s}  {'recover%':>8s}  {'zero-shot':>9s}  {'few-shot300':>11s}  {'few-shot1000':>12s}")
for dlt in drifts:
    fb = lambda Xa, ua, dd=dlt: f_batch_cartpole(Xa, ua, M=1.0, m=0.1 * (1 + dd), l=0.5 * (1 + dd))
    Xf = simulate_batch(Kf, X0, plant.x_eq, plant.u_eq, Ts, NSTEPS, UMAX, fb=fb)
    yd = (np.linalg.norm(Xf - plant.x_eq, axis=1) < 0.1).astype(int)
    rec.append(yd.mean() * 100)
    pred0 = clf_nom.predict(feats)
    zero.append((pred0 == yd).mean() * 100)
    fs0.append(((pred0 == 1) & (yd == 0)).sum() / max((pred0 == 1).sum(), 1) * 100)
    a3 = np.mean([fewshot(yd, 300, s)[0] for s in range(3)])
    r = [fewshot(yd, 1000, s) for s in range(3)]
    a10 = np.mean([x[0] for x in r]); f10 = np.mean([x[1] for x in r])
    few300.append(a3); few1000.append(a10); fsf.append(f10)
    print(f"{dlt*100:5.0f}%  {yd.mean()*100:8.1f}  {zero[-1]:9.1f}  {a3:11.1f}  {a10:12.1f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
dd = np.array(drifts) * 100
ax[0].plot(dd, rec, "o-", color="#7f7f7f")
ax[0].set(title="true recover rate shifts as the plant drifts", xlabel="pole mass and length drift (%)",
          ylabel="% of states that recover")
ax[1].axhline(95, color="gray", lw=1, ls="--")
ax[1].plot(dd, zero, "o-", color="#d62728", label="zero-shot (nominal predictor)")
ax[1].plot(dd, few300, "o-", color="#ff7f0e", label="few-shot (300 drifted tests)")
ax[1].plot(dd, few1000, "o-", color="#2e8b57", label="few-shot (1000 drifted tests)")
ax[1].set(title="predicting the drifted failure region", xlabel="pole mass and length drift (%)",
          ylabel="% correct", ylim=(70, 100))
ax[1].legend(fontsize=8)
fig.suptitle("Cart-pole: transfer of the predictor under plant drift")
fig.tight_layout()
fig.savefig("figures_hd/cartpole_drift.png", dpi=130)
print("\nsaved figures_hd/cartpole_drift.png")
