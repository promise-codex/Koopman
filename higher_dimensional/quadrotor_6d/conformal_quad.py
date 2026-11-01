"""Distribution-free control of the false-safe rate for the 6D quadrotor predictor.

Same procedure as the cart-pole: among the states the predictor declares safe,
bound the fraction that actually fail with confidence 1 - delta, by choosing a
recover-probability threshold on a calibration set with a Clopper-Pearson upper
bound and a Bonferroni correction over a fixed threshold grid. The certificate
wraps any base predictor; here it wraps the model-feature ExtraTrees. The
guarantee is checked over many calibration/test splits.
"""
import numpy as np
from scipy.stats import beta
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

from control_quad import load_koopman, local_linear_from_koopman, lqr, Q_QUAD, R_QUAD
from features_quad import model_features
from systems_quad import get_system

DELTA = 0.10
GRID = np.linspace(0.5, 0.999, 50)
CAL = 8000
TRIALS = 200

d = np.load("results_quad/roa_mc_quad.npz")
X0, y = d["X0"], d["y"]
plant = get_system("quadrotor")
model, A, B, sd, mu = load_koopman(plant.name)
A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
K, P = lqr(A_loc, B_loc, Q_QUAD, R_QUAD)
feats = model_features(model, A, B, sd, mu, K, P, plant.x_eq, plant.u_eq, X0, 8.0)

rng0 = np.random.default_rng(0)
perm = rng0.permutation(len(y))
tr, pool = perm[:6000], perm[6000:]
clf = ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(feats[tr], y[tr])
q_pool = clf.predict_proba(feats[pool])[:, 1]
y_pool = y[pool]


def cp_upper(k, n, dloc):
    if n == 0 or k == n:
        return 1.0
    return float(beta.ppf(1 - dloc, k + 1, n - k))


def certify_tau(qc, yc, alpha):
    dloc = DELTA / len(GRID)
    for tau in GRID:
        safe = qc >= tau
        n = int(safe.sum()); k = int((safe & (yc == 0)).sum())
        if n >= 20 and cp_upper(k, n, dloc) <= alpha:
            return tau
    return None


safe05 = q_pool >= 0.5
print("naive 0.5 threshold (no certificate):")
print(f"  declares {safe05.mean()*100:.1f}% safe, realized false-safe "
      f"{(safe05 & (y_pool==0)).sum()/max(safe05.sum(),1)*100:.2f}%")

alphas = [0.10, 0.05, 0.02]
res = {}
print(f"\ncertified control (confidence {int((1-DELTA)*100)}%), {TRIALS} calib/test splits each:")
for alpha in alphas:
    covs, rates, held = [], [], []
    for t in range(TRIALS):
        rng = np.random.default_rng(1000 + t)
        idx = rng.permutation(len(pool))
        cal, tst = idx[:CAL], idx[CAL:]
        tau = certify_tau(q_pool[cal], y_pool[cal], alpha)
        if tau is None:
            covs.append(0.0); rates.append(0.0); held.append(True); continue
        safe = q_pool[tst] >= tau
        n = int(safe.sum()); k = int((safe & (y_pool[tst] == 0)).sum())
        covs.append(float(safe.mean()))
        rate = k / n if n > 0 else 0.0
        rates.append(rate); held.append(rate <= alpha)
    res[alpha] = (np.mean(covs), np.std(covs), np.mean(rates), np.std(rates), np.mean(held))
    print(f"  alpha={alpha:.2f}: declares {np.mean(covs)*100:4.1f}% safe,  "
          f"realized false-safe {np.mean(rates)*100:4.2f}% +-{np.std(rates)*100:.2f},  "
          f"held in {np.mean(held)*100:5.1f}% of splits (target >= {int((1-DELTA)*100)}%)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
a = np.array(alphas)
rate_m = np.array([res[al][2] for al in alphas]) * 100
rate_s = np.array([res[al][3] for al in alphas]) * 100
cov_m = np.array([res[al][0] for al in alphas]) * 100
ax[0].plot([0, 11], [0, 11], color="gray", ls="--", lw=1, label="target (realized = alpha)")
ax[0].errorbar(a * 100, rate_m, yerr=rate_s, fmt="o-", color="#2e8b57", capsize=3, label="realized false-safe")
ax[0].set(title="certified vs realized false-safe rate", xlabel="target alpha (%)",
          ylabel="realized false-safe on test (%)", xlim=(0, 11), ylim=(0, 11))
ax[0].legend(fontsize=8)
ax[1].plot(a * 100, cov_m, "o-", color="#1f77b4")
ax[1].set(title="cost of a tighter guarantee", xlabel="target alpha (%)",
          ylabel="% of states declared safe", ylim=(0, 100))
fig.suptitle("Planar quadrotor (6D): distribution-free control of the false-safe rate (confidence 90%)")
fig.tight_layout()
fig.savefig("figures_quad/quad_conformal.png", dpi=130)
print("\nsaved figures_quad/quad_conformal.png")
