"""Distribution-free control of the false-safe rate for the cart-pole predictor.

Target: among the states the predictor declares SAFE (predicts recover), at most
a fraction alpha actually fail, with confidence 1 - delta. A threshold on the
recover-probability is chosen on a calibration set using a Clopper-Pearson upper
bound on the conditional false-safe rate, with a Bonferroni correction over a
fixed grid of thresholds (a valid distribution-free selective-risk procedure).
The guarantee is then checked empirically over many calibration/test splits:
the realized test false-safe rate should stay at or below alpha in at least
1 - delta of the splits.
"""
import numpy as np
from scipy.stats import beta
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

from control_hd import load_koopman, local_linear_from_koopman, lqr, Q_CARTPOLE, R_CARTPOLE
from features_hd import model_features
from systems_hd import get_system

DELTA = 0.10
GRID = np.linspace(0.5, 0.999, 50)
CAL = 5000
TRIALS = 200

d = np.load("results_hd/roa_mc_cartpole.npz")
X0, y = d["X0"], d["y"]                 # y = 1 recover, 0 fail
plant = get_system("cartpole")
model, A, B, sd, mu = load_koopman(plant.name)
A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
K, P = lqr(A_loc, B_loc, Q_CARTPOLE, R_CARTPOLE)
feats = model_features(model, A, B, sd, mu, K, P, plant.x_eq, plant.u_eq, X0, 8.0)

# fixed classifier trained on 15% of the states; conformal calibration uses fresh data
rng0 = np.random.default_rng(0)
perm = rng0.permutation(len(y))
tr, pool = perm[:3750], perm[3750:]
clf = ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(feats[tr], y[tr])
q_pool = clf.predict_proba(feats[pool])[:, 1]       # predicted P(recover)
y_pool = y[pool]


def cp_upper(k, n, dloc):
    """Clopper-Pearson (1 - dloc) upper bound on a binomial rate: k fails of n."""
    if n == 0 or k == n:
        return 1.0
    return float(beta.ppf(1 - dloc, k + 1, n - k))


def certify_tau(qc, yc, alpha):
    """Smallest threshold (most states declared safe) whose certified false-safe
    bound is at or below alpha, using Bonferroni over the fixed grid."""
    dloc = DELTA / len(GRID)
    for tau in GRID:                                # ascending -> most coverage first
        safe = qc >= tau
        n = int(safe.sum()); k = int((safe & (yc == 0)).sum())
        if n >= 20 and cp_upper(k, n, dloc) <= alpha:
            return tau
    return None


print(f"classifier false-safe at the naive 0.5 threshold (no guarantee), over the pool:")
safe05 = q_pool >= 0.5
print(f"  declares {safe05.mean()*100:.1f}% safe, realized false-safe "
      f"{(safe05 & (y_pool==0)).sum()/max(safe05.sum(),1)*100:.2f}%  (no certificate)")

alphas = [0.05, 0.02, 0.01]
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
          f"guarantee held in {np.mean(held)*100:5.1f}% of splits (target >= {int((1-DELTA)*100)}%)")

# figure: realized false-safe vs target, and the cost in coverage
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
a = np.array(alphas)
rate_m = np.array([res[al][2] for al in alphas]) * 100
rate_s = np.array([res[al][3] for al in alphas]) * 100
cov_m = np.array([res[al][0] for al in alphas]) * 100
ax[0].plot([0, 6], [0, 6], color="gray", ls="--", lw=1, label="target (realized = alpha)")
ax[0].errorbar(a * 100, rate_m, yerr=rate_s, fmt="o-", color="#2e8b57", capsize=3, label="realized false-safe")
ax[0].set(title="certified vs realized false-safe rate", xlabel="target alpha (%)",
          ylabel="realized false-safe on test (%)", xlim=(0, 6), ylim=(0, 6))
ax[0].legend(fontsize=8)
ax[1].plot(a * 100, cov_m, "o-", color="#1f77b4")
ax[1].set(title="cost of a tighter guarantee", xlabel="target alpha (%)",
          ylabel="% of states declared safe", ylim=(0, 100))
fig.suptitle("Cart-pole: distribution-free control of the false-safe rate (confidence 90%)")
fig.tight_layout()
fig.savefig("figures_hd/cartpole_conformal.png", dpi=130)
print("\nsaved figures_hd/cartpole_conformal.png")
