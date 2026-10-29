"""Validation of the step 9 failure predictor.

Reuses the exact feature pipeline from step 9 (imported, not re-implemented).
For each system it:
  * builds the true recover/fail map at two resolutions (51x51, 81x81) and
    checks that the result is stable as the grid is refined,
  * scores the predictor three ways:
      (R) random hold-out: test scattered points, predict the gaps,
      (L) coarse to fine:  train on a regular coarse lattice, predict the fine
                           map in between,
      (X) region extrapolation: train on one half of the plane, predict the
                           other half (the hardest case, which tests
                           generalisation),
    each averaged over several seeds or both axes.
All numbers are printed as they are reported in the manuscript.
"""
import time
import warnings

import control as ct
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier

from step8_koopman_control import load_koopman, local_linear_from_koopman, lqr, region_of_attraction
from step9_predict_failure import model_features
from systems import get_system

warnings.filterwarnings("ignore")


def setup(system):
    plant = get_system(system)
    model, A, B, sd = load_koopman(plant.name)
    mu = np.load(f"results/koopman_{plant.name}.npz")["mu"]
    A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
    K = lqr(A_loc, B_loc)
    _, S, _ = ct.dlqr(A_loc, B_loc, np.diag([10.0, 1.0]), np.array([[0.1]]))
    ctrl = lambda x: plant.u_eq - float((np.asarray(K) @ (np.asarray(x) - plant.x_eq)).item())
    return plant, model, A, B, sd, mu, np.asarray(K), np.asarray(S), ctrl


def gt_cached(plant, ctrl, umax, grid):
    f = f"results/roa_g{grid}_{plant.name}.npz"
    try:
        d = np.load(f)
        return d["p"], d["v"], d["roa"].astype(bool)
    except FileNotFoundError:
        pass
    t0 = time.time()
    pv, vv, ok = region_of_attraction(plant, ctrl, 0.05, umax, span=3.0, grid=grid, t_end=8.0)
    np.savez(f, p=pv, v=vv, roa=ok, grid=grid, umax=umax)
    print(f"    (ground truth {grid}x{grid}: {time.time()-t0:.0f}s)", flush=True)
    return pv, vv, ok.astype(bool)


def et(seed):
    return ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1)


def score(feats, y, train_mask, seed=0):
    clf = et(seed); clf.fit(feats[train_mask], y[train_mask])
    pred = clf.predict(feats).astype(bool)
    held = ~train_mask
    yb = y.astype(bool)
    acc = (pred[held] == yb[held]).mean()
    fsafe = (pred[held] & ~yb[held]).mean()
    return acc, fsafe


def random_holdout(feats, y, frac, seeds):
    N = len(y); a, f = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        m = np.zeros(N, bool); m[rng.choice(N, int(frac * N), replace=False)] = True
        acc, fs = score(feats, y, m, s)
        a.append(acc); f.append(fs)
    return np.mean(a), np.std(a), np.mean(f), frac


def lattice_holdout(feats, y, shape, step):
    """Train on every `step`-th point in each axis (a coarse regular grid)."""
    R, C = shape
    m = np.zeros((R, C), bool)
    m[::step, ::step] = True
    m = m.ravel()
    acc, fs = score(feats, y, m, 0)
    return acc, fs, m.mean()


def region_extrapolation(feats, y, PV, VV):
    """Train on one half of the plane, predict the other, both axes, averaged."""
    accs, fss, trf = [], [], []
    for coord in (PV.ravel(), VV.ravel()):
        med = np.median(coord)
        for left in (coord <= med, coord > med):
            acc, fs = score(feats, y, left, 0)
            accs.append(acc); fss.append(fs); trf.append(left.mean())
    return np.mean(accs), np.mean(fss), np.mean(trf)


if __name__ == "__main__":
    grids = [51, 81]
    seeds = list(range(10))
    for system, umax in [("duffing", 0.5), ("pendulum", 5.0)]:
        plant, model, A, B, sd, mu, K, P, ctrl = setup(system)
        print(f"\n########## {system} (umax={umax}) ##########", flush=True)
        for grid in grids:
            pv, vv, ok = gt_cached(plant, ctrl, umax, grid)
            feats, _, PV, VV = model_features(model, A, B, sd, mu, K, P,
                                              plant.x_eq, plant.u_eq, pv, vv, umax)
            y = ok.ravel().astype(int)
            N = len(y)
            print(f"  --- {grid}x{grid} = {N} cells, fail-rate {1-y.mean():.3f} ---", flush=True)
            am, asd, af, fr = random_holdout(feats, y, 0.35, seeds)
            print(f"    (R) random hold-out 35%:   acc {am*100:5.1f} +-{asd*100:3.1f}  "
                  f"false-safe {af*100:4.1f}  (over {len(seeds)} seeds)", flush=True)
            for stp in (2, 3, 4):
                acc, fs, tf = lattice_holdout(feats, y, PV.shape, stp)
                print(f"    (L) coarse lattice 1/{stp} (test {tf*100:4.1f}%): "
                      f"predict fine acc {acc*100:5.1f}  false-safe {fs*100:4.1f}", flush=True)
            xa, xf, xtf = region_extrapolation(feats, y, PV, VV)
            print(f"    (X) region extrapolation (train ~{xtf*100:.0f}% one side): "
                  f"acc {xa*100:5.1f}  false-safe {xf*100:4.1f}", flush=True)
    print("\nDONE", flush=True)
