"""Model-only features for a sampled set of initial states (dimension general).

Every feature is computed from the learned model and the local LQR design, with
no extra runs of the true plant. For an n-dimensional state the offset
contributes n features, so the total is n + 12.
"""
import numpy as np
import torch

CLIP = 50.0
HORIZONS = (2, 5, 10, 20)


def feature_names(n):
    off = [f"offset_{i}" for i in range(n)]
    return off + ["|offset|", "push", "push/umax", "lyapunov", "||lift||",
                  "run@2", "run@5", "run@10", "run@20",
                  "min_approach", "sat_frac", "growth"]


def model_features(model, A, B, sd, mu, K, P, x_eq, u_eq, X0, umax):
    """Cheap per-state descriptors taken only from the learned model."""
    n = X0.shape[1]
    X0 = X0.astype(np.float32)
    E = X0 - x_eq
    Kf = np.asarray(K).reshape(-1)
    push = np.abs(E @ Kf)
    lyap = np.einsum("ij,jk,ik->i", E, P, E)
    with torch.no_grad():
        liftnorm = np.linalg.norm(model.lift(torch.from_numpy(X0)).numpy(), axis=1)
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
        X = (Z @ A.T + u[:, None] * Brow)[:, :n] * sd + mu
        X = np.clip(X, x_eq - CLIP, x_eq + CLIP).astype(np.float32)
        d = np.linalg.norm(X - x_eq, axis=1)
        min_app = np.minimum(min_app, d)
        if step in HORIZONS:
            norms[step] = np.log1p(d)
    feats = np.column_stack([
        E, np.linalg.norm(E, axis=1), push, np.minimum(push / umax, 3.0), lyap, liftnorm,
        norms[2], norms[5], norms[10], norms[20],
        np.log1p(min_app), sat / max(HORIZONS), norms[20] - norms[2]])
    return feats
