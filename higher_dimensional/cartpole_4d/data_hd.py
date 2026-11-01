"""Training data for the higher-dimensional Koopman model.

Drives the true nonlinear plant with random piecewise-constant inputs from
random starting states inside the plant's sampling box, sampled at the control
sample time Ts. Dimension general: the number of states is taken from the plant.
"""
import numpy as np
from scipy.integrate import solve_ivp


def rollout_with_inputs(plant, x0, u_seq, Ts):
    """Integrate the nonlinear plant under a given input sequence (ZOH)."""
    x = np.array(x0, dtype=float)
    X = [x]
    for u in u_seq:
        sol = solve_ivp(lambda t, z: plant.f(z, float(u)), (0.0, Ts), x,
                        rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        X.append(x)
    return np.array(X)


def make_dataset(plant, n_traj=800, horizon=25, Ts=0.05, u_amp=6.0, hold=5, seed=0):
    """Build (X, U) arrays of random trajectories.

    X: (n_traj, horizon+1, n) states
    U: (n_traj, horizon, 1)      inputs
    Starts are drawn uniformly from x_eq plus or minus the plant sampling box.
    Inputs are piecewise constant, each value held for `hold` steps.
    """
    rng = np.random.default_rng(seed)
    n = len(plant.x_eq)
    X = np.zeros((n_traj, horizon + 1, n))
    U = np.zeros((n_traj, horizon, 1))
    for i in range(n_traj):
        x0 = plant.x_eq + rng.uniform(-1, 1, size=n) * plant.sample_box
        n_blocks = int(np.ceil(horizon / hold))
        blocks = rng.uniform(-u_amp, u_amp, size=n_blocks)
        u_seq = np.repeat(blocks, hold)[:horizon]
        X[i] = rollout_with_inputs(plant, x0, u_seq, Ts)
        U[i] = u_seq.reshape(-1, 1)
    return X, U


if __name__ == "__main__":
    import time
    from systems_hd import get_system
    plant = get_system("cartpole")
    t0 = time.time()
    X, U = make_dataset(plant, n_traj=40, horizon=25)
    flat = X.reshape(-1, X.shape[-1])
    print(f"generated {X.shape} in {time.time()-t0:.1f}s")
    print("per-state min:", np.round(flat.min(0), 2))
    print("per-state max:", np.round(flat.max(0), 2))
    print("global max |state|:", float(np.abs(flat).max()))
    print("any non-finite:", bool(~np.isfinite(flat).all()))
