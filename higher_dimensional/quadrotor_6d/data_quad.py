"""Training data for the planar-quadrotor Koopman model.

Drives the true nonlinear plant with random piecewise-constant rotor thrusts,
centred on the hover thrust, from random starts inside the sampling box. Thrust
is kept non-negative. The input is two dimensional.
"""
import numpy as np
from scipy.integrate import solve_ivp


def rollout_with_inputs(plant, x0, u_seq, Ts):
    """Integrate the nonlinear plant under a given input sequence (zero-order hold)."""
    x = np.array(x0, dtype=float)
    X = [x]
    for u in u_seq:
        sol = solve_ivp(lambda t, z: plant.f(z, u), (0.0, Ts), x, rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        X.append(x)
    return np.array(X)


def make_dataset(plant, n_traj=2500, horizon=10, Ts=0.05, coll_amp=3.0, diff_amp=0.8, hold=2, seed=0):
    """Build (X, U) arrays of random trajectories.

    X: (n_traj, horizon+1, 6) states
    U: (n_traj, horizon, 2)   rotor thrusts, held `hold` steps

    The thrust is perturbed around hover as a collective part (both rotors
    together, amplitude coll_amp) and a small differential part (amplitude
    diff_amp) that creates torque. Keeping the differential small stops the
    open-loop angle from winding up, so trajectories stay near the hover region
    where the model needs to be accurate.
    """
    rng = np.random.default_rng(seed)
    n = len(plant.x_eq)
    X = np.zeros((n_traj, horizon + 1, n))
    U = np.zeros((n_traj, horizon, plant.n_input))
    for i in range(n_traj):
        x0 = plant.x_eq + rng.uniform(-1, 1, size=n) * plant.sample_box
        n_blocks = int(np.ceil(horizon / hold))
        dT = rng.uniform(-coll_amp, coll_amp, size=n_blocks)
        dd = rng.uniform(-diff_amp, diff_amp, size=n_blocks)
        u1 = np.clip(plant.u_eq[0] + dT + dd, 0.0, None)
        u2 = np.clip(plant.u_eq[1] + dT - dd, 0.0, None)
        blocks = np.stack([u1, u2], axis=1)
        u_seq = np.repeat(blocks, hold, axis=0)[:horizon]
        X[i] = rollout_with_inputs(plant, x0, u_seq, Ts)
        U[i] = u_seq
    return X, U


if __name__ == "__main__":
    import time
    from systems_quad import get_system
    plant = get_system("quadrotor")
    t0 = time.time()
    X, U = make_dataset(plant, n_traj=60, horizon=12)
    flat = X.reshape(-1, X.shape[-1])
    print(f"generated {X.shape}, inputs {U.shape} in {time.time()-t0:.1f}s")
    print("per-state min:", np.round(flat.min(0), 2))
    print("per-state max:", np.round(flat.max(0), 2))
    print("global max |state|:", float(np.abs(flat).max()))
    print("any non-finite:", bool(~np.isfinite(flat).all()))
