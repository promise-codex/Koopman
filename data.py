"""Training data for the Koopman model.

Drives the true nonlinear plant with random piecewise-constant inputs from
random starting states, sampled at the same Ts the controller will use. The
network is trained to reproduce these trajectories with a linear model in a
lifted space.
"""
import numpy as np
from scipy.integrate import solve_ivp


def rollout_with_inputs(plant, x0, u_seq, Ts):
    """Integrate the nonlinear plant under a given input sequence (ZOH).

    Returns X of shape (len(u_seq)+1, n): the state before each input,
    plus the final state.
    """
    x = np.array(x0, dtype=float)
    X = [x]
    for u in u_seq:
        sol = solve_ivp(lambda t, z: plant.f(z, float(u)), (0.0, Ts), x,
                        rtol=1e-8, atol=1e-10)
        x = sol.y[:, -1]
        X.append(x)
    return np.array(X)


def make_dataset(plant, n_traj=400, horizon=30, Ts=0.05, u_amp=3.0,
                 x_range=(2.5, 3.0), hold=4, seed=0):
    """Build (X, U) arrays of random trajectories.

    X: (n_traj, horizon+1, n) states
    U: (n_traj, horizon, 1)      inputs
    `hold` is the number of steps each random input value is held, so the input
    is piecewise-constant rather than white noise, which excites the dynamics
    more effectively.
    """
    rng = np.random.default_rng(seed)
    n = len(plant.x_eq)
    X = np.zeros((n_traj, horizon + 1, n))
    U = np.zeros((n_traj, horizon, 1))
    for i in range(n_traj):
        x0 = plant.x_eq + rng.uniform(-1, 1, size=n) * np.array(x_range)
        # piecewise-constant input
        n_blocks = int(np.ceil(horizon / hold))
        blocks = rng.uniform(-u_amp, u_amp, size=n_blocks)
        u_seq = np.repeat(blocks, hold)[:horizon]
        X[i] = rollout_with_inputs(plant, x0, u_seq, Ts)
        U[i] = u_seq.reshape(-1, 1)
    return X, U


if __name__ == "__main__":
    from systems import get_system
    X, U = make_dataset(get_system("duffing"), n_traj=20, horizon=30)
    print("X", X.shape, "U", U.shape)
    print("state ranges:", X.reshape(-1, 2).min(0), X.reshape(-1, 2).max(0))
