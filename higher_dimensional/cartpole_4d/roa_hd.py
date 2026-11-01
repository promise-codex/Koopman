"""Monte-Carlo region of attraction for the cart-pole, with a vectorised
closed-loop simulator.

In four dimensions the state space cannot be gridded, so the ground truth is a
large random sample of initial states, each simulated under the learned-model
controller and labelled recover or fail. The simulator integrates every sampled
start at once with a fixed-step RK4 scheme, which is fast enough for tens of
thousands of starts. It is validated against the accurate adaptive solver.
"""
import numpy as np


def f_batch_cartpole(X, u, M=1.0, m=0.1, l=0.5, g=9.81):
    """Vectorised cart-pole dynamics. X is (N,4), u is (N,), returns (N,4)."""
    p, v, th, om = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    s, c = np.sin(th), np.cos(th)
    den = M + m * s * s
    pacc = (u + m * l * s * om * om - m * g * s * c) / den
    thacc = (g * s - c * pacc) / l
    return np.stack([v, pacc, om, thacc], axis=1)


def simulate_batch(Kf, X0, x_eq, u_eq, Ts, n_steps, umax, substeps=5, fb=f_batch_cartpole):
    """Closed-loop RK4 rollout of every start at once. Returns final states."""
    X = X0.astype(float).copy()
    dt = Ts / substeps
    lo, hi = x_eq - 1e3, x_eq + 1e3
    for _ in range(n_steps):
        u = np.clip(u_eq - (X - x_eq) @ Kf, -umax, umax)
        for _ in range(substeps):
            k1 = fb(X, u)
            k2 = fb(X + 0.5 * dt * k1, u)
            k3 = fb(X + 0.5 * dt * k2, u)
            k4 = fb(X + dt * k3, u)
            X = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        X = np.clip(X, lo, hi)
        X[~np.isfinite(X)] = 1e3
    return X


def monte_carlo_roa(Kf, plant, Ts, n_steps, umax, N, box, seed=0, tol=0.1):
    rng = np.random.default_rng(seed)
    X0 = plant.x_eq + rng.uniform(-1, 1, size=(N, len(plant.x_eq))) * box
    Xf = simulate_batch(Kf, X0, plant.x_eq, plant.u_eq, Ts, n_steps, umax)
    recover = np.linalg.norm(Xf - plant.x_eq, axis=1) < tol
    return X0, recover


if __name__ == "__main__":
    import numpy as np
    from control_hd import load_koopman, local_linear_from_koopman, lqr, Q_CARTPOLE, R_CARTPOLE
    from sim import simulate_digital_loop
    from systems_hd import get_system

    Ts = 0.05
    plant = get_system("cartpole")
    model, A, B, sd, mu = load_koopman(plant.name)
    A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
    K, P = lqr(A_loc, B_loc, Q_CARTPOLE, R_CARTPOLE)
    Kf = K.reshape(-1)

    # 1) validate the vectorised RK4 simulator against the adaptive solver
    print("=== validation: vectorised RK4 vs adaptive solve_ivp (final state) ===")
    ctrl = lambda x: plant.u_eq - float(Kf @ (np.asarray(x) - plant.x_eq))
    tests = [[0.3, 0, 0.1, 0], [1.0, 0, 0.3, 0.5], [0.0, 0, 0.4, 0],
             [2.0, 0, 0.7, 0], [-1.5, 1.0, 0.6, -1.0]]
    X0 = np.array(tests, dtype=float)
    Xf_rk = simulate_batch(Kf, X0, plant.x_eq, plant.u_eq, Ts, 160, 20.0)
    for s, xf in zip(tests, Xf_rk):
        _, Xk, _ = simulate_digital_loop(plant, ctrl, np.array(s, float), Ts, 160, u_max=20.0)
        d = np.linalg.norm(xf - Xk[-1])
        print(f"  x0={np.array(s)}  rk4 final={np.round(xf,3)}  solver diff={d:.2e}")

    # 2) recover fraction for a few actuator caps and boxes (pick a non-trivial split)
    print("\n=== recover fraction (N=6000) ===")
    for umax in (8.0, 10.0, 12.0, 15.0):
        for box in ([2.0, 2.0, 0.7, 2.5], [2.5, 3.0, 0.9, 3.0]):
            _, rec = monte_carlo_roa(Kf, plant, Ts, 160, umax, 6000, np.array(box), seed=1)
            print(f"  umax={umax:5.1f}  box={box}  recover={rec.mean()*100:5.1f}%")
