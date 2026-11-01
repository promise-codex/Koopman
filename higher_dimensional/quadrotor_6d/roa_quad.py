"""Monte-Carlo region of attraction for the planar quadrotor, with a vectorised
two-input closed-loop simulator.

Six dimensions cannot be gridded, so the ground truth is a large random sample of
initial states, each simulated under the learned-model controller and labelled
recover or fail. The simulator integrates every sampled start at once with a
fixed-step RK4 scheme and is validated against the accurate adaptive solver.
Rotor thrust is clipped to a non-negative range with an upper limit per rotor.
"""
import numpy as np


def f_batch_quad(X, u, m=1.0, I=0.05, r=0.2, g=9.81):
    """Vectorised planar-quadrotor dynamics. X is (N,6), u is (N,2), returns (N,6)."""
    phi, vx, vz, w = X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    u1, u2 = u[:, 0], u[:, 1]
    s, c = np.sin(phi), np.cos(phi)
    thrust = (u1 + u2) / m
    ax = -thrust * s
    az = thrust * c - g
    aphi = r * (u1 - u2) / I
    return np.stack([vx, vz, w, ax, az, aphi], axis=1)


def simulate_batch(K, X0, x_eq, u_eq, Ts, n_steps, umax, substeps=5, fb=f_batch_quad, u_lo=0.0):
    """Closed-loop RK4 rollout of every start at once. K is (2,6). Returns final states."""
    X = X0.astype(float).copy()
    dt = Ts / substeps
    lo, hi = x_eq - 1e3, x_eq + 1e3
    KT = np.asarray(K).T                          # (6,2)
    for _ in range(n_steps):
        u = np.clip(u_eq - (X - x_eq) @ KT, u_lo, umax)   # (N,2)
        for _ in range(substeps):
            k1 = fb(X, u)
            k2 = fb(X + 0.5 * dt * k1, u)
            k3 = fb(X + 0.5 * dt * k2, u)
            k4 = fb(X + dt * k3, u)
            X = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        X = np.clip(X, lo, hi)
        X[~np.isfinite(X)] = 1e3
    return X


def monte_carlo_roa(K, plant, Ts, n_steps, umax, N, box, seed=0, tol=0.15):
    rng = np.random.default_rng(seed)
    X0 = plant.x_eq + rng.uniform(-1, 1, size=(N, len(plant.x_eq))) * box
    Xf = simulate_batch(K, X0, plant.x_eq, plant.u_eq, Ts, n_steps, umax)
    recover = np.linalg.norm(Xf - plant.x_eq, axis=1) < tol
    return X0, recover


if __name__ == "__main__":
    from scipy.integrate import solve_ivp
    from control_quad import load_koopman, local_linear_from_koopman, lqr, Q_QUAD, R_QUAD
    from systems_quad import get_system

    Ts = 0.05
    plant = get_system("quadrotor")
    model, A, B, sd, mu = load_koopman(plant.name)
    A_loc, B_loc = local_linear_from_koopman(model, A, B, sd, plant.x_eq)
    K, P = lqr(A_loc, B_loc, Q_QUAD, R_QUAD)

    # 1) validate the vectorised RK4 simulator against the adaptive solver
    print("=== validation: vectorised RK4 vs adaptive solve_ivp (final state) ===")
    KT = np.asarray(K)
    tests = [[0.3, 0.3, 0.1, 0, 0, 0], [0.8, -0.5, 0.2, 0.3, 0, 0],
             [-0.6, 0.4, -0.2, 0, 0.5, -0.5], [1.0, 1.0, 0.3, 0, 0, 1.0]]

    def ctrl_ode(t, x):
        u = np.clip(plant.u_eq - KT @ (x - plant.x_eq), 0.0, 12.0)
        return plant.f(x, u)

    X0 = np.array(tests, dtype=float)
    Xf_rk = simulate_batch(K, X0, plant.x_eq, plant.u_eq, Ts, 200, 12.0)
    for s, xf in zip(tests, Xf_rk):
        x = np.array(s, float)
        for _ in range(200):
            sol = solve_ivp(ctrl_ode, (0, Ts), x, rtol=1e-8, atol=1e-10)
            x = sol.y[:, -1]
        print(f"  x0={np.array(s)}  rk4 last={np.round(xf,3)}  solver diff={np.linalg.norm(xf-x):.2e}")

    # 2) recover fraction over a few actuator caps and boxes
    print("\n=== recover fraction (N=6000) ===")
    for umax in (9.0, 12.0, 15.0):
        for box in ([1.0, 1.0, 0.5, 1.0, 1.0, 1.5], [1.5, 1.5, 0.7, 1.5, 1.5, 2.0]):
            _, rec = monte_carlo_roa(K, plant, Ts, 240, umax, 6000, np.array(box), seed=1)
            print(f"  umax={umax:5.1f}  box={box}  recover={rec.mean()*100:5.1f}%")
