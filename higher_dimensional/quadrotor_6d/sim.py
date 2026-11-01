"""Simulation helpers shared across the pipeline."""
import numpy as np
from scipy.integrate import solve_ivp


def simulate_open_loop(plant, x0, t_end, u_func=lambda t: 0.0, dt_out=0.01):
    """Integrate the nonlinear plant with a continuous input u(t)."""
    t_eval = np.arange(0.0, t_end, dt_out)
    sol = solve_ivp(lambda t, x: plant.f(x, u_func(t)), (0.0, t_end), x0,
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y.T


def simulate_digital_loop(plant, controller, x0, Ts, n_steps, u_max=None):
    """Sampled-data loop: sample x_k, compute u_k, hold u_k over Ts (ZOH),
    integrate the true nonlinear plant over that interval, repeat.

    This reproduces how a digital controller runs on hardware, and is the loop
    the learned (Koopman) controller is evaluated in.
    """
    xs, us = [np.array(x0, dtype=float)], []
    for _ in range(n_steps):
        x = xs[-1]
        u = float(controller(x))
        if u_max is not None:
            u = float(np.clip(u, -u_max, u_max))
        sol = solve_ivp(lambda t, z: plant.f(z, u), (0.0, Ts), x, rtol=1e-8, atol=1e-10)
        xs.append(sol.y[:, -1])
        us.append(u)
    t = np.arange(n_steps + 1) * Ts
    return t, np.array(xs), np.array(us)
