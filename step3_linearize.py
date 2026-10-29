"""Step 3: linearize at the equilibrium and discretize at sample time Ts.

Run:  python step3_linearize.py --system duffing --Ts 0.05
"""
import argparse

import control as ct
import numpy as np

from systems import get_system


def jacobians(plant, x_eq, u_eq, eps=1e-6):
    """Central-difference Jacobians A = df/dx, B = df/du at (x_eq, u_eq)."""
    n = len(x_eq)
    A = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        A[:, i] = (plant.f(x_eq + dx, u_eq) - plant.f(x_eq - dx, u_eq)) / (2 * eps)
    B = ((plant.f(x_eq, u_eq + eps) - plant.f(x_eq, u_eq - eps)) / (2 * eps)).reshape(n, 1)
    return A, B


def linearize_and_discretize(plant, Ts):
    A, B = jacobians(plant, plant.x_eq, plant.u_eq)
    n = A.shape[0]
    sys_c = ct.ss(A, B, np.eye(n), np.zeros((n, 1)))
    sys_d = ct.c2d(sys_c, Ts, method="zoh")
    return A, B, np.asarray(sys_d.A), np.asarray(sys_d.B)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="duffing", choices=["duffing", "pendulum"])
    p.add_argument("--Ts", type=float, default=0.05)
    args = p.parse_args()

    np.set_printoptions(precision=4, suppress=True)
    plant = get_system(args.system)
    A, B, Ad, Bd = linearize_and_discretize(plant, args.Ts)

    print(f"System: {plant.name}   equilibrium x_eq = {plant.x_eq}   Ts = {args.Ts} s\n")
    print("Continuous A =\n", A, "\nContinuous B =\n", B)
    print("\nContinuous eigenvalues:", np.linalg.eigvals(A),
          " (any with real part > 0 => unstable)")
    print("\nDiscrete Ad =\n", Ad, "\nDiscrete Bd =\n", Bd)
    print("\nDiscrete eigenvalues:", np.linalg.eigvals(Ad))
    print("|eig(Ad)|:", np.abs(np.linalg.eigvals(Ad)), " (any > 1 => unstable)")
    print("Sanity check: eig(Ad) should equal exp(eig(A)*Ts):",
          np.exp(np.linalg.eigvals(A) * args.Ts))
    ctrb = ct.ctrb(Ad, Bd)
    print("\nControllability matrix rank:", np.linalg.matrix_rank(ctrb), f"(need {A.shape[0]})")
