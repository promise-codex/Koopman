"""Planar quadrotor: a six-dimensional, two-input benchmark for the region-of-
attraction study.

The planar quadrotor moves in a vertical plane. Its state is

    x = [px, pz, phi, vx, vz, wphi]

where px and pz are the horizontal and vertical positions, phi is the body tilt
from the vertical, and vx, vz, wphi are the matching velocities. Two rotor
thrusts u = [u1, u2] act at the ends of an arm of half-length r. With mass m,
moment of inertia I and gravity g, the standard rigid-body equations are

    m px_ddot  = -(u1 + u2) sin(phi)
    m pz_ddot  =  (u1 + u2) cos(phi) - m g
    I phi_ddot =  r (u1 - u2)

(see Tedrake, Underactuated Robotics, the planar quadrotor model). Hover is the
target: phi = 0, all velocities zero, with each rotor carrying half the weight,
so u_eq = [m g / 2, m g / 2]. The open loop is a chain of integrators, so the
task is genuinely unstable and needs active control. The state is six
dimensional, so a regular grid of the state space is hopeless: even ten points
per axis is a million simulations, and twenty-five per axis is 244 million.
"""
import numpy as np


class PlanarQuadrotor:
    name = "quadrotor"
    labels = ["px (m)", "pz (m)", "phi (rad)", "vx (m/s)", "vz (m/s)", "wphi (rad/s)"]
    # per-state half-width for sampling TRAINING starts (near hover, small angle)
    sample_box = np.array([0.7, 0.7, 0.35, 0.9, 0.9, 1.2])
    n_input = 2

    def __init__(self, m=1.0, I=0.05, r=0.2, g=9.81):
        self.m, self.I, self.r, self.g = m, I, r, g
        self.x_eq = np.zeros(6)
        self.u_eq = np.array([m * g / 2.0, m * g / 2.0])

    def f(self, x, u):
        px, pz, phi, vx, vz, w = x
        u1, u2 = u[0], u[1]
        s, c = np.sin(phi), np.cos(phi)
        thrust = (u1 + u2) / self.m
        ax = -thrust * s
        az = thrust * c - self.g
        aphi = self.r * (u1 - u2) / self.I
        return np.array([vx, vz, w, ax, az, aphi])


SYSTEMS = {"quadrotor": PlanarQuadrotor}


def get_system(name):
    return SYSTEMS[name]()


if __name__ == "__main__":
    import control as ct
    plant = PlanarQuadrotor()
    n = len(plant.x_eq)
    np.set_printoptions(precision=4, suppress=True)

    # 1) hover must be an equilibrium: f(x_eq, u_eq) = 0
    fe = plant.f(plant.x_eq, plant.u_eq)
    print("f(x_eq, u_eq) =", fe, " (should be all zero)")
    assert np.allclose(fe, 0.0), "hover is not an equilibrium"

    # 2) linearize at hover with central differences
    eps = 1e-6
    A = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = eps
        A[:, i] = (plant.f(plant.x_eq + e, plant.u_eq) - plant.f(plant.x_eq - e, plant.u_eq)) / (2 * eps)
    B = np.zeros((n, plant.n_input))
    for j in range(plant.n_input):
        e = np.zeros(plant.n_input); e[j] = eps
        B[:, j] = (plant.f(plant.x_eq, plant.u_eq + e) - plant.f(plant.x_eq, plant.u_eq - e)) / (2 * eps)
    print("\ncontinuous A =\n", A)
    print("continuous B =\n", B)
    w = np.linalg.eigvals(A)
    print("\neigenvalues of A:", np.round(w, 4))
    print("asymptotically stable:", bool(np.all(w.real < -1e-9)), "(expect False: integrators)")
    rank = np.linalg.matrix_rank(ct.ctrb(A, B))
    print(f"controllability rank: {rank} (need {n})")

    # 3) check the analytic linear couplings: ax = -g*phi, az = (du1+du2)/m, aphi = r(du1-du2)/I
    print("\nanalytic checks:")
    print(f"  dax/dphi   = {A[3,2]:.4f}   expect -g = {-plant.g:.4f}")
    print(f"  daz/du1    = {B[4,0]:.4f}   expect 1/m = {1/plant.m:.4f}")
    print(f"  dwphi/du1  = {B[5,0]:.4f}   expect r/I = {plant.r/plant.I:.4f}")
    print(f"  dwphi/du2  = {B[5,1]:.4f}   expect -r/I = {-plant.r/plant.I:.4f}")
