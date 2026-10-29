"""Nonlinear plant models.

Each system exposes:
    f(x, u)   -> dx/dt  (continuous-time dynamics)
    x_eq, u_eq          (the equilibrium to stabilize)
    labels              (state variable names, for plots)
"""
import numpy as np


class Duffing:
    """x'' + delta*x' + alpha*x + beta*x^3 = u

    With alpha < 0, beta > 0 this is the double-well Duffing oscillator:
    stable wells at x = +/-sqrt(-alpha/beta) and an unstable equilibrium at
    x = 0. The control objective is to stabilize the unstable equilibrium.
    """
    name = "duffing"
    labels = ["x (position)", "v (velocity)"]

    def __init__(self, delta=0.3, alpha=-1.0, beta=1.0):
        self.delta, self.alpha, self.beta = delta, alpha, beta
        self.x_eq = np.array([0.0, 0.0])
        self.u_eq = 0.0

    def f(self, x, u):
        pos, vel = x
        acc = u - self.delta * vel - self.alpha * pos - self.beta * pos**3
        return np.array([vel, acc])


class Pendulum:
    """theta'' = -(g/l) sin(theta) - b*theta' + u/(m l^2)

    theta = 0 is the downward (stable) rest position; theta = pi is upright
    (unstable). The control objective is to stabilize the upright position.
    """
    name = "pendulum"
    labels = ["theta (rad)", "omega (rad/s)"]

    def __init__(self, g=9.81, l=1.0, m=1.0, b=0.1):
        self.g, self.l, self.m, self.b = g, l, m, b
        self.x_eq = np.array([np.pi, 0.0])
        self.u_eq = 0.0

    def f(self, x, u):
        th, om = x
        acc = -(self.g / self.l) * np.sin(th) - self.b * om + u / (self.m * self.l**2)
        return np.array([om, acc])


SYSTEMS = {"duffing": Duffing, "pendulum": Pendulum}


def get_system(name):
    return SYSTEMS[name]()
