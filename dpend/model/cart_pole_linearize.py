"""Equilibria and linearization of the single-pole cart-pole plant.

Analytic Jacobians A = df/dz, B_lin = df/du about an equilibrium,
cross-checked against finite differences in tests; controllability via the
generic ctrb/rank_and_cond helpers in model.linearize.

Units/frame: SI; z = [x, theta, xdot, thetadot].
"""
from __future__ import annotations

import numpy as np

from dpend.model.cart_pole_dynamics import mass_matrix
from dpend.model.cart_pole_params import CartPoleParams

# Equilibria (state [x, theta, xdot, thetadot], m & rad): upright = origin by convention.
UPRIGHT = np.zeros(4)
HANGING = np.array([0.0, np.pi, 0.0, 0.0])


def linearize(cp: CartPoleParams, z_eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic A = df/dz, B_lin = df/du at an equilibrium (qdot=0, u=0).

    Velocity-coupling terms vanish at an equilibrium (as in model.linearize),
    leaving

        A = [[0_2, I_2], [-M^-1 G, -M^-1 D]],   B_lin = [[0_2], [M^-1 B]],

    where G = dg/dq (2x2) = [[0, 0], [0, -mp*g0*l*cos(theta)]] (gravity
    depends on theta only), D = diag(bc, bp), and B = [1, 0]^T (cart
    actuation only). Only valid at equilibria; FD cross-check in tests.
    """
    theta = float(z_eq[1])
    Minv = np.linalg.inv(mass_matrix(cp, theta))

    G = np.array([[0.0, 0.0],
                  [0.0, -cp.mp * cp.g0 * cp.l * np.cos(theta)]])
    Dv = np.diag([cp.bc, cp.bp])

    A = np.zeros((4, 4))
    A[:2, 2:] = np.eye(2)
    A[2:, :2] = -Minv @ G
    A[2:, 2:] = -Minv @ Dv

    B_vec = np.array([1.0, 0.0])
    B_lin = np.vstack([np.zeros((2, 1)), (Minv @ B_vec).reshape(2, 1)])

    return A, B_lin
