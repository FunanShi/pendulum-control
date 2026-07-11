"""Equilibria and linearization of the cart-mounted double pendulum.

Analytic Jacobians A = ∂f/∂z, B_lin = ∂f/∂u about an equilibrium,
cross-checked against finite differences in tests; controllability via the
generic ctrb/obsv helpers in model.linearize.

Units/frame: SI; z = [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂].
"""
from __future__ import annotations

import numpy as np

from dpend.model.cart_dynamics import mass_matrix
from dpend.model.cart_params import CartParams
from dpend.model.linearize import _gravity_jacobian

# Equilibria (state [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂], m & rad): upright = origin by convention.
UPRIGHT = np.zeros(6)
HANGING = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])


def linearize(cp: CartParams, z_eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic A = ∂f/∂z, B_lin = ∂f/∂u at an equilibrium (q̇=0, u=0).

    Velocity-coupling terms vanish at an equilibrium (as in model.linearize),
    leaving

        A = [[0₃, I₃], [−M⁻¹ ∂g/∂q, −M⁻¹ diag(bc,b₁,b₂)]],
        B_lin = [[0₃], [M⁻¹ B]],

    where B = [1, 0, 0]ᵀ (cart actuation only). Only valid at equilibria;
    FD cross-check in tests.
    """
    x_eq, theta1, theta2 = float(z_eq[0]), float(z_eq[1]), float(z_eq[2])
    Minv = np.linalg.inv(mass_matrix(cp, theta1, theta2))

    # Gravity Jacobian (3×3): cart row/col is zero; the lower-right 2×2 is the
    # fixed-pivot block from linearize.py (single source of truth).
    G_fixed = _gravity_jacobian(cp.pend, theta1, theta2)
    G = np.zeros((3, 3))
    G[1:, 1:] = G_fixed

    Dv = np.diag([cp.bc, cp.pend.b1, cp.pend.b2])

    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = -Minv @ G
    A[3:, 3:] = -Minv @ Dv

    B_vec = np.array([1.0, 0.0, 0.0])
    B_lin = np.vstack([np.zeros((3, 1)), (Minv @ B_vec).reshape(3, 1)])

    return A, B_lin
