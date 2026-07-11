"""Equilibria and linearization of the double pendulum.

Analytic Jacobians A = ∂f/∂x, B = ∂f/∂u about an equilibrium (cross-checked
against finite differences in tests), plus ctrb/obsv helpers with rank and
condition number for comparing the Acrobot/Pendubot/full configs.

Units/frame: SI; x = [θ₁, θ₂, θ̇₁, θ̇₂].
"""
from __future__ import annotations

import numpy as np

from dpend.model.dynamics import mass_matrix
from dpend.model.params import Params

# Equilibria (state [θ₁,θ₂,θ̇₁,θ̇₂], rad & rad/s): upright = origin by convention.
UPRIGHT = np.zeros(4)
HANGING = np.array([np.pi, 0.0, 0.0, 0.0])


def _gravity_jacobian(p: Params, theta1: float, theta2: float) -> np.ndarray:
    """∂g/∂q (2,2), symmetric (it is a Hessian of V). Analytic."""
    c1 = np.cos(theta1)
    c12 = np.cos(theta1 + theta2)
    k1 = p.m1 * p.lc1 + p.m2 * p.l1
    k2 = p.m2 * p.lc2
    g11 = -p.g0 * (k1 * c1 + k2 * c12)
    g12 = -p.g0 * k2 * c12
    # NOT a typo: ∂g₂/∂θ₂ = ∂g₂/∂θ₁ = ∂g₁/∂θ₂ = g12 for this plant — link 2's
    # potential depends on q only through (θ₁+θ₂), so all three match.
    return np.array([[g11, g12],
                     [g12, g12]])


def linearize(p: Params, B: np.ndarray, x_eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytic A = ∂f/∂x, B_lin = ∂f/∂u at an equilibrium (q̇=0, u=0).

    Velocity-coupling terms vanish at an equilibrium (C·q̇ is quadratic in q̇;
    ∂(M⁻¹)/∂q multiplies a zero residual torque), leaving

        A = [[0₂, I₂], [−M⁻¹ ∂g/∂q, −M⁻¹ diag(b)]],   B_lin = [[0], [M⁻¹ B]].

    Only valid at equilibria; FD cross-check in tests.
    """
    theta1, theta2 = float(x_eq[0]), float(x_eq[1])
    Minv = np.linalg.inv(mass_matrix(p, theta2))
    G = _gravity_jacobian(p, theta1, theta2)
    Dv = np.diag([p.b1, p.b2])
    A = np.zeros((4, 4))
    A[:2, 2:] = np.eye(2)
    A[2:, :2] = -Minv @ G
    A[2:, 2:] = -Minv @ Dv
    B_lin = np.vstack([np.zeros((2, B.shape[1])), Minv @ B])
    return A, B_lin


def ctrb(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Controllability matrix [B, AB, …, Aⁿ⁻¹B], shape (n, n·m). n from A."""
    n = A.shape[0]
    blocks, Bk = [], B
    for _ in range(n):
        blocks.append(Bk)
        Bk = A @ Bk
    return np.hstack(blocks)


def obsv(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Observability matrix [C; CA; …; CAⁿ⁻¹], shape (n·p, n). Dual of ctrb."""
    n = A.shape[0]
    blocks, Ck = [], C
    for _ in range(n):
        blocks.append(Ck)
        Ck = Ck @ A
    return np.vstack(blocks)


def rank_and_cond(M: np.ndarray) -> tuple[int, float]:
    """(numerical rank, σ_max/σ_min). cond = inf when rank-deficient."""
    s = np.linalg.svd(M, compute_uv=False)
    tol = s.max() * max(M.shape) * np.finfo(float).eps
    rank = int((s > tol).sum())
    cond = float(s.max() / s.min()) if s.min() > 0 else float("inf")
    return rank, cond
