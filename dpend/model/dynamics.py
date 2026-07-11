"""Equations of motion for the planar double pendulum.

Manipulator-form terms M(q), C(q, q̇), g(q) and the forward dynamics

    ẋ = f(x, u)   with   M(q) q̈ + C(q, q̇) q̇ + g(q) + friction(q̇) = B u,

where x = [θ₁, θ₂, θ̇₁, θ̇₂]. Hand-derived closed form, cross-checked against a
symbolic (sympy) derivation in the tests.

Units/frame: SI; angles from the upward vertical, upright = origin (docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import numpy as np

from dpend.model.params import Params

# Lumped constants:  a = I₁+I₂+m₁l_c1²+m₂(l₁²+l_c2²),
# b = m₂ l₁ l_c2,  d = I₂+m₂l_c2²,  h = b·sinθ₂.


def _abd(p: Params) -> tuple[float, float, float]:
    a = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2)
    b = p.m2 * p.l1 * p.lc2
    d = p.I2 + p.m2 * p.lc2**2
    return a, b, d


def mass_matrix(p: Params, theta2: float) -> np.ndarray:
    """Inertia M(θ₂) [kg·m²], symmetric PD; depends on θ₂ only (no θ₁ arg)."""
    a, b, d = _abd(p)
    c2 = np.cos(theta2)
    return np.array([[a + 2.0 * b * c2, d + b * c2],
                     [d + b * c2,       d          ]])


def coriolis_matrix(p: Params, theta2: float, w1: float, w2: float) -> np.ndarray:
    """Coriolis/centrifugal C(q,q̇) [N·m·s·rad⁻¹], Christoffel form so that
    Ṁ − 2C is skew-symmetric (passivity). C @ [w1,w2] gives the torques."""
    _, b, _ = _abd(p)
    h = b * np.sin(theta2)
    return np.array([[-h * w2, -h * (w1 + w2)],
                     [ h * w1,  0.0          ]])


def gravity(p: Params, theta1: float, theta2: float) -> np.ndarray:
    """Gravity torque g(q) = ∂V/∂q [N·m]. Leading minus: angles are measured
    from the upward vertical, so upright is the potential maximum
    (destabilizing — the inverted behavior we want)."""
    s1 = np.sin(theta1)
    s12 = np.sin(theta1 + theta2)
    k1 = p.m1 * p.lc1 + p.m2 * p.l1
    k2 = p.m2 * p.lc2
    return np.array([-p.g0 * (k1 * s1 + k2 * s12),
                     -p.g0 * k2 * s12])


def friction(p: Params, qdot: np.ndarray) -> np.ndarray:
    """Viscous joint friction D(q̇) = diag(b₁,b₂)·q̇ [N·m]. Default b=0."""
    return np.array([p.b1 * qdot[0], p.b2 * qdot[1]])


def f(x: np.ndarray, u: np.ndarray, p: Params, B: np.ndarray) -> np.ndarray:
    """Forward dynamics ẋ = f(x, u).  x = [θ₁,θ₂,θ̇₁,θ̇₂] [rad, rad/s],
    u ∈ ℝᵐ [N·m], τ = B u.  q̈ = M⁻¹(Bu − Cq̇ − g − Dq̇)."""
    theta1, theta2 = float(x[0]), float(x[1])
    qdot = x[2:]
    M = mass_matrix(p, theta2)
    C = coriolis_matrix(p, theta2, float(qdot[0]), float(qdot[1]))
    tau = B @ np.asarray(u, dtype=float)
    qddot = np.linalg.solve(M, tau - C @ qdot - gravity(p, theta1, theta2) - friction(p, qdot))
    return np.concatenate([qdot, qddot])


def energy(p: Params, x: np.ndarray) -> float:
    """Total energy E = ½q̇ᵀM(θ₂)q̇ + V(q) [J].
    V = g₀((m₁l_c1+m₂l₁)cosθ₁ + m₂l_c2·cos(θ₁+θ₂)), zero at pivot height;
    conserved when u = 0 and friction = 0 (verified in tests)."""
    theta1, theta2 = float(x[0]), float(x[1])
    qdot = x[2:]
    T = 0.5 * float(qdot @ (mass_matrix(p, theta2) @ qdot))
    k1 = p.m1 * p.lc1 + p.m2 * p.l1
    k2 = p.m2 * p.lc2
    V = p.g0 * (k1 * np.cos(theta1) + k2 * np.cos(theta1 + theta2))
    return T + V
