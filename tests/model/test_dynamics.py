"""Fixed-pivot dynamics internal-consistency checks; independent oracles
(scipy energy, sympy) live in test_dynamics_oracles.py."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.model.params import Params, actuation_matrix

RNG = np.random.default_rng(0)  # fixed seed: reproducible property sampling


def test_mass_matrix_symmetric_positive_definite():
    """M(θ₂) symmetric PD for all θ₂ (θ₁-independence is enforced by the
    signature, which takes no θ₁)."""
    from dpend.model.dynamics import mass_matrix

    p = Params()
    for theta2 in np.linspace(-np.pi, np.pi, 73):
        M = mass_matrix(p, float(theta2))
        assert M.shape == (2, 2)
        np.testing.assert_allclose(M, M.T, atol=1e-14)
        eig = np.linalg.eigvalsh(M)
        assert eig.min() > 0.0, f"M not PD at theta2={theta2}: eigs={eig}"


def test_equilibria_zero_acceleration():
    """ẋ = 0 at upright (0,0,0,0) and hanging (π,0,0,0) with u = 0."""
    from dpend.model.dynamics import f

    p = Params()
    B = actuation_matrix("acrobot")
    u0 = np.zeros(B.shape[1])
    for x_eq in (np.zeros(4), np.array([np.pi, 0.0, 0.0, 0.0])):
        np.testing.assert_allclose(f(x_eq, u0, p, B), np.zeros(4), atol=1e-12)


def test_generic_state_is_not_equilibrium():
    """Sanity guard: a tipped state must accelerate (catches an accidentally
    all-zero f)."""
    from dpend.model.dynamics import f

    p = Params()
    B = actuation_matrix("acrobot")
    xdot = f(np.array([0.3, -0.2, 0.0, 0.0]), np.zeros(1), p, B)
    assert np.linalg.norm(xdot[2:]) > 1e-3


def test_mdot_minus_2c_skew_symmetric():
    """Passivity: N = Ṁ − 2C skew-symmetric. Ṁ via central FD of mass_matrix
    w.r.t. θ₂ times θ̇₂ (M depends on θ₂ only)."""
    from dpend.model.dynamics import coriolis_matrix, mass_matrix

    p = Params()
    eps = 1e-7  # FD step: sqrt(eps_machine)-ish, error ~1e-14/1e-7 + 1e-14·|M''|
    for _ in range(50):
        theta2 = float(RNG.uniform(-np.pi, np.pi))
        w1, w2 = RNG.uniform(-5.0, 5.0, size=2)  # rad/s, generous range
        dM_dth2 = (mass_matrix(p, theta2 + eps) - mass_matrix(p, theta2 - eps)) / (2 * eps)
        Mdot = dM_dth2 * w2  # chain rule: Ṁ = (∂M/∂θ₂)·θ̇₂
        N = Mdot - 2.0 * coriolis_matrix(p, theta2, float(w1), float(w2))
        np.testing.assert_allclose(N + N.T, np.zeros((2, 2)), atol=1e-6)  # FD-limited


def test_forward_dynamics_shapes_and_actuation():
    """f returns (4,); ẋ[:2] = q̇; actuated joint responds to torque, and with
    B=acrobot a shoulder torque channel must not exist (u ∈ ℝ¹ → elbow)."""
    from dpend.model.dynamics import f

    p = Params()
    x = np.array([0.1, -0.2, 0.3, 0.4])
    B = actuation_matrix("acrobot")
    xdot0 = f(x, np.zeros(1), p, B)
    xdot1 = f(x, np.array([1.0]), p, B)   # +1 N·m at the elbow
    assert xdot0.shape == (4,)
    np.testing.assert_allclose(xdot0[:2], x[2:])          # kinematic identity
    assert not np.allclose(xdot0[2:], xdot1[2:])          # torque changes q̈
    np.testing.assert_allclose(xdot0[:2], xdot1[:2])      # …but not q̇ passthrough


def test_energy_value_at_rest():
    """E at rest = V(q) with V zero at pivot height: upright = +g₀(m₁l_c1+m₂l₁+m₂l_c2),
    hanging = −same. Pins the potential's sign convention and zero point."""
    from dpend.model.dynamics import energy

    p = Params()
    v_up = p.g0 * (p.m1 * p.lc1 + p.m2 * p.l1 + p.m2 * p.lc2)  # J
    assert energy(p, np.zeros(4)) == pytest.approx(v_up)
    assert energy(p, np.array([np.pi, 0.0, 0.0, 0.0])) == pytest.approx(-v_up)


def test_friction_dissipates():
    """With b>0 and u=0, Ė = −q̇ᵀD q̇ < 0: check via one f() evaluation,
    Ė = q̇ᵀ(Mq̈) + ½q̇ᵀṀq̇ + q̇ᵀg — equivalently just check q̈ gains a term
    opposing q̇ relative to the frictionless case."""
    from dpend.model.dynamics import f

    p0 = Params()
    pb = Params(b1=0.5, b2=0.5)  # N·m·s·rad⁻¹
    B = actuation_matrix("acrobot")
    x = np.array([0.2, 0.1, 1.0, -1.0])
    dq_acc = f(x, np.zeros(1), pb, B)[2:] - f(x, np.zeros(1), p0, B)[2:]
    # friction torque = −diag(b)q̇ = [−0.5, +0.5]; q̈ shift = M⁻¹·that; check sign
    # via the work rate: q̇ᵀ · (M·Δq̈) = −q̇ᵀ diag(b) q̇ < 0
    from dpend.model.dynamics import mass_matrix

    M = mass_matrix(p0, float(x[1]))
    assert float(x[2:] @ (M @ dq_acc)) < 0.0
