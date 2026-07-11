"""Cart plant internal-consistency layer; oracles in test_cart_oracles.py."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.model.cart_params import CartParams

RNG = np.random.default_rng(0)


def test_mass_matrix_spd_and_block_structure():
    from dpend.model.cart_dynamics import mass_matrix
    from dpend.model.dynamics import mass_matrix as fixed_M

    cp = CartParams()
    for th1, th2 in RNG.uniform(-np.pi, np.pi, size=(50, 2)):
        M = mass_matrix(cp, float(th1), float(th2))
        np.testing.assert_allclose(M, M.T, atol=1e-14)   # exact-symmetric construction; fp noise only
        assert np.linalg.eigvalsh(M).min() > 0
        # mutual oracle: θ-block ≡ the verified fixed-pivot M(θ₂)
        np.testing.assert_allclose(M[1:, 1:], fixed_M(cp.pend, float(th2)), atol=1e-14)


def test_equilibria_including_offset_x():
    from dpend.model.cart_dynamics import f

    cp = CartParams()
    for z_eq in (np.zeros(6),
                 np.array([0.0, np.pi, 0, 0, 0, 0]),
                 np.array([0.7, 0.0, 0, 0, 0, 0])):   # translation family: any in-rail x
        np.testing.assert_allclose(f(z_eq, np.zeros(1), cp), np.zeros(6), atol=1e-12)


def test_mdot_minus_2c_skew():
    from dpend.model.cart_dynamics import coriolis_matrix, mass_matrix

    cp = CartParams()
    eps = 1e-7  # central FD step (see test_dynamics.py rationale)
    for _ in range(50):
        th1, th2 = RNG.uniform(-np.pi, np.pi, 2)
        w1, w2 = RNG.uniform(-5, 5, 2)
        dM = ((mass_matrix(cp, th1 + eps, th2) - mass_matrix(cp, th1 - eps, th2)) / (2 * eps) * w1
              + (mass_matrix(cp, th1, th2 + eps) - mass_matrix(cp, th1, th2 - eps)) / (2 * eps) * w2)
        N = dM - 2 * coriolis_matrix(cp, th1, th2, w1, w2)
        np.testing.assert_allclose(N + N.T, np.zeros((3, 3)), atol=1e-6)  # FD-limited


def test_force_moves_cart_and_couples():
    from dpend.model.cart_dynamics import f

    cp = CartParams()
    zdot = f(np.zeros(6), np.array([1.0]), cp)   # 1 N at upright
    assert zdot[3] > 0            # cart accelerates +x
    assert abs(zdot[4]) > 1e-3    # coupling tips link 1


def test_stop_force_inside_rail_zero_and_restoring_outside():
    from dpend.model.cart_dynamics import stop_force

    cp = CartParams()
    assert stop_force(cp, 1.4, 3.0) == 0.0            # inside rail: exactly zero
    assert stop_force(cp, cp.L_rail + 0.1, 0.0) < 0   # beyond +end: pushes back
    assert stop_force(cp, -cp.L_rail - 0.1, 0.0) > 0


def test_energy_at_rest_and_stop_spring_term():
    from dpend.model.cart_dynamics import energy

    cp = CartParams()
    v_up = cp.pend.g0 * (cp.pend.m1 * cp.pend.lc1 + cp.pend.m2 * cp.pend.l1 + cp.pend.m2 * cp.pend.lc2)
    assert energy(cp, np.zeros(6)) == pytest.approx(v_up)          # x-independent V
    assert energy(cp, np.array([0.9, 0, 0, 0, 0, 0])) == pytest.approx(v_up)
    over = cp.L_rail + 0.2
    assert energy(cp, np.array([over, 0, 0, 0, 0, 0])) == pytest.approx(
        v_up + 0.5 * cp.k_stop * 0.2**2)                            # stop spring PE while in contact


def test_viscous_damping_dissipates_and_kinematic_identity():
    """Mirror of test_dynamics.py::test_friction_dissipates for the cart, plus
    the nonzero-velocity kinematic identity — a bc/b1/b2 swap or a qdot
    mis-stack would otherwise pass silently (the sympy oracle can't see
    friction: it's outside the Lagrangian)."""
    from dpend.model.cart_dynamics import f, mass_matrix
    from dpend.model.params import Params

    cp0 = CartParams()
    cpb = CartParams(bc=0.3, pend=Params(b1=0.2, b2=0.2))  # [N·s/m], [N·m·s·rad⁻¹]
    z = np.array([0.2, 0.3, -0.1, 0.8, 1.0, -1.2])  # in-rail, nonzero velocities
    u0 = np.zeros(1)
    # kinematic identity at nonzero velocity: ż[:3] must be exactly q̇
    np.testing.assert_array_equal(f(z, u0, cp0)[:3], z[3:])
    # damping shifts q̈ opposite q̇: q̇ᵀ(M·Δq̈) = −q̇ᵀdiag(bc,b1,b2)q̇ < 0
    dq_acc = f(z, u0, cpb)[3:] - f(z, u0, cp0)[3:]
    M = mass_matrix(cp0, float(z[1]), float(z[2]))
    assert float(z[3:] @ (M @ dq_acc)) < 0.0
