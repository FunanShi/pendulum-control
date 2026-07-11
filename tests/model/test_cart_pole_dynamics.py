"""Cart-pole (single-pole) internal-consistency layer; oracles live in
test_cart_pole_oracles.py. Mirrors test_cart_dynamics.py, one DOF fewer
(q = [x, theta])."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.model.cart_pole_params import CartPoleParams

RNG = np.random.default_rng(0)


def test_mass_matrix_symmetric_positive_definite():
    """Mirrors test_dynamics.py's M-PD test: a theta GRID (not random draws),
    since M depends only on theta here (no second angle to sample)."""
    from dpend.model.cart_pole_dynamics import mass_matrix

    cp = CartPoleParams()
    for theta in np.linspace(-np.pi, np.pi, 73):
        M = mass_matrix(cp, float(theta))
        assert M.shape == (2, 2)
        np.testing.assert_allclose(M, M.T, atol=1e-14)  # exact-symmetric construction; fp noise only
        assert np.linalg.eigvalsh(M).min() > 0


def test_equilibria_including_offset_x():
    from dpend.model.cart_pole_dynamics import f

    cp = CartPoleParams()
    for z_eq in (np.zeros(4),
                 np.array([0.0, np.pi, 0.0, 0.0]),
                 np.array([0.7, 0.0, 0.0, 0.0])):  # translation family: any in-rail x
        np.testing.assert_allclose(f(z_eq, np.zeros(1), cp), np.zeros(4), atol=1e-12)


def test_mdot_minus_2c_skew():
    from dpend.model.cart_pole_dynamics import coriolis_matrix, mass_matrix

    cp = CartPoleParams()
    eps = 1e-7  # central FD step: truncation ~eps^2 vs roundoff ~1e-16/eps balance (test_dynamics.py rationale)
    for _ in range(50):
        theta = float(RNG.uniform(-np.pi, np.pi))
        thetadot = float(RNG.uniform(-5.0, 5.0))
        dM = (mass_matrix(cp, theta + eps) - mass_matrix(cp, theta - eps)) / (2 * eps) * thetadot
        N = dM - 2.0 * coriolis_matrix(cp, theta, thetadot)
        np.testing.assert_allclose(N + N.T, np.zeros((2, 2)), atol=1e-6)  # FD-limited


def test_force_moves_cart_and_tips_pole():
    from dpend.model.cart_pole_dynamics import f

    cp = CartPoleParams()
    zdot = f(np.zeros(4), np.array([1.0]), cp)  # 1 N at upright
    assert zdot[2] > 0            # cart accelerates +x
    assert abs(zdot[3]) > 1e-3    # coupling tips the pole


def test_stop_force_inside_rail_zero_and_restoring_outside():
    from dpend.model.cart_pole_dynamics import stop_force

    cp = CartPoleParams()
    assert stop_force(cp, 1.4, 3.0) == 0.0             # inside rail: exactly zero
    assert stop_force(cp, cp.L_rail + 0.1, 0.0) < 0    # beyond +end: pushes back
    assert stop_force(cp, -cp.L_rail - 0.1, 0.0) > 0


def test_energy_at_rest_and_stop_spring_term():
    from dpend.model.cart_pole_dynamics import energy

    cp = CartPoleParams()
    v_up = cp.mp * cp.g0 * cp.l
    assert energy(cp, np.zeros(4)) == pytest.approx(v_up)                       # x-independent V
    assert energy(cp, np.array([0.9, 0.0, 0.0, 0.0])) == pytest.approx(v_up)
    over = cp.L_rail + 0.2
    assert energy(cp, np.array([over, 0.0, 0.0, 0.0])) == pytest.approx(
        v_up + 0.5 * cp.k_stop * 0.2**2)                                        # stop spring PE while in contact


def test_viscous_damping_dissipates_and_kinematic_identity():
    """Mirror of test_cart_dynamics.py's version: a bc/bp swap or a qdot
    mis-stack would otherwise pass silently (the sympy oracle in
    test_cart_pole_oracles.py can't see friction -- it's outside the
    Lagrangian)."""
    from dpend.model.cart_pole_dynamics import f, mass_matrix

    cp0 = CartPoleParams()
    cpb = CartPoleParams(bc=0.3, bp=0.2)  # [N*s/m], [N*m*s/rad]
    z = np.array([0.2, 0.3, 0.8, -1.2])   # in-rail, nonzero velocities
    u0 = np.zeros(1)
    # kinematic identity at nonzero velocity: zdot[:2] must be exactly qdot
    np.testing.assert_array_equal(f(z, u0, cp0)[:2], z[2:])
    # damping shifts q_ddot opposite qdot: qdot^T(M*Delta q_ddot) = -qdot^T diag(bc,bp) qdot < 0
    dq_acc = f(z, u0, cpb)[2:] - f(z, u0, cp0)[2:]
    M = mass_matrix(cp0, float(z[1]))
    assert float(z[2:] @ (M @ dq_acc)) < 0.0


# --- EnergyShapingCapable primitives ---

def test_pendulum_energy_at_upright_and_hanging_rest():
    from dpend.model.cart_pole_dynamics import energy_upright, pendulum_energy

    cp = CartPoleParams()
    e_up = energy_upright(cp)
    assert pendulum_energy(cp, np.zeros(4)) == pytest.approx(e_up)
    assert e_up == pytest.approx(cp.mp * cp.g0 * cp.l)

    e_hang = pendulum_energy(cp, np.array([0.0, np.pi, 0.0, 0.0]))
    assert e_hang == pytest.approx(-cp.mp * cp.g0 * cp.l)
    assert e_hang == pytest.approx(-e_up)


def test_accel_to_force_is_exact_pfl_inverse():
    """accel_to_force is the exact inverse of f's cart-acceleration channel
    for random in-rail states and a_cmd, on a frictionless instance (its
    derivation assumes bc=bp=0, tau_ext=0, in-rail). atol=1e-9: exact algebra,
    so any sign/term error shows far above float64 noise."""
    from dpend.model.cart_pole_dynamics import accel_to_force, f

    cp_frictionless = CartPoleParams(bc=0.0, bp=0.0)
    rng = np.random.default_rng(7)
    max_resid = 0.0
    for _ in range(30):
        x = float(rng.uniform(-1.0, 1.0))            # in-rail (L_rail=1.5)
        theta = float(rng.uniform(-np.pi, np.pi))
        xdot = float(rng.uniform(-3.0, 3.0))
        thetadot = float(rng.uniform(-5.0, 5.0))
        z = np.array([x, theta, xdot, thetadot])
        a_cmd = float(rng.uniform(-10.0, 10.0))

        u = accel_to_force(cp_frictionless, z, a_cmd)
        zdot = f(z, np.array([u]), cp_frictionless)
        resid = abs(float(zdot[2]) - a_cmd)
        max_resid = max(max_resid, resid)
        assert resid < 1e-9

    print(f"\n[accel_to_force PFL inverse] max residual over 30 random draws: {max_resid:.3e}")
