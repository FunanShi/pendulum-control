"""Plant protocol conformance + wrapper equivalence (each plant wrapper must
be a zero-behavior-change adapter over its verified dynamics module)."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.model.params import Params, actuation_matrix


def test_fixed_pivot_plant_conforms_and_matches_dynamics():
    from dpend.model.dynamics import energy, f
    from dpend.model.plant import Plant, fixed_pivot_plant

    p = fixed_pivot_plant(Params(), "acrobot")
    assert isinstance(p, Plant)
    assert (p.n, p.m) == (4, 1)
    z = np.array([0.3, -0.2, 0.5, -0.1])
    u = np.array([0.7])
    # adapter equivalence: identical ẋ and E to the underlying dynamics functions (exact — same code path)
    np.testing.assert_array_equal(p.f(z, u), f(z, u, Params(), actuation_matrix("acrobot")))
    assert p.energy(z) == energy(Params(), z)
    assert len(p.state_labels) == 4 and len(p.input_labels) == 1
    assert p.rail is None and p.reach == 2.0  # l1+l2 for defaults [m]


def test_fixed_pivot_fk_matches_viz_convention():
    from dpend.model.plant import fixed_pivot_plant

    p = fixed_pivot_plant()
    cart, pts = p.fk(np.zeros(4))
    assert cart is None
    np.testing.assert_allclose(pts, [[0, 0], [0, 1], [0, 2]], atol=1e-12)  # base, elbow, tip upright


def test_fixed_pivot_tau_ext_channel():
    """tau_ext ∈ ℝ² adds generalized torque: pushing joint 1 changes q̈ like B would."""
    from dpend.model.plant import fixed_pivot_plant

    p = fixed_pivot_plant()
    z = np.array([0.3, -0.2, 0.0, 0.0])
    a0 = p.f(z, np.zeros(1))
    a1 = p.f(z, np.zeros(1), tau_ext=np.array([1.0, 0.0]))
    assert not np.allclose(a0[2:], a1[2:])


def test_registry_open():
    from dpend.model.plant import PLANTS

    assert "fixed" in PLANTS and callable(PLANTS["fixed"])


def test_plant_linearize_delegates_exactly():
    """Plant.linearize(z_eq) is the plant-generic (A, B_lin) surface for
    controller factories; each plant delegates to its verified linearize
    module — outputs bit-exact (same code path, no re-derivation)."""
    from dpend.model import cart_linearize as cl
    from dpend.model import linearize as fl
    from dpend.model.cart_params import CartParams
    from dpend.model.plant import cart_plant, fixed_pivot_plant

    fp = fixed_pivot_plant(Params(), "acrobot")
    A, B = fp.linearize(fp.upright)
    A_ref, B_ref = fl.linearize(Params(), actuation_matrix("acrobot"), fl.UPRIGHT)
    np.testing.assert_array_equal(A, A_ref)
    np.testing.assert_array_equal(B, B_ref)

    cp = cart_plant()
    A, B = cp.linearize(cp.upright)
    A_ref, B_ref = cl.linearize(CartParams(), cl.UPRIGHT)
    np.testing.assert_array_equal(A, A_ref)
    np.testing.assert_array_equal(B, B_ref)


def test_plant_generic_consumer_needs_no_plant_specific_imports():
    """A consumer needing (A, B) — the LQR factory — works on every registered
    plant through the protocol alone; rank == n re-states the controllability
    gates the plants already passed at module level."""
    from dpend.model.linearize import ctrb, rank_and_cond
    from dpend.model.plant import PLANTS, Plant

    for name, factory in PLANTS.items():
        plant = factory()
        assert isinstance(plant, Plant)
        assert plant.upright.shape == (plant.n,)
        A, B_lin = plant.linearize(plant.upright)
        assert A.shape == (plant.n, plant.n) and B_lin.shape == (plant.n, plant.m)
        rank, _ = rank_and_cond(ctrb(A, B_lin))
        assert rank == plant.n, f"{name}: not controllable at upright?!"


def test_plant_upright_is_fresh_copy():
    """upright returns a copy — mutating it must not poison the module constant."""
    from dpend.model.plant import cart_plant

    p = cart_plant()
    u1 = p.upright
    u1[0] = 99.0
    np.testing.assert_array_equal(p.upright, np.zeros(6))


# --- hanging + lqr_weights (Plant protocol additions) ---

def test_fixed_and_cart_plant_hanging_and_lqr_weights_match_existing_defaults():
    """The fixed and cart plants' lqr_weights reproduce the historical
    n-keyed Q/R defaults exactly — a mismatch here would silently change
    LQR/MPC behavior."""
    from dpend.model import cart_linearize as cl
    from dpend.model import linearize as fl
    from dpend.model.plant import cart_plant, fixed_pivot_plant

    fp = fixed_pivot_plant(Params(), "acrobot")
    np.testing.assert_array_equal(fp.hanging, fl.HANGING)
    Q, R = fp.lqr_weights
    assert Q == [10.0, 10.0, 1.0, 1.0]
    assert R == [0.1] * fp.m

    cp_plant = cart_plant()
    np.testing.assert_array_equal(cp_plant.hanging, cl.HANGING)
    Q, R = cp_plant.lqr_weights
    assert Q == [10.0, 50.0, 50.0, 1.0, 5.0, 5.0]
    assert R == [0.1]


def test_fixed_plant_hanging_is_fresh_copy():
    from dpend.model.plant import fixed_pivot_plant

    p = fixed_pivot_plant()
    h1 = p.hanging
    h1[0] = 99.0
    assert p.hanging[0] != 99.0


# --- cart-pole plant ---

def test_cart_pole_plant_conforms_and_is_energy_shaping_capable():
    from dpend.model.cart_pole_params import CartPoleParams
    from dpend.model.plant import EnergyShapingCapable, Plant, cart_pole_plant

    p = cart_pole_plant()
    assert isinstance(p, (Plant, EnergyShapingCapable))
    assert isinstance(p, Plant)                          # stronger: both hold independently
    assert isinstance(p, EnergyShapingCapable)
    assert (p.n, p.m) == (4, 1)
    assert p.rail == CartPoleParams().L_rail
    assert p.reach == CartPoleParams().l
    assert len(p.state_labels) == 4 and len(p.input_labels) == 1


def test_cart_pole_plant_upright_hanging_fresh_copies():
    from dpend.model import cart_pole_linearize as cpl
    from dpend.model.plant import cart_pole_plant

    p = cart_pole_plant()
    np.testing.assert_array_equal(p.upright, cpl.UPRIGHT)
    np.testing.assert_array_equal(p.hanging, cpl.HANGING)
    assert p.upright.shape == (4,) and p.hanging.shape == (4,)

    u1 = p.upright
    u1[0] = 99.0
    np.testing.assert_array_equal(p.upright, np.zeros(4))  # fresh copy per call

    h1 = p.hanging
    h1[0] = 99.0
    np.testing.assert_array_equal(p.hanging, cpl.HANGING)  # fresh copy per call


def test_cart_pole_plant_lqr_weights_shape():
    from dpend.model.plant import cart_pole_plant

    p = cart_pole_plant()
    Q, R = p.lqr_weights
    assert len(Q) == p.n
    assert len(R) == p.m


def test_cart_pole_plant_linearize_delegates_exactly():
    from dpend.model import cart_pole_linearize as cpl
    from dpend.model.cart_pole_params import CartPoleParams
    from dpend.model.plant import cart_pole_plant

    p = cart_pole_plant()
    A, B = p.linearize(p.upright)
    A_ref, B_ref = cpl.linearize(CartPoleParams(), cpl.UPRIGHT)
    np.testing.assert_array_equal(A, A_ref)
    np.testing.assert_array_equal(B, B_ref)


def test_cart_pole_plant_fk():
    """Upright: pole tip directly above the cart at height l (l=0.5 default)."""
    from dpend.model.plant import cart_pole_plant

    p = cart_pole_plant()
    cart_xy, pts = p.fk(np.zeros(4))
    np.testing.assert_allclose(cart_xy, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(pts, [[0.0, 0.0], [0.0, 0.5]], atol=1e-12)


def test_cart_pole_plant_energy_shaping_methods_delegate():
    """Adapter equivalence: identical values to the free cart_pole_dynamics
    functions — same code path."""
    from dpend.model import cart_pole_dynamics as cpd
    from dpend.model.cart_pole_params import CartPoleParams
    from dpend.model.plant import cart_pole_plant

    p = cart_pole_plant()
    z = np.array([0.1, 0.2, -0.3, 0.4])
    assert p.energy_upright == pytest.approx(cpd.energy_upright(CartPoleParams()))
    assert p.pendulum_energy(z) == pytest.approx(cpd.pendulum_energy(CartPoleParams(), z))
    assert p.accel_to_force(z, 1.5) == pytest.approx(cpd.accel_to_force(CartPoleParams(), z, 1.5))


def test_registry_has_cartpole():
    from dpend.model.plant import PLANTS

    assert "cartpole" in PLANTS and callable(PLANTS["cartpole"])
