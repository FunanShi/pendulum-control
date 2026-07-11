"""EnergySwingUp unit tests (closed-form law sign/edge behavior) plus batch
swing-up reachability from hanging through simulate()."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.controllers.energy_swingup import EnergySwingUp
from dpend.model.plant import cart_pole_plant
from dpend.registry import CONTROLLERS, ESTIMATORS, SENSORS
from dpend.sim.simulator import simulate
from dpend.util.angles import angle_diff

# Catch-basin threshold in LQR V = e^T P e units — same number as registry.py's
# BASIN_V_LQR_CALIBRATED, cross-referenced rather than imported across the production/test boundary.
BASIN_V_THRESHOLD = 2.0


def test_upright_rest_zero_control():
    """At upright rest every term of the law is zero, so u == 0 — no spurious
    control action at the target."""
    plant = cart_pole_plant()
    ctrl = EnergySwingUp(plant)
    z = np.zeros(4)

    E = plant.pendulum_energy(z)
    assert E == plant.energy_upright  # exact: cos(0)=1, thetadot=0

    u = ctrl.update(0.0, z)
    assert u.shape == (1,)
    assert abs(u[0]) < 1e-9


def test_pump_sign_near_hanging():
    """Near hanging (theta=pi, small thetadot>0, so E<E_up): sign(u) equals
    sign((E_up-E)*thetadot*cos(theta)) — the Astrom-Furuta law, not flipped.
    theta=pi zeroes accel_to_force's sin(theta) offset terms and x=xdot=0
    zeroes centering, so u is an exact positive multiple of a_cmd and the
    check isolates the pump sign."""
    plant = cart_pole_plant()
    ctrl = EnergySwingUp(plant, k_E=1.0, k_x=2.0, k_d=1.0)
    theta, thetadot = np.pi, 0.3
    z = np.array([0.0, theta, 0.0, thetadot])

    E = plant.pendulum_energy(z)
    E_up = plant.energy_upright
    assert E < E_up  # near-hanging: far below the upright energy

    expected_sign = np.sign((E_up - E) * thetadot * np.cos(theta))
    u = ctrl.update(0.0, z)[0]
    print(f"\n[pump sign] E={E:.4f} J, E_up={E_up:.4f} J, expected_sign={expected_sign:+.0f}, u={u:.4f} N")
    assert np.sign(u) == expected_sign


def test_clip_saturates_at_u_max():
    """A state forcing |a_cmd| far outside ±u_max clips u to exactly ±u_max
    (np.clip returns the bound verbatim — exact equality, not approx)."""
    plant = cart_pole_plant()
    u_max = 0.5
    ctrl = EnergySwingUp(plant, k_E=1.0, k_x=2.0, k_d=1.0, u_max=u_max)
    z = np.array([50.0, np.pi, 0.0, 5.0])  # large x -> large centering term
    u = ctrl.update(0.0, z)[0]
    print(f"\n[clip] u={u} (u_max={u_max})")
    assert abs(u) == u_max


# --- Swing-up batch reachability (from hanging, through simulate()) ---

def test_swingup_reaches_catch_basin_from_hanging():
    """From hanging through simulate() (8 s, standard rates): pendulum energy
    comes within 5% of E_up, min V = e^T P e drops below BASIN_V_THRESHOLD
    (the trajectory genuinely enters the catch basin; measured min V ~0.016),
    and the cart never leaves the rail.
    The 8 s budget absorbs ~5 s of dead time: hanging is an exact fixed point,
    so the pump term (~thetadot) starts at 0 and only float64 noise seeds the
    swing — deterministic, not flaky."""
    plant = cart_pole_plant()
    ctrl = EnergySwingUp(plant)  # tuned defaults: k_E=1.5, k_x=4.0, k_d=3.0
    sensor = SENSORS["perfect"](plant, {})
    estimator = ESTIMATORS["identity"](plant, {})
    x0 = plant.hanging

    tel = simulate(plant=plant, x0=x0, duration_s=8.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
                   sensor=sensor, estimator=estimator, controller=ctrl, seed=0)

    t_s = tel.t_ns / 1e9
    E = np.array([plant.pendulum_energy(z) for z in tel.x_true])
    E_up = plant.energy_upright
    frac_E = np.abs(E - E_up) / abs(E_up)

    lqr = CONTROLLERS["lqr"](plant, {})
    P = lqr.P
    upright = plant.upright
    e = tel.x_true - upright
    e[:, 1] = np.array([angle_diff(float(th), upright[1]) for th in tel.x_true[:, 1]])
    Vs = np.einsum("ti,ij,tj->t", e, P, e)

    min_V = float(np.min(Vs))
    t_min_V = float(t_s[np.argmin(Vs)])
    max_abs_x = float(np.max(np.abs(tel.x_true[:, 0])))

    thetadot = tel.x_true[:, 3]
    reversals = int(np.sum(np.diff(np.sign(thetadot)) != 0))
    t_basin_E = float(t_s[np.argmax(frac_E < 0.05)]) if np.any(frac_E < 0.05) else None
    under_basin = Vs < BASIN_V_THRESHOLD
    t_basin_V = float(t_s[np.argmax(under_basin)]) if np.any(under_basin) else None
    peak_u = float(np.max(np.abs(tel.u)))

    print(f"\n[swingup reachability] velocity reversals (~swings) over the full 8s run: {reversals}")
    print(f"[swingup reachability] time E first within 5% of E_up: {t_basin_E} s")
    print(f"[swingup reachability] time-to-basin (V<{BASIN_V_THRESHOLD}): {t_basin_V} s")
    print(f"[swingup reachability] MEASURED min V = {min_V:.6f} at t={t_min_V:.3f} s "
          f"(this is the number swingup_factory's c_catch is calibrated from)")
    print(f"[swingup reachability] max|x| = {max_abs_x:.4f} m (rail={plant.rail} m)")
    print(f"[swingup reachability] peak|u| = {peak_u:.4f} N")
    print(f"[swingup reachability] final E = {E[-1]:.4f} J, E_up = {E_up:.4f} J, "
          f"final frac diff = {frac_E[-1]:.4f}")

    assert np.any(frac_E < 0.05), "pendulum_energy never reached within 5% of energy_upright"
    assert min_V < BASIN_V_THRESHOLD, f"min V={min_V:.6f} never dropped below {BASIN_V_THRESHOLD}"
    assert max_abs_x < plant.rail, f"cart left the rail: max|x|={max_abs_x:.4f} >= {plant.rail}"


def test_swingup_catches_nudged_near_upright_pole():
    """Regression: swing-up built at a near-upright pole with thetadot=1.5 rad/s
    hands off to the linear catch, not energy-pumping the pole over the top.
    The nudge is deep inside the catch's true basin but has V=2.25 — a tight
    c_catch=0.05 spins forever; the widened BASIN_V_THRESHOLD catches it."""
    plant = cart_pole_plant()
    ctrl = CONTROLLERS["swingup"](plant, {"catch": "lqr"})
    sensor = SENSORS["perfect"](plant, {})
    estimator = ESTIMATORS["identity"](plant, {})
    z0 = plant.upright.copy()
    z0[3] = 1.5  # thetadot kick [rad/s] -- a cart nudge imparts pole velocity

    tel = simulate(plant=plant, x0=z0, duration_s=8.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
                   sensor=sensor, estimator=estimator, controller=ctrl, seed=0)

    t_s = tel.t_ns / 1e9
    theta_err = np.array([abs(angle_diff(float(th), plant.upright[1]))
                          for th in tel.x_true[:, 1]])
    thetadot = np.abs(tel.x_true[:, 3])
    tail = t_s > (t_s[-1] - 1.0)  # settled and staying settled over the final 1 s
    print(f"\n[nudge-catch] final theta_err={theta_err[-1]:.4f} rad, "
          f"final |thetadot|={thetadot[-1]:.4f} rad/s, max wander={float(np.max(theta_err)):.3f} rad, "
          f"tail max theta_err={float(np.max(theta_err[tail])):.4f} rad, final mode={ctrl.mode}")
    assert np.max(theta_err[tail]) < 0.05, (
        f"pole never caught (spun): tail max |theta error|={np.max(theta_err[tail]):.4f} rad")
    assert np.max(thetadot[tail]) < 0.3, (
        f"pole not settled: tail max |thetadot|={np.max(thetadot[tail]):.4f} rad/s")


# --- swingup_factory + registry ---

def test_swingup_factory_auto_selects_energy_method_on_cartpole():
    """No `method` in params auto-selects "energy" (the only SWINGUP_METHODS
    entry whose capability the plant implements): a ModeSwitch with an
    EnergySwingUp child, mode SWINGING, and angle_idx inferred as (1,) from
    the lone "[rad]"-suffixed state label."""
    from dpend.controllers.mode_switch import ModeSwitch
    from dpend.registry import swingup_factory

    plant = cart_pole_plant()
    ctrl = swingup_factory(plant, {})
    assert isinstance(ctrl, ModeSwitch)
    assert isinstance(ctrl.swingup, EnergySwingUp)
    assert ctrl.mode == "SWINGING"
    assert ctrl.angle_idx == (1,)
    print(f"\n[swingup_factory] auto-selected method, angle_idx={ctrl.angle_idx}, "
          f"c_catch={ctrl.c_catch}, c_release={ctrl.c_release}")
    assert ctrl.c_catch == pytest.approx(BASIN_V_THRESHOLD)
    assert ctrl.c_release == pytest.approx(2 * BASIN_V_THRESHOLD)


def test_swingup_factory_catch_lqr_and_mpc_both_build_a_working_mode_switch():
    """catch="lqr" and catch="mpc" both build a ModeSwitch wired to the
    requested catch (shared P) that returns finite, correctly-shaped control
    from hanging (SWINGING) and from upright (CATCHING)."""
    from dpend.registry import swingup_factory

    plant = cart_pole_plant()
    for catch_name in ("lqr", "mpc"):
        ctrl = swingup_factory(plant, {"catch": catch_name})
        assert ctrl.P is not None
        np.testing.assert_array_equal(ctrl.P, ctrl.catch.P)

        ctrl.reset(0.0, plant.hanging)
        u_swinging = ctrl.update(0.0, plant.hanging)
        assert ctrl.mode == "SWINGING"
        assert u_swinging.shape == (1,)
        assert np.all(np.isfinite(u_swinging))

        ctrl.reset(0.0, plant.upright)
        u_catching = ctrl.update(0.0, plant.upright)
        assert ctrl.mode == "CATCHING"  # V=0 at upright < any positive c_catch
        assert u_catching.shape == (1,)
        assert np.all(np.isfinite(u_catching))
        print(f"\n[swingup_factory catch={catch_name}] u_swinging={u_swinging}, "
              f"u_catching={u_catching}")


def test_swingup_factory_raises_on_non_capable_plant_with_no_method():
    """PLANTS["cart"] lacks EnergyShapingCapable; with no explicit `method`
    the factory raises a ValueError naming the available methods."""
    from dpend.model.plant import PLANTS
    from dpend.registry import swingup_factory

    plant = PLANTS["cart"]()
    with pytest.raises(ValueError, match="energy"):
        swingup_factory(plant, {})


# ---------------------------------------------------------------------------
# c_catch basin containment via λ_min(P_catch, P_lqr): a provable guarantee
# that the catch-engage set lies inside the LQR-validated basin. Oracle: scipy eigh.
# ---------------------------------------------------------------------------

def test_min_generalized_eig_matches_scipy_oracle():
    """registry._min_generalized_eig (numpy-only: Cholesky + eigvalsh) equals
    scipy's generalized-eigenvalue smallest root on random symmetric-PD pairs
    (rtol 1e-9: both machine-accurate; catches a Cholesky-side/transpose bug)."""
    from dpend.registry import _min_generalized_eig
    from scipy.linalg import eigh

    rng = np.random.default_rng(7)
    for _ in range(6):
        Ma = rng.standard_normal((4, 4)); A = Ma @ Ma.T + 3.0 * np.eye(4)
        Mb = rng.standard_normal((4, 4)); B = Mb @ Mb.T + 3.0 * np.eye(4)
        got = _min_generalized_eig(A, B)
        want = float(eigh(A, B, eigvals_only=True)[0])
        assert got == pytest.approx(want, rel=1e-9)


def test_swingup_factory_lqr_catch_c_catch_equals_calibrated_base():
    """For an LQR catch, P_catch is the reference LQR P, so λ_min(P_catch,P_lqr)=1
    exactly and c_catch equals the calibrated base with no rescale."""
    plant = cart_pole_plant()
    ms = CONTROLLERS["swingup"](plant, {"catch": "lqr"})
    assert ms.c_catch == pytest.approx(BASIN_V_THRESHOLD, rel=1e-9)
    assert ms.c_release == pytest.approx(2 * BASIN_V_THRESHOLD, rel=1e-9)


def test_swingup_factory_mpc_catch_c_catch_guarantees_basin_containment():
    """c_catch = base·λ_min(P_catch, P_lqr) provably contains the catch-engage
    sublevel set inside the LQR-validated basin: max of eᵀP_lqr e over
    {eᵀP_catch e = c_catch} is c_catch/λ_min = base exactly, so the worst-case
    ratio is ≤1 (a ‖·‖-ratio scaling gives ~1.0005 and fails this test)."""
    plant = cart_pole_plant()
    P_lqr = CONTROLLERS["lqr"](plant, {}).P
    ms = CONTROLLERS["swingup"](plant, {"catch": "mpc"})
    P_catch, c_catch = ms.catch.P, ms.c_catch

    from scipy.linalg import eigh  # test-only generalized-eig oracle
    lam_min = float(eigh(P_catch, P_lqr, eigvals_only=True)[0])
    assert c_catch == pytest.approx(BASIN_V_THRESHOLD * lam_min, rel=1e-9)

    # worst case attained along the min generalized eigenvector; ratio ≤ 1 ⇔ contained
    worst_ratio = c_catch / (lam_min * BASIN_V_THRESHOLD)
    print(f"\n[λ_min containment] λ_min={lam_min:.4f}  c_catch={c_catch:.4f}  "
          f"worst-case Vlqr/base on the catch boundary = {worst_ratio:.6f} (≤1 ⇒ contained)")
    assert worst_ratio <= 1.0 + 1e-12
