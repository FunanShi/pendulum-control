"""LQR end-to-end through simulate() on both plants: cart regulation +
tracking, and the fixed/acrobot plant's measured region-of-attraction
boundary — convergent and divergent edges both pinned so the boundary
cannot drift silently (physics story: docs/design-notes/lqr-riccati.md)."""
from __future__ import annotations

import numpy as np

from dpend.model.plant import cart_plant, fixed_pivot_plant
from dpend.reference import ReferenceSource
from dpend.registry import CONTROLLERS, ESTIMATORS, SENSORS
from dpend.sim.simulator import simulate


def _closed_loop(plant, params=None):
    """Build (sensor, estimator, controller) as batch.py does — perfect sensing
    + identity estimation, so x_hat == x_true and the loop is exercised directly."""
    params = params or {}
    sensor = SENSORS["perfect"](plant, {})
    estimator = ESTIMATORS["identity"](plant, {})
    controller = CONTROLLERS["lqr"](plant, params)
    return sensor, estimator, controller


def _settle_time(t_s: np.ndarray, norms: np.ndarray, thresh: float):
    """First t at which `norms` drops under `thresh` and stays there for the
    rest of the run (a real settle, not a transient dip); None if never."""
    below = np.where(norms < thresh)[0]
    for i in below:
        if np.all(norms[i:] < thresh):
            return float(t_s[i])
    return None


# --- (a) cart regulation e2e: scenarios/cart_lqr.py's own IC ---

def test_cart_lqr_regulates_tip_to_upright():
    """A (0.15, -0.10) rad tip regulates to upright (8 s, factory Q/R): u is
    nonzero on the first tick and <1e-3 N by the end; prints the settle time."""
    plant = cart_plant()
    sensor, estimator, controller = _closed_loop(plant)
    x0 = np.array([0.0, 0.15, -0.10, 0.0, 0.0, 0.0])

    tel = simulate(plant=plant, x0=x0, duration_s=8.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
                   sensor=sensor, estimator=estimator, controller=controller, seed=0)

    t_s = tel.t_ns / 1e9
    norms = np.linalg.norm(tel.x_true, axis=1)
    settle = _settle_time(t_s, norms, 0.01)
    print(f"\n[cart regulation] settle (||z||<0.01): {settle:.3f} s; "
          f"final ||z|| = {norms[-1]:.3e}")
    assert settle is not None
    assert norms[-1] < 1e-3

    u = tel.u[:, 0]
    print(f"[cart regulation] u[0] (t=0) = {u[0]:.4f} N")
    assert u[0] != 0.0  # controller engages immediately from the tip error

    late = t_s >= 7.5
    max_u_late = np.max(np.abs(u[late]))
    print(f"[cart regulation] max|u| for t>=7.5s = {max_u_late:.3e} N")
    assert max_u_late < 1e-3


# --- (b) fixed acrobot: convergent edge of the measured RoA ---

def test_fixed_acrobot_lqr_converges_from_measured_roa_tip():
    """x0=(0.02, -0.01, 0, 0): the measured-convergent edge of the acrobot's
    RoA at canonical Q/R (docs/design-notes/lqr-riccati.md).
    The trajectory transiently reaches ||x||~2.8 driven by angular rate, not
    angle (angles stay within ~0.14 rad) — a brisk catch, not a large swing."""
    plant = fixed_pivot_plant(actuation="acrobot")
    sensor, estimator, controller = _closed_loop(
        plant, {"Q": [10.0, 10.0, 1.0, 1.0], "R": [0.1]})
    x0 = np.array([0.02, -0.01, 0.0, 0.0])

    tel = simulate(plant=plant, x0=x0, duration_s=8.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
                   sensor=sensor, estimator=estimator, controller=controller, seed=0)

    t_s = tel.t_ns / 1e9
    norms = np.linalg.norm(tel.x_true, axis=1)
    settle = _settle_time(t_s, norms, 0.01)
    print(f"\n[fixed acrobot, x0 tip=0.02 rad] settle (||x||<0.01): {settle:.3f} s; "
          f"final ||x|| = {norms[-1]:.3e}")
    assert settle is not None
    assert norms[-1] < 1e-6


# --- (c) fixed acrobot: divergent edge of the measured RoA (companion pin) ---

def test_fixed_acrobot_lqr_diverges_from_canonical_tip():
    """x0=(0.10, -0.05, 0, 0) diverges under the true nonlinear dynamics even
    though the linearized loop is Hurwitz there — the divergent edge of the
    same RoA boundary (real physics, a small RoA; see the design note).
    np.errstate suppresses the overflow RuntimeWarning: the state explodes
    past 1e3 by ~0.12 s and RK4 goes non-finite by ~0.75 s — expected for a
    scripted divergence, not a hidden bug."""
    plant = fixed_pivot_plant(actuation="acrobot")
    sensor, estimator, controller = _closed_loop(
        plant, {"Q": [10.0, 10.0, 1.0, 1.0], "R": [0.1]})
    x0 = np.array([0.10, -0.05, 0.0, 0.0])

    with np.errstate(over="ignore", invalid="ignore"):
        tel = simulate(plant=plant, x0=x0, duration_s=2.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
                       sensor=sensor, estimator=estimator, controller=controller, seed=0)

        t_s = tel.t_ns / 1e9
        norms = np.linalg.norm(tel.x_true, axis=1)
        over = np.where(norms > 1.0)[0]
        assert len(over) > 0, "expected the canonical 0.10 rad tip to diverge past ||x||=1"
        print(f"\n[fixed acrobot, x0 tip=0.10 rad canonical] first ||x||>1.0 at "
              f"t={t_s[over[0]]:.3f} s")
        assert t_s[over[0]] < 2.0

        # genuinely diverged, not a transient blip: the convergent 0.02 rad
        # case also transiently exceeds ||x||=1, so this second check is what
        # discriminates "diverges" from "brisk catch"
        final = tel.x_true[-1]
        final_finite = bool(np.all(np.isfinite(final)))
        print(f"[fixed acrobot, x0 tip=0.10 rad canonical] final state finite: "
              f"{final_finite}, value: {final}")
        assert (not final_finite) or np.linalg.norm(final) > 1.0


# --- (d) cart tracking e2e ---

def test_cart_lqr_tracks_shifted_setpoint_while_balancing():
    """From the upright origin with a +0.8 m rail target: x settles within 2%
    of 0.8 m and the final angles are <1e-3 rad (they pass through nonzero
    values en route; only the final angles are asserted small)."""
    plant = cart_plant()
    sensor, estimator, controller = _closed_loop(plant, {"reference": ReferenceSource(0.8)})
    x0 = np.zeros(6)

    tel = simulate(plant=plant, x0=x0, duration_s=8.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
                   sensor=sensor, estimator=estimator, controller=controller, seed=0)

    t_s = tel.t_ns / 1e9
    x = tel.x_true[:, 0]
    target = 0.8
    band = 0.02 * target
    in_band = np.abs(x - target) < band
    settle = None
    for i in range(len(x)):
        if np.all(in_band[i:]):
            settle = float(t_s[i])
            break
    print(f"\n[cart tracking +0.8] settle (within 2% of 0.8 m): {settle} s; "
          f"final x = {x[-1]:.4f} m")
    assert settle is not None

    th1, th2 = tel.x_true[-1, 1], tel.x_true[-1, 2]
    print(f"[cart tracking +0.8] final theta1={th1:.3e} rad, theta2={th2:.3e} rad")
    assert abs(th1) < 1e-3
    assert abs(th2) < 1e-3


# --- (e) railed-plants-only tracking rule: fixed plant ignores the reference ---

def test_fixed_plant_lqr_ignores_supplied_reference():
    """Tracking is wired only for railed plants (a rail is a translation-
    invariant DOF, so a setpoint shift is an exact equilibrium shift); the
    fixed plant's equilibrium is isolated, so it must ignore
    params["reference"] entirely: u identical with and without one."""
    plant = fixed_pivot_plant(actuation="acrobot")
    ctrl_no_ref = CONTROLLERS["lqr"](plant, {})
    ctrl_with_ref = CONTROLLERS["lqr"](plant, {"reference": ReferenceSource(5.0)})

    x_err = np.array([0.05, -0.02, 0.01, 0.0])
    u_no_ref = ctrl_no_ref.update(0.0, x_err)
    u_with_ref = ctrl_with_ref.update(0.0, x_err)
    print(f"\n[fixed plant reference-ignored] u_no_ref={u_no_ref}, u_with_ref={u_with_ref}")
    np.testing.assert_array_equal(u_no_ref, u_with_ref)
