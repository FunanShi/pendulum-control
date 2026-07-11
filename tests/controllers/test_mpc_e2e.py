"""MPC end-to-end through simulate() on the cart plant: regulation, the
MPC-vs-LQR rail showdown, tracking with a genuinely-active rail, a
fallback-tick audit, and the per-tick solve-time budget."""
from __future__ import annotations

import time

import numpy as np

from dpend.model.plant import cart_plant
from dpend.reference import ReferenceSource
from dpend.registry import CONTROLLERS, ESTIMATORS, SENSORS
from dpend.sim.simulator import simulate


def _closed_loop(plant, controller_name="mpc", params=None):
    """Build (sensor, estimator, controller) as batch.py does — perfect sensing
    + identity estimation, so x_hat == x_true and the loop is exercised directly."""
    params = params or {}
    sensor = SENSORS["perfect"](plant, {})
    estimator = ESTIMATORS["identity"](plant, {})
    controller = CONTROLLERS[controller_name](plant, params)
    return sensor, estimator, controller


def _settle_time(t_s: np.ndarray, norms: np.ndarray, thresh: float):
    """First t at which `norms` drops under `thresh` and stays there for the
    rest of the run (a real settle, not a transient dip); None if never."""
    below = np.where(norms < thresh)[0]
    for i in below:
        if np.all(norms[i:] < thresh):
            return float(t_s[i])
    return None


def _fallback_count(ctrl) -> tuple[list, callable]:
    """Wrap `ctrl.update` to record ctrl.status after every tick without
    touching simulate()'s call signature. Returns (statuses, original update)."""
    statuses: list = []
    orig_update = ctrl.update

    def spy(t, x_hat):
        u = orig_update(t, x_hat)
        statuses.append(ctrl.status)
        return u

    ctrl.update = spy
    return statuses, orig_update


# --- (a) cart regulation e2e: scenarios/cart_mpc.py's own IC ---

def test_cart_mpc_regulates_tip_to_upright():
    """0.15 rad tip regulates to upright (8 s, factory defaults); cart stays
    inside the controller's promised x_max at every tick, not just at the end."""
    plant = cart_plant()
    sensor, estimator, controller = _closed_loop(plant)
    x0 = np.array([0.0, 0.15, 0.0, 0.0, 0.0, 0.0])

    tel = simulate(plant=plant, x0=x0, duration_s=8.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
                   sensor=sensor, estimator=estimator, controller=controller, seed=0)

    t_s = tel.t_ns / 1e9
    norms = np.linalg.norm(tel.x_true, axis=1)
    settle = _settle_time(t_s, norms, 0.01)
    print(f"\n[cart mpc regulation] settle (||z||<0.01): {settle:.3f} s; "
          f"final ||z|| = {norms[-1]:.3e}")
    assert settle is not None
    assert norms[-1] < 1e-3

    x_max = controller._x_max
    max_abs_x = float(np.max(np.abs(tel.x_true[:, 0])))
    print(f"[cart mpc regulation] max|x_cart| over the run = {max_abs_x:.4f} m "
          f"(x_max={x_max} m)")
    assert max_abs_x <= x_max


# --- (b) MPC-vs-LQR rail showdown ---

def test_rail_showdown_mpc_vs_lqr_headline():
    """Identical x0 (cart at 1.24 m drifting 0.30 m/s toward the wall, tip
    0.05 rad): MPC honors its promised x_max=1.4 m; rail-blind LQR overshoots
    the same bound by ~1 cm. Both stabilize the pendulum, neither saturates
    u_max (peak|u| ~11 N of 150) — a horizon-awareness separation, not torque.
    x0 is calibrated: the separating band is narrow (from 1.20 m both stay
    inside; much harder and both fail). A few MPC ticks hit OSQP's iteration
    cap at the tightest point and use the fallback plan — zero-fallback is
    asserted only on the nominal runs (test (d)), not at this feasibility edge."""
    plant = cart_plant()
    x0 = np.array([1.24, 0.05, 0.0, 0.30, 0.0, 0.0])
    duration_s = 8.0

    sensor_m, estimator_m, mpc = _closed_loop(plant, "mpc")
    sensor_l, estimator_l, lqr = _closed_loop(plant, "lqr", {"u_max": 150.0})

    mpc_statuses, _ = _fallback_count(mpc)

    tel_mpc = simulate(plant=plant, x0=x0.copy(), duration_s=duration_s, sim_dt_s=1e-3,
                       ctrl_dt_s=5e-3, sensor=sensor_m, estimator=estimator_m,
                       controller=mpc, seed=0)
    tel_lqr = simulate(plant=plant, x0=x0.copy(), duration_s=duration_s, sim_dt_s=1e-3,
                       ctrl_dt_s=5e-3, sensor=sensor_l, estimator=estimator_l,
                       controller=lqr, seed=0)

    x_max = mpc._x_max
    peak_x_mpc = float(np.max(np.abs(tel_mpc.x_true[:, 0])))
    peak_x_lqr = float(np.max(np.abs(tel_lqr.x_true[:, 0])))
    peak_u_mpc = float(np.max(np.abs(tel_mpc.u)))
    peak_u_lqr = float(np.max(np.abs(tel_lqr.u)))
    final_norm_mpc = float(np.linalg.norm(tel_mpc.x_true[-1]))
    final_norm_lqr = float(np.linalg.norm(tel_lqr.x_true[-1]))
    n_fallback = mpc_statuses.count("fallback")

    print(f"\n[rail showdown] x_max={x_max} m, rail={plant.rail} m, x0={list(x0)}")
    print(f"[rail showdown] MPC: peak|x_cart|={peak_x_mpc:.5f} m  peak|u|={peak_u_mpc:.3f} N  "
          f"final||x||={final_norm_mpc:.3e}  fallback_ticks={n_fallback}/{len(mpc_statuses)}")
    print(f"[rail showdown] LQR: peak|x_cart|={peak_x_lqr:.5f} m  peak|u|={peak_u_lqr:.3f} N  "
          f"final||x||={final_norm_lqr:.3e}")
    print(f"[rail showdown] LQR exceeds x_max by {peak_x_lqr - x_max:.5f} m "
          f"({'PAST the promised bound' if peak_x_lqr > x_max else 'still inside it'})")

    # MPC's promise: its own QP constraint, to the +1e-3 closed-loop tolerance
    assert peak_x_mpc <= x_max + 1e-3
    # the differentiator: rail-blind LQR overshoots the same bound
    assert peak_x_lqr > x_max
    # both must still stabilize the pendulum — a rail story, not a stability one
    assert final_norm_mpc < 1e-2
    assert final_norm_lqr < 1e-2
    assert np.all(np.isfinite(tel_lqr.x_true))


# --- (c) tracking with a genuinely-active rail near the target ---

def test_cart_mpc_tracks_with_rail_genuinely_active_near_target():
    """Tracking to +0.8 m with x_max tightened to 0.812 m so the rail actually
    shapes the trajectory — the closed-loop counterpart of test_mpc.py's
    absolute-rail bookkeeping test.
    x_max=0.812 is calibrated: the unconstrained transient peaks at 0.812125 m
    (looser bounds never bind); 0.812 rides the wall within solver tolerance
    with 3/1600 fallback ticks, while 0.8115-0.8116 collapses into ~800
    fallback ticks and loses the pendulum. Deterministic across runs."""
    plant = cart_plant()
    x_max = 0.812
    sensor, estimator, controller = _closed_loop(
        plant, "mpc", {"reference": ReferenceSource(0.8), "x_max": x_max})
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
    max_x = float(np.max(x))
    slack = x_max - max_x
    print(f"\n[cart mpc tracking, tight rail] settle (within 2% of 0.8 m): {settle} s; "
          f"final x = {x[-1]:.4f} m")
    print(f"[cart mpc tracking, tight rail] max_x={max_x:.6f} m, x_max={x_max} m, "
          f"slack={slack:.3e} m")
    assert settle is not None

    # self-certification: if the wall never bound, correct absolute-coordinate
    # bookkeeping would be indistinguishable from a broken one
    assert abs(slack) < 1e-3, "rail never became genuinely active near the target"

    # the hard-constraint promise, even mid-tracking
    assert max_x <= x_max + 1e-3

    th1, th2 = tel.x_true[-1, 1], tel.x_true[-1, 2]
    print(f"[cart mpc tracking, tight rail] final theta1={th1:.3e} rad, theta2={th2:.3e} rad")
    assert abs(th1) < 1e-3
    assert abs(th2) < 1e-3


# --- (d) receding-horizon status audit: nominal runs never fall back ---

def test_mpc_zero_fallback_ticks_on_nominal_runs():
    """Two calm closed loops (regulation tip 0.15; tracking +0.8 at default
    x_max — nowhere near the rail) produce zero "fallback" ticks; the
    deliberately adversarial runs (b) and (c) are not held to this."""
    plant = cart_plant()

    # regulation, tip 0.15
    sensor, estimator, ctrl_reg = _closed_loop(plant, "mpc")
    statuses_reg, _ = _fallback_count(ctrl_reg)
    x0 = np.array([0.0, 0.15, 0.0, 0.0, 0.0, 0.0])
    simulate(plant=plant, x0=x0, duration_s=8.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
            sensor=sensor, estimator=estimator, controller=ctrl_reg, seed=0)
    n_fb_reg = statuses_reg.count("fallback")
    print(f"\n[fallback audit] regulation (tip 0.15): {n_fb_reg}/{len(statuses_reg)} fallback ticks")
    assert n_fb_reg == 0

    # tracking, +0.8, default x_max
    plant = cart_plant()
    sensor, estimator, ctrl_trk = _closed_loop(plant, "mpc", {"reference": ReferenceSource(0.8)})
    statuses_trk, _ = _fallback_count(ctrl_trk)
    x0 = np.zeros(6)
    simulate(plant=plant, x0=x0, duration_s=8.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
            sensor=sensor, estimator=estimator, controller=ctrl_trk, seed=0)
    n_fb_trk = statuses_trk.count("fallback")
    print(f"[fallback audit] tracking (+0.8, default x_max): {n_fb_trk}/{len(statuses_trk)} "
          f"fallback ticks")
    assert n_fb_trk == 0


# --- (e) per-tick solve-time budget ---

def test_mpc_mean_solve_time_within_real_time_budget():
    """Wall-clock inside controller.update() over the 1600-tick regulation run:
    mean < 2 ms leaves >60% of the 5 ms tick for sensor/estimator/telemetry/
    physics work; max < 5 ms gates the worst case, since one overrun tick is a
    missed actuation deadline regardless of the mean (measured mean ~0.08 ms,
    max ~0.26-0.29 ms in this container)."""
    plant = cart_plant()
    sensor, estimator, controller = _closed_loop(plant)
    x0 = np.array([0.0, 0.15, 0.0, 0.0, 0.0, 0.0])

    durations_s: list = []
    orig_update = controller.update

    def timed(t, x_hat):
        t0 = time.perf_counter()
        u = orig_update(t, x_hat)
        durations_s.append(time.perf_counter() - t0)
        return u

    controller.update = timed

    simulate(plant=plant, x0=x0, duration_s=8.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
            sensor=sensor, estimator=estimator, controller=controller, seed=0)

    durations_ms = np.asarray(durations_s) * 1e3
    mean_ms = float(np.mean(durations_ms))
    max_ms = float(np.max(durations_ms))
    print(f"\n[solve time] n_ticks={len(durations_ms)}  mean={mean_ms:.4f} ms  "
          f"max={max_ms:.4f} ms  (ctrl_dt=5.000 ms budget)")
    assert mean_ms < 2.0
    # worst-case gate: one tick past ctrl_dt=5 ms is a missed deadline
    assert max_ms < 5.0
