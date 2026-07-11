"""Swing-up e2e: hang -> swing up -> catch -> hold for both catch controllers,
plus disturbance recovery (knock out of the basin, re-swing, re-catch). All
controllers built through the real CONTROLLERS["swingup"] factory."""
from __future__ import annotations

import numpy as np

from dpend.model.plant import cart_pole_plant
from dpend.registry import CONTROLLERS, ESTIMATORS, SENSORS
from dpend.sim.simulator import simulate
from dpend.util.angles import angle_diff


def _wrapped_norms(x_true: np.ndarray, z_up: np.ndarray) -> np.ndarray:
    """||z - z_up|| per row with the angle (index 1) wrapped via angle_diff —
    raw subtraction would report a huge norm for a state wound past ±pi
    (theta ~ 2pi is upright)."""
    e = x_true - z_up
    e[:, 1] = np.array([angle_diff(float(th), float(z_up[1])) for th in x_true[:, 1]])
    return np.linalg.norm(e, axis=1)


def _mode_timeline(ctrl):
    """Wrap ctrl.update to record (t, mode) after every tick — ModeSwitch's
    `mode` is controller-internal state, not a Telemetry column."""
    timeline: list = []
    orig_update = ctrl.update

    def spy(t, x_hat):
        u = orig_update(t, x_hat)
        timeline.append((t, ctrl.mode))
        return u

    ctrl.update = spy
    return timeline


def _run_swingup(plant, catch_name, x0, duration_s, disturbance=None):
    ctrl = CONTROLLERS["swingup"](plant, {"catch": catch_name})
    sensor = SENSORS["perfect"](plant, {})
    estimator = ESTIMATORS["identity"](plant, {})
    timeline = _mode_timeline(ctrl)

    tel = simulate(plant=plant, x0=x0, duration_s=duration_s, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
                   sensor=sensor, estimator=estimator, controller=ctrl, seed=0,
                   disturbance=disturbance)
    return ctrl, tel, timeline


def _catch_time(timeline) -> float | None:
    """First t at which mode == CATCHING (None if it never catches)."""
    for t, mode in timeline:
        if mode == "CATCHING":
            return t
    return None


# --- headline: swing up from hanging, catch, and hold, both catch laws ---

def _headline_swingup_and_hold(catch_name: str):
    """Shared body for the LQR/MPC headline tests: x0=hanging, T=12 s. Final
    ||z-z_up|| < 0.05 and staying < 0.1 for the whole last 2 s — swung up,
    caught, and held, not a fleeting close pass."""
    plant = cart_pole_plant()
    x0 = plant.hanging
    duration_s = 12.0

    ctrl, tel, timeline = _run_swingup(plant, catch_name, x0, duration_s)

    t_s = tel.t_ns / 1e9
    upright = plant.upright
    norms = _wrapped_norms(tel.x_true, upright)

    catch_t = _catch_time(timeline)
    swingup_duration = catch_t  # swing-up starts at t=0, so these coincide
    final_norm = float(norms[-1])
    last_2s_mask = t_s >= (duration_s - 2.0)
    last_2s_max = float(np.max(norms[last_2s_mask]))

    print(f"\n[swingup e2e, catch={catch_name}] catch time (first mode->CATCHING): {catch_t} s")
    print(f"[swingup e2e, catch={catch_name}] swing-up duration: {swingup_duration} s")
    print(f"[swingup e2e, catch={catch_name}] final ||z-z_up|| (wrapped) = {final_norm:.6f}")
    print(f"[swingup e2e, catch={catch_name}] max ||z-z_up|| over last 2s = {last_2s_max:.6f}")
    print(f"[swingup e2e, catch={catch_name}] final z = {tel.x_true[-1]}")

    assert catch_t is not None, "mode never entered CATCHING within the run"
    assert final_norm < 0.05
    assert last_2s_max < 0.1


def test_swingup_e2e_catch_lqr_swings_up_and_holds():
    _headline_swingup_and_hold("lqr")


def test_swingup_e2e_catch_mpc_swings_up_and_holds():
    """Same assertions as the LQR headline — ModeSwitch is catch-agnostic;
    the rescaled c_catch (registry.py) is what makes the switch trigger with
    MPC's ~200x-larger DARE P."""
    _headline_swingup_and_hold("mpc")


# --- disturbance recovery: knock a caught/balanced run out, re-catch ---

def test_disturbance_recovery_re_swings_and_recatches():
    """One continuous 16 s run: hang -> catch (~6.5 s), then a 200 N x 20 ms
    cart kick at t=9 s (from an already-held state) pushes V past c_release,
    the mode drops to SWINGING, and the pump re-catches within ~1 s.
    Kick sizing (measured sweep): kicks up to ~100 N/20 ms are absorbed
    without leaving the basin; ~150-450 N all give the same clean round trip,
    so 200 N sits well inside the window. Asserts the exact mode sequence
    SWINGING -> CATCHING -> SWINGING -> CATCHING."""
    plant = cart_pole_plant()
    x0 = plant.hanging
    duration_s = 16.0
    kick_start_s, kick_duration_s, kick_force = 9.0, 0.02, 200.0

    def disturbance(t, x):
        if kick_start_s <= t < kick_start_s + kick_duration_s:
            return np.array([kick_force, 0.0])  # [Fx on cart, torque on theta]
        return np.zeros(2)

    ctrl, tel, timeline = _run_swingup(plant, "lqr", x0, duration_s, disturbance=disturbance)

    t_s = tel.t_ns / 1e9
    upright = plant.upright
    norms = _wrapped_norms(tel.x_true, upright)

    transitions = [(t, m) for i, (t, m) in enumerate(timeline) if i == 0 or m != timeline[i - 1][1]]
    modes_seq = [m for _, m in transitions]

    final_norm = float(norms[-1])
    last_2s_mask = t_s >= (duration_s - 2.0)
    last_2s_max = float(np.max(norms[last_2s_mask]))

    print(f"\n[disturbance recovery] mode timeline (t, new_mode): {transitions}")
    print(f"[disturbance recovery] final ||z-z_up|| (wrapped) = {final_norm:.6f}")
    print(f"[disturbance recovery] max ||z-z_up|| over last 2s = {last_2s_max:.6f}")
    print(f"[disturbance recovery] final z = {tel.x_true[-1]}")

    # exact 4-phase round trip, no extra chatter
    assert modes_seq == ["SWINGING", "CATCHING", "SWINGING", "CATCHING"], (
        f"expected a single knock-out/recatch round trip, got mode sequence {modes_seq}"
    )
    assert final_norm < 0.05
    assert last_2s_max < 0.1
