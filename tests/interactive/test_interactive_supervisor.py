"""RoA supervisor tests: hard-drag trip (LQR + MPC), notice clear/rearm on
R + reselect, V_limit calibration, controllers with no P are unsupervised,
a swing-up hybrid does not arm it, and the IS_HYBRID gate itself."""
from __future__ import annotations

import numpy as np
import pygame
import pytest

from dpend.interactive.app import ROA_CALIBRATION_TIP_RAD
from dpend.model.plant import cart_pole_plant
from tests._interactive_helpers import _make_app, _hard_drag_until_fired


def test_lqr_hard_drag_trips_roa_supervisor_back_to_manual():
    """A vigorous sustained drag trips the RoA supervisor — mode auto-returns
    to MANUAL, notice non-empty — and the original incident's failure modes
    must not recur: across the drag + a 60-tick cool-down the cart stays
    within rail + 0.5 m and |u| never exceeds the u_max=150 N default."""
    app = _make_app(controller_name="lqr")
    try:
        assert app.mode == "CONTROLLER"
        P = app._active_controller.P  # captured before the drag: the disengage
                                      # swaps the active controller to "zero", so
                                      # V at the fire tick must be recomputed from
                                      # the current true state + this P, not a
                                      # post-fire hud()["v"] read (None by then)
        v_limit = app.hud()["v_limit"]
        assert v_limit is not None  # LQR exposes P: supervisor is armed

        fire_tick, records = _hard_drag_until_fired(app)
        # app.loop.x (not records[-1]["x_true"], the pre-tick state) is the
        # exact post-integration state the supervisor's V-check evaluated;
        # set_controller("zero") never touches loop.x.
        v_at_fire = float(app.loop.x @ P @ app.loop.x)
        print(f"\n[RoA supervisor] V_lim={v_limit:.4e}  fired_at_tick={fire_tick}  "
              f"V_at_fire={v_at_fire:.4e}  notice={app.notice!r}")

        assert fire_tick is not None, "supervisor never fired within the drag window"
        assert app.mode == "MANUAL"
        assert app.notice != ""
        assert v_at_fire > v_limit

        # release + cool-down: the safety net must hold afterward too
        end_px = app.w2s.to_px((float(app._mouse_world_x), 0.0))
        app.step_once(synthetic_events=[pygame.event.Event(
            pygame.MOUSEBUTTONUP, pos=(round(end_px[0]), round(end_px[1])), button=1)])
        for _ in range(60):
            records += app.step_once(synthetic_events=[])

        xs = np.array([r["x_true"] for r in records])
        us = np.array([r["u"] for r in records]).reshape(len(records), -1)
        print(f"[RoA supervisor] max|x_cart|={np.max(np.abs(xs[:, 0])):.4f} m  "
              f"max|u|={np.max(np.abs(us)):.4f} N  (rail={app.plant.rail} m)")
        assert np.max(np.abs(xs[:, 0])) < app.plant.rail + 0.5
        assert np.max(np.abs(us)) <= 150.0
    finally:
        pygame.quit()


def test_supervisor_notice_clears_on_r_and_on_reselecting_controller():
    """Both R and reselecting a controller clear a stale supervisor notice
    (notice seeded directly; deterministic, no drag dynamics). The proof the
    supervisor re-arms lives in test_supervisor_rearms_after_reset_and_reengage."""
    app = _make_app(controller_name="lqr")
    try:
        app.set_controller("zero")
        app.notice = "LQR disengaged: left the basin (V > V_lim) — R to reset"

        app.step_once(synthetic_events=[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)])
        assert app.notice == ""
        assert app.mode == "MANUAL"              # R resets state; it does not re-engage
        np.testing.assert_allclose(app.loop.x, app.x0, atol=1e-9)

        app.notice = "LQR disengaged: left the basin (V > V_lim) — R to reset"
        app.set_controller("lqr")
        assert app.mode == "CONTROLLER"            # safe to reselect: state is at x0 (V=0)
        assert app.notice == ""                    # reselecting clears it too, not just R
    finally:
        pygame.quit()


def test_supervisor_rearms_after_reset_and_reengage():
    """Not a one-shot latch: fire, R + reselect (the recovery flow the notice
    prescribes), repeat the same hard drag — it fires again. Reselecting
    without resetting first is deliberately not exercised: V would still
    exceed V_lim and it would re-disengage that same tick, a separate behavior."""
    app = _make_app(controller_name="lqr")
    try:
        fire_tick_1, _ = _hard_drag_until_fired(app)
        assert fire_tick_1 is not None
        assert app.mode == "MANUAL"
        assert app.notice != ""

        app.step_once(synthetic_events=[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)])
        assert app.notice == ""
        assert app.dragging is False  # R also clears a stale grab

        app.set_controller("lqr")
        assert app.mode == "CONTROLLER"  # safe re-engage: state is back at x0 (V=0 << V_lim)
        assert app.notice == ""

        fire_tick_2, _ = _hard_drag_until_fired(app)
        print(f"\n[RoA supervisor] re-arm proof: fired_at_tick_1={fire_tick_1}  "
              f"fired_at_tick_2={fire_tick_2}")
        assert fire_tick_2 is not None, "supervisor did not re-arm after reset + re-engage"
        assert app.mode == "MANUAL"
        assert app.notice != ""
    finally:
        pygame.quit()


def test_supervisor_v_limit_matches_calibration_tip_and_controller_p():
    """V_lim equals z_tip^T P z_tip for z_tip = zeros except theta1 =
    ROA_CALIBRATION_TIP_RAD — the formula App.__init__ documents; and at rest
    at the origin V(x0) is exactly 0."""
    app = _make_app(controller_name="lqr")
    try:
        plant = app.plant
        P = app._active_controller.P
        z_tip = np.zeros(plant.n)
        z_tip[1] = ROA_CALIBRATION_TIP_RAD
        expected_v_limit = float(z_tip @ P @ z_tip)

        assert app.hud()["v_limit"] == pytest.approx(expected_v_limit)
        assert app.hud()["v"] == pytest.approx(0.0, abs=1e-12)  # at rest at x0: V(0)=0 exactly
    finally:
        pygame.quit()


def test_zero_controller_has_no_supervisor_v_and_v_limit_are_none():
    """Controllers with no `P` (e.g. "zero") must leave the supervisor
    inactive: hud()["v"]/["v_limit"] are None, never a stale/bogus number."""
    app = _make_app()  # default controller_name="zero"
    try:
        hud = app.hud()
        assert hud["v"] is None
        assert hud["v_limit"] is None
        assert hud["notice"] == ""
    finally:
        pygame.quit()


def test_swingup_controller_does_not_arm_the_roa_supervisor():
    """Regression: ModeSwitch also exposes .P, so a naive hasattr-P gate would
    arm the small-tip linear supervisor for swing-up too — and with
    V(hanging)=226 vs V_lim~2 it would trip on the first CONTROLLER tick and
    revert to MANUAL before the pump ever acts. A hybrid supervises itself;
    the linear supervisor stays off. LQR/MPC remain supervised (the paired
    'must fire' tests above)."""
    app = _make_app(plant=cart_pole_plant(), controller_name="swingup", start="hanging")
    try:
        hud = app.hud()
        assert hud["v"] is None
        assert hud["v_limit"] is None

        assert app.mode == "CONTROLLER"  # must NOT bounce straight back to MANUAL
    finally:
        pygame.quit()


def test_mpc_hard_drag_trips_roa_supervisor_back_to_manual():
    """The LQR hard-drag reproduction rerun with MPC: the supervisor fires
    from MPC's own P under the same post-incident safety bounds (rail+0.5 m,
    |u|<=150 N, drag + 60-tick cool-down) — an apples-to-apples parity check.
    Also pins the notice naming the actual active controller
    ("MPC disengaged...", not a hardcoded "LQR")."""
    app = _make_app(controller_name="mpc")
    try:
        assert app.mode == "CONTROLLER"
        P = app._active_controller.P  # captured before the drag — see the LQR
                                      # test's twin comment
        v_limit = app.hud()["v_limit"]
        assert v_limit is not None  # MPC exposes P: supervisor is armed

        fire_tick, records = _hard_drag_until_fired(app)
        # app.loop.x, not records[-1]["x_true"] (pre-tick) — see the LQR test
        v_at_fire = float(app.loop.x @ P @ app.loop.x)
        print(f"\n[RoA supervisor, mpc] V_lim={v_limit:.4e}  fired_at_tick={fire_tick}  "
              f"V_at_fire={v_at_fire:.4e}  notice={app.notice!r}")

        assert fire_tick is not None, "supervisor never fired within the drag window"
        assert app.mode == "MANUAL"
        assert app.notice != ""
        assert "MPC" in app.notice  # notice names the actual active controller
        assert v_at_fire > v_limit

        end_px = app.w2s.to_px((float(app._mouse_world_x), 0.0))
        app.step_once(synthetic_events=[pygame.event.Event(
            pygame.MOUSEBUTTONUP, pos=(round(end_px[0]), round(end_px[1])), button=1)])
        for _ in range(60):
            records += app.step_once(synthetic_events=[])

        xs = np.array([r["x_true"] for r in records])
        us = np.array([r["u"] for r in records]).reshape(len(records), -1)
        print(f"[RoA supervisor, mpc] max|x_cart|={np.max(np.abs(xs[:, 0])):.4f} m  "
              f"max|u|={np.max(np.abs(us)):.4f} N  (rail={app.plant.rail} m)")
        assert np.max(np.abs(xs[:, 0])) < app.plant.rail + 0.5
        assert np.max(np.abs(us)) <= 150.0
    finally:
        pygame.quit()


def test_roa_supervisor_gate_uses_IS_HYBRID_flag_not_mode_attribute():
    """ModeSwitch declares IS_HYBRID=True and is not supervised; a non-hybrid
    that merely exposes a `.mode` attribute is still supervised — guards against
    a `.mode` duck-typing gate, under which it would silently lose its safety net."""
    from dpend.controllers.mode_switch import ModeSwitch
    from dpend.registry import CONTROLLERS

    assert ModeSwitch.IS_HYBRID is True

    class _FakeModeButNotHybrid:
        def __init__(self, plant, params):
            self.mode = "WHATEVER"        # incidental attr, not a hybrid marker
            self.P = np.eye(plant.n)       # exposes a P → must be supervised
        def reset(self, t0, x0):
            pass
        def update(self, t, x_hat):
            return np.zeros(1)

    CONTROLLERS["_fake_mode"] = lambda plant, params: _FakeModeButNotHybrid(plant, params)
    try:
        app = _make_app(plant=cart_pole_plant(), controller_name="_fake_mode")
        try:
            # v_limit is populated iff the supervisor armed — it must, despite `.mode`.
            assert app.hud()["v_limit"] is not None
        finally:
            pygame.quit()
    finally:
        del CONTROLLERS["_fake_mode"]
