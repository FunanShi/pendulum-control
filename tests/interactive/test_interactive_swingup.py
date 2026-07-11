"""Cart-pole swing-up interactive tests: hanging boot/reset, energy-pump
reachability, the swing_mode HUD field, and LQR-selection-triggers-swing-up
on the cart-pole plant."""
from __future__ import annotations

import numpy as np
import pygame
import pytest

from dpend.model.plant import cart_pole_plant
from tests._interactive_helpers import _make_app


def test_start_defaults_to_upright_for_backward_compatibility():
    """No `start` kwarg (every pre-existing call site) must still boot at
    plant.upright — the origin — exactly as before this feature existed."""
    plant = cart_pole_plant()
    app = _make_app(plant=plant)
    try:
        np.testing.assert_allclose(app.x0, plant.upright, atol=1e-12)
        np.testing.assert_allclose(app.loop.x, plant.upright, atol=1e-12)
    finally:
        pygame.quit()


def test_start_hanging_boots_cart_pole_plant_at_hanging_state():
    """start="hanging" boots x0 (and loop.x) at plant.hanging = [0, pi, 0, 0]
    — the swing-up demo's actual starting point, not the upright origin."""
    plant = cart_pole_plant()
    app = _make_app(plant=plant, controller_name="swingup", start="hanging")
    try:
        np.testing.assert_allclose(app.x0, plant.hanging, atol=1e-12)
        np.testing.assert_allclose(app.loop.x, plant.hanging, atol=1e-12)
        assert float(app.loop.x[1]) == pytest.approx(np.pi)
    finally:
        pygame.quit()


def test_start_hanging_swingup_pumps_pole_angle_toward_upright():
    """Headless swing-up through the live App path: --start hanging
    --controller swingup, 1700 ticks (8.5 s, past the batch e2e's ~7.1 s
    catch time). Asserts on cos(theta) — branch-cut-free on S^1: a successful
    pump can wind theta past ±pi, and cos rising from cos(pi)=-1 means
    "toward upright" regardless of winding."""
    plant = cart_pole_plant()
    app = _make_app(plant=plant, controller_name="swingup", start="hanging")
    try:
        theta0 = float(app.loop.x[1])
        assert theta0 == pytest.approx(np.pi)

        assert app.mode == "CONTROLLER"

        n_ticks = 1700  # 8.5 s at ctrl_dt=5ms
        records = []
        for _ in range(n_ticks):
            records += app.step_once(synthetic_events=[])

        theta_final = float(records[-1]["x_true"][1])
        print(f"\n[UI swingup] theta: {theta0:.4f} -> {theta_final:.4f} rad "
              f"(cos: {np.cos(theta0):.4f} -> {np.cos(theta_final):.4f}), "
              f"swing_mode={getattr(app._active_controller, 'mode', None)}")
        assert np.cos(theta_final) > np.cos(theta0) + 1.0  # substantial pump toward upright
    finally:
        pygame.quit()


def test_r_key_resets_cart_pole_swingup_state_to_hanging_not_upright():
    """R resets to the chosen start (hanging), not a hardcoded upright origin.
    A held arrow key moves the cart off hanging quickly — deliberately
    decoupled from the pump's ~5 s dead time from exact hanging, since this
    test is about R/x0 wiring, not swing-up timing."""
    plant = cart_pole_plant()
    app = _make_app(plant=plant, controller_name="swingup", start="hanging")
    try:
        push = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
        for _ in range(20):
            app.step_once(synthetic_events=[push])
        assert not np.allclose(app.loop.x, plant.hanging, atol=1e-6)  # moved off hanging

        release = pygame.event.Event(pygame.KEYUP, key=pygame.K_RIGHT)
        app.step_once(synthetic_events=[release])  # let go before reset (isolate R's effect)

        app.step_once(synthetic_events=[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)])
        np.testing.assert_allclose(app.loop.x, plant.hanging, atol=1e-9)
    finally:
        pygame.quit()


def test_hud_swing_mode_is_none_for_non_swingup_controllers():
    """Controllers with no `.mode` attribute report swing_mode=None — never a
    stale string — so draw_scene omits the swing: line for non-hybrids."""
    for controller_name in ("zero", "lqr"):
        app = _make_app(controller_name=controller_name)
        try:
            assert app.hud()["swing_mode"] is None
        finally:
            pygame.quit()


def test_hud_swing_mode_transitions_swinging_to_catching_over_a_hanging_run():
    """The HUD reflects live ModeSwitch state: over a 1700-tick hanging run,
    hud()["swing_mode"] shows both "SWINGING" and "CATCHING" — a live read of
    ctrl.mode, not a snapshot."""
    plant = cart_pole_plant()
    app = _make_app(plant=plant, controller_name="swingup", start="hanging")
    try:
        assert app.hud()["swing_mode"] == "SWINGING"  # starts SWINGING (ModeSwitch's own contract)

        assert app.mode == "CONTROLLER"

        seen_modes = set()
        for _ in range(1700):
            app.step_once(synthetic_events=[])
            seen_modes.add(app.hud()["swing_mode"])

        print(f"\n[UI swingup HUD] swing_mode values observed over the run: {seen_modes}")
        assert seen_modes == {"SWINGING", "CATCHING"}
    finally:
        pygame.quit()


def test_ui_lqr_selection_swings_up_from_hanging_on_cartpole():
    from dpend.controllers.mode_switch import ModeSwitch
    app = _make_app(plant=cart_pole_plant(), controller_name="lqr", start="hanging")
    try:
        assert isinstance(app._active_controller, ModeSwitch)   # lqr wrapped as swing-up + lqr catch
        assert app.mode == "CONTROLLER"                          # selected → running, no M needed
        theta0 = float(app.loop.x[1]); assert theta0 == pytest.approx(np.pi)   # hanging
        records = []
        for _ in range(1700):                                    # ~8.5 s, past the measured catch
            records += app.step_once(synthetic_events=[])
        theta_f = float(records[-1]["x_true"][1])
        assert np.cos(theta_f) > np.cos(theta0) + 1.0            # swung up substantially toward upright
    finally:
        pygame.quit()


def test_ui_lqr_selection_catches_immediately_when_already_upright():
    app = _make_app(plant=cart_pole_plant(), controller_name="lqr", start="upright")
    try:
        app.step_once(synthetic_events=[])                       # one tick; selected controller runs
        assert app._active_controller.mode == "CATCHING"         # already in the basin → immediate catch
    finally:
        pygame.quit()
