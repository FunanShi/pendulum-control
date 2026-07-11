"""Controller-selection-governs-mode tests: a selected controller runs with no
M toggle, live set_controller swaps (plain-cart vs cart-pole hybrid-wrap), and
the in-sim controller strip."""
from __future__ import annotations

import numpy as np
import pygame

from dpend.model.plant import cart_plant, cart_pole_plant
from dpend.registry import CONTROLLERS
from tests._interactive_helpers import _make_app


def test_selected_controller_runs_and_selecting_none_stops_it():
    """The selected controller is the mode: a real controller runs (consulted
    every tick); selecting 'None' (zero) stops it. No M toggle."""
    instances = []
    def _spy_factory(plant, params):
        class _Spy:
            def __init__(self): self.calls = 0
            def reset(self, t0, x0): pass
            def update(self, t, x_hat):
                self.calls += 1
                return np.zeros(plant.m)
        spy = _Spy(); instances.append(spy); return spy
    CONTROLLERS["_spy_test"] = _spy_factory
    try:
        app = _make_app(controller_name="_spy_test")   # selected → running immediately
        try:
            spy = instances[0]
            assert app.mode == "CONTROLLER"               # a real controller is selected
            for _ in range(4):
                app.step_once(synthetic_events=[])
            assert spy.calls == 4                          # consulted every tick, no M needed
            app.set_controller("zero")                     # select 'None' → manual
            assert app.mode == "MANUAL"
            n_before = spy.calls
            for _ in range(3):
                app.step_once(synthetic_events=[])
            assert spy.calls == n_before                   # None selected: the spy is no longer consulted
        finally:
            pygame.quit()
    finally:
        del CONTROLLERS["_spy_test"]


def test_selected_controller_runs_with_active_drag_at_cadence_and_tau_ext_nonzero():
    """Drag stays live as a disturbance while a controller runs: with a spy
    selected and a live drag, every tick both consults the spy and carries
    the drag's nonzero tau_ext."""
    instances = []
    def _spy_factory(plant, params):
        class _Spy:
            def __init__(self): self.calls = 0
            def reset(self, t0, x0): pass
            def update(self, t, x_hat):
                self.calls += 1
                return np.zeros(plant.m)
        spy = _Spy(); instances.append(spy); return spy
    CONTROLLERS["_combo_spy"] = _spy_factory
    try:
        app = _make_app(controller_name="_combo_spy")   # selected → running
        try:
            spy = instances[0]
            assert app.mode == "CONTROLLER"
            cart_px = app.w2s.to_px((0.0, 0.0)); mouse_px = app.w2s.to_px((0.4, 0.0))
            app.step_once(synthetic_events=[pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, pos=(round(cart_px[0]), round(cart_px[1])), button=1)])
            assert app.dragging is True
            n_before = spy.calls
            motion = pygame.event.Event(pygame.MOUSEMOTION,
                                        pos=(round(mouse_px[0]), round(mouse_px[1])),
                                        rel=(0, 0), buttons=(1, 0, 0))
            drag_records = []
            for _ in range(4):
                drag_records += app.step_once(synthetic_events=[motion])
            assert spy.calls == n_before + 4               # cadence unbroken by the drag
            assert len(drag_records) == 4
            assert all(r["tau_ext"][0] != 0.0 for r in drag_records)   # drag = live disturbance
        finally:
            pygame.quit()
    finally:
        del CONTROLLERS["_combo_spy"]


def test_set_controller_on_plain_cart_keeps_naked_lqr_mpc_and_arms_supervisor():
    app = _make_app(plant=cart_plant(), controller_name="lqr", start="upright")  # plain cart → naked lqr
    try:
        x0 = app.loop.x.copy()
        assert app.hud()["v_limit"] is not None          # naked lqr armed
        app.set_controller("zero")
        assert app.controller_name == "zero"
        np.testing.assert_allclose(app.loop.x, x0)       # set_controller keeps state (no step)
        assert app.hud()["v_limit"] is None              # None → not armed
        app.set_controller("mpc")
        assert app.hud()["v_limit"] is not None          # naked mpc re-armed
    finally:
        pygame.quit()


def test_set_controller_on_cartpole_wraps_lqr_mpc_as_swingup_hybrid():
    from dpend.controllers.mode_switch import ModeSwitch
    app = _make_app(plant=cart_pole_plant(), controller_name="zero", start="upright")
    try:
        app.set_controller("lqr")
        assert app.controller_name == "lqr"
        assert isinstance(app._active_controller, ModeSwitch)   # robust: swing-up + lqr catch
        assert app.hud()["v_limit"] is None                     # hybrid self-supervises → App RoA not armed
        app.set_controller("mpc")
        assert isinstance(app._active_controller, ModeSwitch)   # swing-up + mpc catch
        assert app.hud()["v_limit"] is None
    finally:
        pygame.quit()


def test_in_sim_strip_swaps_controller_live_and_menu_button_requests_menu():
    """Clicking a strip controller button live-swaps the active controller
    (keeping plant state, not starting a drag); the Menu button sets
    want_menu (the Shell reads it to return to the launcher)."""
    app = _make_app(plant=cart_pole_plant(), controller_name="zero", start="upright")
    try:
        x0 = app.loop.x.copy()
        lqr_btn = next(b for b in app._controller_strip.buttons if b.value == "lqr")
        app.step_once(synthetic_events=[pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=lqr_btn.rect.center, button=1)])
        assert app.controller_name == "lqr"     # strip click swapped the controller
        assert app.dragging is False             # ... and did NOT start a cart drag
        np.testing.assert_allclose(app.loop.x, x0, atol=1e-9)  # state kept (upright is an equilibrium)

        app.step_once(synthetic_events=[pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=app._menu_button.rect.center, button=1)])
        assert app.want_menu is True
    finally:
        pygame.quit()


def test_strip_labels_zero_as_none_but_keeps_key_value():
    app = _make_app(plant=cart_pole_plant(), controller_name="zero")
    try:
        labels = {b.value: b.label for b in app._controller_strip.buttons}
        assert labels["zero"] == "None"
        assert labels["lqr"] == "lqr"
        assert "swingup" not in labels        # dropped from the UI strip too
    finally:
        pygame.quit()
