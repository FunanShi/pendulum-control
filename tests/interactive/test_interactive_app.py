"""Core App tests: boot/wiring, drag + keyboard input mechanics, R-reset,
right-click reference, scenario disturbance, quit/ESC/close exit paths, and
tip-trace history."""
from __future__ import annotations

import numpy as np
import pygame
import pytest

from dpend.telemetry.formats import load_npz
from tests._interactive_helpers import _make_app


def test_app_boots_on_cart_plant():
    app = _make_app()
    try:
        assert app.plant.name == "cart"
        assert app.mode == "MANUAL"
        assert app.controller_name == "zero"
        assert app.w2s.window_px == app.cfg.window_px

        records = app.step_once(synthetic_events=[])
        assert len(records) == 1  # exactly one control tick per frame at this fake-clock rate
    finally:
        pygame.quit()


def test_app_rejects_unknown_controller_name_clearly():
    with pytest.raises(ValueError, match="nope-not-registered"):
        _make_app(controller_name="nope-not-registered")
    pygame.quit()  # App.__init__ raised before assigning self.running; no session to close


def test_drag_grab_produces_nonzero_tau_ext_while_dragging_then_zero_after_release():
    app = _make_app()
    try:
        cart_px = app.w2s.to_px((0.0, 0.0))          # cart starts at x=0 (x0 = zeros)
        mouse_px = app.w2s.to_px((0.3, 0.0))          # drag target: 0.3 m to the right

        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(round(cart_px[0]), round(cart_px[1])), button=1))
        app.step_once()  # real queue: grab succeeds (click lands inside the cart rect)
        assert app.dragging is True

        pygame.event.post(pygame.event.Event(
            pygame.MOUSEMOTION, pos=(round(mouse_px[0]), round(mouse_px[1])),
            rel=(0, 0), buttons=(1, 0, 0)))
        r1 = app.step_once()  # real queue: dragging away from the cart -> spring force
        assert r1[0]["tau_ext"][0] != 0.0

        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONUP, pos=(round(mouse_px[0]), round(mouse_px[1])), button=1))
        app.step_once()          # real queue: release
        assert app.dragging is False
        r3 = app.step_once()     # idle: no drag, no key
        assert r3[0]["tau_ext"][0] == 0.0
    finally:
        pygame.quit()


def test_click_outside_cart_rect_does_not_grab():
    app = _make_app()
    try:
        far_px = app.w2s.to_px((1.3, 0.0))  # well clear of the ~0.3m-wide cart rect
        r = app.step_once(synthetic_events=[pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(round(far_px[0]), round(far_px[1])), button=1)])
        assert app.dragging is False
        assert r[0]["tau_ext"][0] == 0.0
    finally:
        pygame.quit()


def test_held_arrow_key_adds_signed_force_without_dragging():
    app = _make_app()
    try:
        right_down = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
        r0 = app.step_once(synthetic_events=[right_down])
        assert r0[0]["tau_ext"][0] == pytest.approx(app.cfg.f_key)  # +f_key, held, not dragging

        right_up = pygame.event.Event(pygame.KEYUP, key=pygame.K_RIGHT)
        r1 = app.step_once(synthetic_events=[right_up])
        assert r1[0]["tau_ext"][0] == 0.0  # released -> back to idle

        left_down = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)
        r2 = app.step_once(synthetic_events=[left_down])
        assert r2[0]["tau_ext"][0] == pytest.approx(-app.cfg.f_key)
    finally:
        pygame.quit()


def test_both_arrow_keys_held_cancel_to_zero():
    app = _make_app()
    try:
        app.step_once(synthetic_events=[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)])
        r = app.step_once(synthetic_events=[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)])
        assert r[0]["tau_ext"][0] == 0.0  # both held -> key_dir cancels to 0
    finally:
        pygame.quit()


def test_window_focus_lost_clears_held_keys():
    """Alt-tab away while LEFT is held: the matching KEYUP never arrives, so
    without an explicit clear on WINDOWFOCUSLOST the key would stay held — and
    its force applied — forever. No KEYUP is posted here; the focus-lost event
    alone must zero the contribution."""
    app = _make_app()
    try:
        left_down = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)
        r0 = app.step_once(synthetic_events=[left_down])
        assert r0[0]["tau_ext"][0] == pytest.approx(-app.cfg.f_key)  # held: force applied

        focus_lost = pygame.event.Event(pygame.WINDOWFOCUSLOST)
        r1 = app.step_once(synthetic_events=[focus_lost])  # still no KEYUP posted
        assert app.key_dir == 0            # _keys_held cleared
        assert r1[0]["tau_ext"][0] == 0.0  # so hand force contribution is 0
    finally:
        pygame.quit()


def test_r_key_resets_state_to_x0():
    app = _make_app()
    try:
        push = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
        for _ in range(20):
            app.step_once(synthetic_events=[push])
        assert not np.allclose(app.loop.x, app.x0)  # pushed well off x0

        release = pygame.event.Event(pygame.KEYUP, key=pygame.K_RIGHT)
        app.step_once(synthetic_events=[release])     # let go before reset (isolate R's effect)

        reset_ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)
        records = app.step_once(synthetic_events=[reset_ev])
        assert records[0]["t_s"] == 0.0  # fresh session's first tick
        np.testing.assert_allclose(records[0]["x_true"], app.x0, atol=1e-9)  # exactly from x0

        # R mid-drag: grab the at-rest cart, drag it away (a live, force-
        # producing grab), then press R without ever posting MOUSEBUTTONUP —
        # a stale grab must not keep pulling the just-reset cart.
        cart_px = app.w2s.to_px((0.0, 0.0))
        mouse_px = app.w2s.to_px((0.3, 0.0))
        app.step_once(synthetic_events=[pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(round(cart_px[0]), round(cart_px[1])), button=1)])
        assert app.dragging is True

        drag_motion = pygame.event.Event(pygame.MOUSEMOTION,
                                         pos=(round(mouse_px[0]), round(mouse_px[1])),
                                         rel=(0, 0), buttons=(1, 0, 0))
        r_drag = app.step_once(synthetic_events=[drag_motion])
        assert r_drag[0]["tau_ext"][0] != 0.0  # confirms the grab was live BEFORE the reset

        r_reset2 = app.step_once(synthetic_events=[pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_r)])
        assert app.dragging is False              # R cleared the stale grab
        assert r_reset2[0]["tau_ext"][0] == 0.0    # same frame: force already zero

        r_after = app.step_once(synthetic_events=[])  # idle frame: mouse still "physically" down
        assert r_after[0]["tau_ext"][0] == 0.0        # next frame: no lingering pull either
    finally:
        pygame.quit()


def test_right_click_sets_reference_target_and_hud_reflects_it():
    app = _make_app()
    try:
        target_px = app.w2s.to_px((0.5, 0.0))
        rc = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(round(target_px[0]), round(target_px[1])), button=3)
        app.step_once(synthetic_events=[rc])

        assert app.reference.r(0.0) == pytest.approx(0.5, abs=1e-2)  # px roundtrip tolerance
        assert app.hud()["target"] == pytest.approx(0.5, abs=1e-2)
    finally:
        pygame.quit()


def test_right_click_beyond_rail_clamps_to_rail_end():
    app = _make_app()  # default cart: L_rail = 1.5 m
    try:
        far_px = app.w2s.to_px((5.0, 0.0))  # well beyond the rail
        rc = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(round(far_px[0]), round(far_px[1])), button=3)
        app.step_once(synthetic_events=[rc])
        assert app.reference.r(0.0) == pytest.approx(app.plant.rail, abs=1e-2)
    finally:
        pygame.quit()


def test_left_and_right_click_are_independent_gestures():
    app = _make_app()
    try:
        cart_px = app.w2s.to_px((0.0, 0.0))
        left = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(round(cart_px[0]), round(cart_px[1])), button=1)
        app.step_once(synthetic_events=[left])
        assert app.reference.r(0.0) == pytest.approx(0.0)  # left-click never touches the reference
    finally:
        pygame.quit()


def test_scenario_disturbance_summed_unclamped_with_hand_force():
    big_kick = lambda t, z: np.array([100.0, 0.0, 0.0])  # exceeds f_max=60 on purpose
    app = _make_app(scenario_params={"disturbance": big_kick})
    try:
        records = app.step_once(synthetic_events=[])  # no hand on the cart at all
        assert records[0]["tau_ext"][0] == pytest.approx(100.0)  # unclamped: not capped to f_max
    finally:
        pygame.quit()


def test_quit_event_stops_the_session():
    app = _make_app()
    try:
        app.step_once(synthetic_events=[pygame.event.Event(pygame.QUIT)])
        assert app.running is False
        assert app.step_once(synthetic_events=[]) == []  # a dead session steps no physics
    finally:
        pygame.quit()


def test_esc_through_real_queue_exits_run_loop_and_writes_npz(tmp_path):
    """The production exit path end-to-end: physics frames, then an ESC posted
    to pygame's real event queue (the path run() pumps), then run() itself must
    consume it, flip `running`, and land in close() → npz. out_dir keeps the
    write out of the real artifacts/ tree."""
    app = _make_app()
    try:
        n_frames = 5
        for _ in range(n_frames):
            app.step_once(synthetic_events=[])  # bypasses the real queue: ESC stays queued

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        app.run(out_dir=tmp_path / "live_esc")  # blocking loop: must exit on its own

        assert app.running is False
        tel = load_npz(tmp_path / "live_esc" / "telemetry.npz")
        # The ESC frame itself records no tick (step_once returns before
        # advancing once running is False), so exactly the 5 pre-ESC frames land.
        assert tel.x_true.shape[0] == n_frames
        assert tel.tau_ext.shape == (n_frames, app.plant.n // 2)
    finally:
        pygame.quit()  # idempotent; close() already tore pygame down on the happy path


def test_close_writes_npz_with_tau_ext_column_and_expected_row_count(tmp_path):
    app = _make_app()
    n_frames = 7
    for _ in range(n_frames):
        app.step_once(synthetic_events=[])

    out_dir = app.close(out_dir=tmp_path / "live_test")
    tel = load_npz(out_dir / "telemetry.npz")
    assert tel.tau_ext.shape == (n_frames, app.plant.n // 2)
    assert tel.x_true.shape[0] == n_frames
    assert tel.x_true.shape[1] == app.plant.n


def test_close_with_zero_ticks_prints_notice_and_skips_finalize(tmp_path, capsys):
    """Quitting before any control tick advanced must not crash on
    Recorder.finalize()'s >=1-row contract: close() prints a one-line notice,
    writes no npz, and creates no artifacts dir at all."""
    app = _make_app()
    try:
        out_dir = tmp_path / "live_empty"
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        app.run(out_dir=out_dir)  # real queue: ESC consumed on frame 1, before any advance()

        assert app.running is False
        assert not out_dir.exists()  # close() must not even mkdir when there's nothing to save
        assert "no telemetry recorded (0 control ticks)" in capsys.readouterr().out
    finally:
        pygame.quit()  # idempotent; close() already tore pygame down on the happy path


def test_tip_trace_accumulates_is_bounded_and_clears_on_reset():
    app = _make_app()
    try:
        for _ in range(5):
            app.step_once(synthetic_events=[])
        assert len(app.tip_trace) == 5              # one point per physics-advancing frame
        assert app.hud()["tip_trace"] is app.tip_trace  # the hud hands draw_scene this buffer

        maxlen = app.tip_trace.maxlen
        for _ in range(maxlen + 10):
            app.step_once(synthetic_events=[])
        assert len(app.tip_trace) == maxlen          # bounded: old points fall off

        app.step_once(synthetic_events=[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)])
        # R cleared the stale history; the reset frame's own fresh tick then
        # appended exactly one new point from x0.
        assert len(app.tip_trace) == 1
    finally:
        pygame.quit()
