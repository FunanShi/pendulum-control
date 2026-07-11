"""render.py tests: WorldToScreen projection, cart_rect_px hit-testing, and
draw_scene/_hud_lines smoke checks."""
from __future__ import annotations

import numpy as np
import pygame
import pytest

from dpend.interactive.render import WorldToScreen, cart_rect_px, draw_scene
from dpend.model.plant import cart_plant, cart_pole_plant, fixed_pivot_plant


def test_worldtoscreen_scale_derived_from_reach_and_rail_never_hardcoded():
    """Scale = min(w / (2.2*max(reach,rail)), h / (2.2*reach)) — derived from
    plant geometry on both axes, whichever is tighter, so nothing clips; for
    cart_plant() at 960x540 the vertical-derived scale wins. Origin maps to
    window center; +y world moves up the screen (the y-flip)."""
    plant = cart_plant()
    w2s = WorldToScreen(plant, (960, 540))

    horiz_m = 2.2 * max(plant.reach, plant.rail or 0.0)
    vert_m = 2.2 * plant.reach
    expected_scale = min(960 / horiz_m, 540 / vert_m)
    assert w2s.scale == pytest.approx(expected_scale)
    assert w2s.scale == pytest.approx(540 / vert_m)  # pin WHICH axis binds for these numbers

    px_origin = w2s.to_px((0.0, 0.0))
    assert px_origin == pytest.approx((480.0, 270.0))

    px_up = w2s.to_px((0.0, 1.0))
    assert px_up[1] < px_origin[1]  # +y world -> smaller pixel row (up on screen)


def test_worldtoscreen_never_clips_the_chains_full_reach_on_screen():
    """Regression for an empirically-found bug: under a horizontal-only scale
    the upright tip (0, reach) mapped ~166 px above the 960x540 window even at
    rest. Every plant's tip, straight up or down from a cart at either rail
    extreme, must land inside the window on both axes."""
    for plant in (cart_plant(), fixed_pivot_plant()):
        w2s = WorldToScreen(plant, (960, 540))
        rail = plant.rail or 0.0
        for cart_x in (-rail, 0.0, rail):
            for sign in (1.0, -1.0):
                tip_world = (cart_x, sign * plant.reach)
                px, py = w2s.to_px(tip_world)
                assert 0.0 <= px <= w2s.window_px[0], (plant.name, tip_world, (px, py))
                assert 0.0 <= py <= w2s.window_px[1], (plant.name, tip_world, (px, py))


def test_worldtoscreen_to_world_is_exact_inverse_of_to_px():
    plant = cart_plant()
    w2s = WorldToScreen(plant, (960, 540))
    wx, wy = w2s.to_world(w2s.to_px((0.37, -0.21)))
    assert (wx, wy) == pytest.approx((0.37, -0.21))


def test_cart_rect_contains_projected_cart_point_and_is_none_without_a_cart():
    plant = cart_plant()
    w2s = WorldToScreen(plant, (960, 540))
    z = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    rect = cart_rect_px(plant, z, w2s)
    assert rect.collidepoint(w2s.to_px((0.2, 0.0)))

    fixed = fixed_pivot_plant()
    w2s_fixed = WorldToScreen(fixed, (960, 540))
    assert cart_rect_px(fixed, np.zeros(4), w2s_fixed) is None  # no cart pose to hit-test


def test_draw_scene_runs_headless_and_actually_draws_something():
    """Smoke test under the dummy driver: must not crash, and must produce a
    non-blank frame (not just an untouched fill) — proves drawing calls ran,
    not merely that `screen.fill()` executed."""
    pygame.init()
    try:
        plant = cart_plant()
        screen = pygame.display.set_mode((960, 540))
        w2s = WorldToScreen(plant, (960, 540))
        hud = {"mode": "MANUAL", "controller": "zero", "t": 1.23, "energy": 4.5,
              "fps": 59.7, "dropped_s": 0.0, "force": 12.0, "f_max": 60.0, "target": 0.3}
        draw_scene(screen, plant, np.array([0.1, 0.05, -0.02, 0.0, 0.0, 0.0]), hud, w2s)
        assert pygame.surfarray.array3d(screen).std() > 0.0
    finally:
        pygame.quit()


def test_hud_lines_includes_swing_line_only_when_swing_mode_present():
    """Pure unit test of _hud_lines, no pygame Surface. `mode:` is the UI's
    own MANUAL/CONTROLLER interaction mode; `swing:` is the swing-up sub-mode
    — distinct labels, and the swing line appears only when swing_mode is not None."""
    from dpend.interactive.render import _hud_lines

    base_hud = {"mode": "CONTROLLER", "controller": "swingup", "t": 1.0, "energy": 2.0,
                "fps": 60.0, "dropped_s": 0.0}

    lines_absent = _hud_lines(base_hud)  # key entirely absent
    assert not any(line.startswith("swing:") for line in lines_absent)

    lines_none = _hud_lines({**base_hud, "swing_mode": None})  # key present, None (zero/lqr/mpc)
    assert not any(line.startswith("swing:") for line in lines_none)

    lines_swinging = _hud_lines({**base_hud, "swing_mode": "SWINGING"})
    assert "swing: SWINGING" in lines_swinging
    assert lines_swinging[0].startswith("mode: CONTROLLER")  # the two lines stay distinct

    lines_catching = _hud_lines({**base_hud, "swing_mode": "CATCHING"})
    assert "swing: CATCHING" in lines_catching


def test_draw_scene_renders_swing_mode_line_without_crashing():
    """A hud with `swing_mode` set must not crash and must still produce a
    non-blank frame under the dummy driver."""
    pygame.init()
    try:
        plant = cart_pole_plant()
        screen = pygame.display.set_mode((960, 540))
        w2s = WorldToScreen(plant, (960, 540))
        hud = {"mode": "CONTROLLER", "controller": "swingup", "t": 1.23, "energy": 4.5,
              "fps": 59.7, "dropped_s": 0.0, "force": 0.0, "f_max": 60.0, "target": None,
              "swing_mode": "SWINGING"}
        draw_scene(screen, plant, plant.hanging, hud, w2s)
        assert pygame.surfarray.array3d(screen).std() > 0.0
    finally:
        pygame.quit()


def test_draw_scene_handles_no_rail_and_no_target_generically():
    """Fixed-pivot plant (no rail, no cart) + target=None must not crash —
    the generic-over-any-Plant contract, exercised on the other plant."""
    pygame.init()
    try:
        plant = fixed_pivot_plant()
        screen = pygame.display.set_mode((640, 480))
        w2s = WorldToScreen(plant, (640, 480))
        hud = {"mode": "CONTROLLER", "controller": "zero", "t": 0.0, "energy": 0.0,
              "fps": 0.0, "dropped_s": 0.0, "force": 0.0, "f_max": 60.0, "target": None}
        draw_scene(screen, plant, np.zeros(4), hud, w2s)
    finally:
        pygame.quit()
