"""Shared test-only helpers for the interactive-app suite: a deterministic
fake wall clock, an App-builder default-factory, and the RoA-supervisor
hard-drag driver. Underscore-prefixed so pytest does not collect this module;
test files import via `from tests._interactive_helpers import ...`."""
from __future__ import annotations

import numpy as np
import pygame

from dpend.interactive.app import DEFAULT_CTRL_DT_S, App
from dpend.interactive.ui_config import InteractiveConfig
from dpend.model.plant import cart_plant


class _StepClock:
    """Deterministic fake wall clock: advances by exactly `dt` on every call
    (reset()'s re-anchor and advance()'s elapsed-time read each consume one),
    so each App.step_once() produces exactly one control tick regardless of
    how fast the test actually runs."""

    def __init__(self, dt: float = DEFAULT_CTRL_DT_S):
        self._t = 0.0
        self._dt = dt

    def __call__(self) -> float:
        self._t += self._dt
        return self._t


def _make_app(**kwargs) -> App:
    # fps set high so clock.tick(cfg.fps) (real wall-clock pacing, independent
    # of the injected now_fn) doesn't slow the suite; correctness unaffected.
    cfg = kwargs.pop("cfg", None) or InteractiveConfig(fps=2000)
    plant = kwargs.pop("plant", None) or cart_plant()
    now_fn = kwargs.pop("now_fn", None) or _StepClock()
    pygame.init()
    screen = pygame.display.set_mode(cfg.window_px)
    return App(plant, cfg, screen, now_fn=now_fn, **kwargs)


def _hard_drag_until_fired(app, target_x: float = 1.5, reach_frames: int = 10,
                           max_frames: int = 90):
    """Grab the cart and yank the mouse target to `target_x` within
    `reach_frames` frames (the k_drag=60 N/m spring saturates against
    f_max=60 N almost immediately), then hold for the rest of `max_frames`.
    This front-loaded yank-and-hold is the profile measured to cross the RoA
    supervisor's V trigger; a constant-velocity ramp to the same endpoint is
    not (it spends most of the window at low spring force). Returns
    (fire_tick_or_None, records); fire_tick is 1-indexed from the grab. The
    mouse is left dragging — the caller releases it."""
    cart_x0 = float(app.loop.x[0])
    cart_px = app.w2s.to_px((cart_x0, 0.0))
    app.step_once(synthetic_events=[pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, pos=(round(cart_px[0]), round(cart_px[1])), button=1)])
    assert app.dragging is True

    records = []
    fire_tick = None
    for i in range(1, max_frames + 1):
        wx = cart_x0 + target_x * min(1.0, i / reach_frames)
        px = app.w2s.to_px((wx, 0.0))
        motion = pygame.event.Event(pygame.MOUSEMOTION, pos=(round(px[0]), round(px[1])),
                                    rel=(0, 0), buttons=(1, 0, 0))
        records += app.step_once(synthetic_events=[motion])
        if app.mode == "MANUAL":
            fire_tick = i
            break
    return fire_tick, records
