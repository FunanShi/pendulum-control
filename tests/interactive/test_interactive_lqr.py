"""LQR interactive tests: drag-release recovery + manual on/off contract,
right-click target tracking while balancing, and the K/P shape contract."""
from __future__ import annotations

import numpy as np
import pygame
import pytest

from dpend.model.plant import cart_plant
from dpend.registry import CONTROLLERS
from tests._interactive_helpers import _make_app


def test_lqr_drag_release_recovers_then_manual_zeros_u():
    """Grab, drag 0.3 m, release, run 100 ticks: V(x) = x^T P x (the
    controller's own Lyapunov function, read off ctrl.P) strictly decreases
    on every tick after release. Euclidean ||x|| is deliberately not the
    metric: measured, ~48/99 ticks increase it as position and velocity trade
    off in mixed units — monotone ||x|| would be flaky by construction.
    u must be nonzero in the window; selecting None -> MANUAL then yields u
    exactly 0 (ZeroController contract). This gentle run doubles as the RoA
    supervisor's "must not fire" control case (max V ~2e-2 << V_lim): mode
    never auto-reverts, notice stays empty."""
    app = _make_app(controller_name="lqr")
    try:
        P = app._active_controller.P  # the same P the factory solved for

        assert app.mode == "CONTROLLER"

        cart_px = app.w2s.to_px((0.0, 0.0))
        app.step_once(synthetic_events=[pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(round(cart_px[0]), round(cart_px[1])), button=1)])
        assert app.dragging is True

        mouse_px = app.w2s.to_px((0.3, 0.0))  # drag 0.3 m to the right
        motion = pygame.event.Event(
            pygame.MOUSEMOTION, pos=(round(mouse_px[0]), round(mouse_px[1])),
            rel=(0, 0), buttons=(1, 0, 0))
        app.step_once(synthetic_events=[motion])

        app.step_once(synthetic_events=[pygame.event.Event(
            pygame.MOUSEBUTTONUP, pos=(round(mouse_px[0]), round(mouse_px[1])), button=1)])
        assert app.dragging is False

        records = []
        for _ in range(100):
            records += app.step_once(synthetic_events=[])
        assert len(records) == 100

        xs = np.array([r["x_true"] for r in records])
        us = np.array([r["u"] for r in records]).reshape(len(records), -1)
        Vs = np.einsum("ij,jk,ik->i", xs, P, xs)
        print(f"\n[UI lqr recovery] V(x) first={Vs[0]:.4e}  last={Vs[-1]:.4e}  "
              f"n_upticks={int(np.sum(np.diff(Vs) > 0.0))}/99")
        assert np.all(np.diff(Vs) < 0.0)   # strictly decreasing, every one of the 100 ticks
        assert np.any(us != 0.0)           # the controller did real (nonzero) work

        # RoA supervisor stays quiet on this gentle, well-inside-the-basin recovery
        assert app.mode == "CONTROLLER"
        assert app.notice == ""

        # Select None -> MANUAL: u exactly 0 from here on (ZeroController's exact contract)
        app.set_controller("zero")
        assert app.mode == "MANUAL"
        manual_records = []
        for _ in range(10):
            manual_records += app.step_once(synthetic_events=[])
        us_manual = np.array([r["u"] for r in manual_records])
        print(f"[UI lqr recovery] u after selecting None (unique values): {np.unique(us_manual)}")
        assert np.all(us_manual == 0.0)
    finally:
        pygame.quit()


def test_lqr_right_click_tracks_target_while_balancing():
    """Right-click sets a rail target; App.reference is the same
    ReferenceSource the lqr factory closed over, so the cart glides toward it
    while the pendulum stays near upright (|theta1| < 0.3 rad throughout) —
    the headless proof of the "right-click: glide-to-target" demo."""
    app = _make_app(controller_name="lqr")
    try:
        assert app.mode == "CONTROLLER"

        target_world_x = 0.4
        target_px = app.w2s.to_px((target_world_x, 0.0))
        rc = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(round(target_px[0]), round(target_px[1])), button=3)
        app.step_once(synthetic_events=[rc])
        assert app.reference.r(0.0) == pytest.approx(target_world_x, abs=1e-2)

        records = []
        for _ in range(800):  # 4 s at ctrl_dt_s=5 ms
            records += app.step_once(synthetic_events=[])

        xs = np.array([r["x_true"] for r in records])
        x_cart, th1 = xs[:, 0], xs[:, 1]
        print(f"\n[UI lqr tracking] cart x: {x_cart[0]:.4f} -> {x_cart[-1]:.4f} "
              f"(target {target_world_x})")
        print(f"[UI lqr tracking] max|theta1| over run = {np.max(np.abs(th1)):.4f} rad")
        assert abs(x_cart[-1] - target_world_x) < abs(x_cart[0] - target_world_x)  # net progress
        assert abs(x_cart[-1] - target_world_x) < 0.05   # meaningfully close by 4 s
        assert np.max(np.abs(th1)) < 0.3                 # tracking while balancing, not a fall
    finally:
        pygame.quit()


def test_lqr_controller_exposes_k_and_p_with_right_shapes():
    """LQRController.K (m,n) and .P (n,n) are public constructed attributes —
    the RoA supervisor and other Lyapunov diagnostics read them off the
    instance instead of re-deriving Q/R/solve_care."""
    plant = cart_plant()
    ctrl = CONTROLLERS["lqr"](plant, {})
    assert ctrl.K.shape == (plant.m, plant.n)
    assert ctrl.P.shape == (plant.n, plant.n)
    np.testing.assert_array_equal(ctrl.P, ctrl.P.T)   # CARE solution: exactly symmetric
    assert np.all(np.linalg.eigvalsh(ctrl.P) > 0.0)    # ... and positive definite
