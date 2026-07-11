"""MPC interactive tests: registry build-via-flag, drag-release recovery, and
registry/artifacts leakage hygiene."""
from __future__ import annotations

import numpy as np
import pygame

from dpend.controllers.mpc import MPCController
from dpend.registry import CONTROLLERS
from tests._interactive_helpers import _make_app


def test_mpc_controller_builds_via_cli_flag():
    """--controller mpc resolves through the registry to a real MPCController
    instance — no App-side special-casing."""
    app = _make_app(controller_name="mpc")
    try:
        assert app.controller_name == "mpc"
        assert isinstance(app._active_controller, MPCController)
        assert hasattr(app._active_controller, "P")  # arms the RoA supervisor (see below)
    finally:
        pygame.quit()


def test_mpc_drag_release_recovers_then_manual_zeros_u():
    """The same gentle drag/release recovery as the LQR test, consuming the
    MPC's P — the DARE terminal cost (~200x LQR's CARE P; V vs V_lim stays
    self-consistent since both come from the same matrix). Measured: V
    strictly decreases on all 99 steps — the drag is well inside u_max/x_max,
    the QP's active set is empty, so the applied u equals the −K_d x̃ law the
    DARE P is the Lyapunov function for."""
    app = _make_app(controller_name="mpc")
    try:
        P = app._active_controller.P

        assert app.mode == "CONTROLLER"

        cart_px = app.w2s.to_px((0.0, 0.0))
        app.step_once(synthetic_events=[pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=(round(cart_px[0]), round(cart_px[1])), button=1)])
        assert app.dragging is True

        mouse_px = app.w2s.to_px((0.3, 0.0))
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
        print(f"\n[UI mpc recovery] V(x) first={Vs[0]:.4e}  last={Vs[-1]:.4e}  "
              f"n_upticks={int(np.sum(np.diff(Vs) > 0.0))}/99  v_limit={app.hud()['v_limit']:.4e}")
        assert np.all(np.diff(Vs) < 0.0)   # strictly decreasing, every one of the 100 ticks
        assert np.any(us != 0.0)           # the controller did real (nonzero) work

        # RoA supervisor stays quiet (parity with the LQR "must not fire" case)
        assert app.mode == "CONTROLLER"
        assert app.notice == ""

        # Select None -> MANUAL: u exactly 0 from here on (ZeroController's exact contract)
        app.set_controller("zero")
        assert app.mode == "MANUAL"
        manual_records = []
        for _ in range(10):
            manual_records += app.step_once(synthetic_events=[])
        us_manual = np.array([r["u"] for r in manual_records])
        print(f"[UI mpc recovery] u after selecting None (unique values): {np.unique(us_manual)}")
        assert np.all(us_manual == 0.0)
    finally:
        pygame.quit()


def test_mpc_no_registry_or_artifacts_leakage(tmp_path):
    """Building/driving/closing an MPC App must not mutate the shared
    CONTROLLERS registry and must not write into the real artifacts/ tree
    (every close() passes an explicit tmp out_dir)."""
    from pathlib import Path

    keys_before = set(CONTROLLERS.keys())
    real_artifacts = Path("artifacts")
    entries_before = set(real_artifacts.iterdir()) if real_artifacts.exists() else set()

    app = _make_app(controller_name="mpc")
    try:
        for _ in range(20):
            app.step_once(synthetic_events=[])
        app.set_controller("zero")
        out_dir = app.close(out_dir=tmp_path / "live_mpc_leak_check")
        assert out_dir == tmp_path / "live_mpc_leak_check"
        assert (out_dir / "telemetry.npz").exists()
    finally:
        pygame.quit()

    keys_after = set(CONTROLLERS.keys())
    entries_after = set(real_artifacts.iterdir()) if real_artifacts.exists() else set()
    print(f"\n[leakage check] registry keys before={sorted(keys_before)} after={sorted(keys_after)}")
    print(f"[leakage check] real artifacts/ entries before={len(entries_before)} "
          f"after={len(entries_after)}")
    assert keys_after == keys_before
    assert entries_after == entries_before
