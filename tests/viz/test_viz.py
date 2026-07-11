"""FK correctness (exact geometry pins the frame convention) + smoke render
to files under the Agg backend (no display in test runs)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # before pyplot is imported anywhere in this process

import numpy as np


def test_forward_kinematics_frame_convention():
    """World frame x right / y up, θ from +y CCW+: upright puts the elbow at
    (0, l1) and tip at (0, l1+l2); θ1=+π/2 (CCW quarter turn) puts the elbow
    at (−l1, 0) — CCW from up leans toward −x. Unit link lengths."""
    from dpend.viz.animation import forward_kinematics

    elbow, tip = forward_kinematics(1.0, 1.0, np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(elbow[0], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(tip[0], [0.0, 2.0], atol=1e-12)

    elbow, tip = forward_kinematics(1.0, 1.0, np.array([np.pi / 2]), np.array([0.0]))
    np.testing.assert_allclose(elbow[0], [-1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(tip[0], [-2.0, 0.0], atol=1e-12)

    # Relative θ2: fold the elbow by +π/2 with link 1 upright → tip at (−l2, l1).
    elbow, tip = forward_kinematics(1.0, 1.0, np.array([0.0]), np.array([np.pi / 2]))
    np.testing.assert_allclose(tip[0], [-1.0, 1.0], atol=1e-12)


def _tiny_telemetry(n=25):
    from dpend.telemetry.recorder import Recorder

    r = Recorder()
    for k in range(n):
        th = 0.2 * np.sin(2 * np.pi * k / n)
        r.append(t_s=k * 0.02, x_true=np.array([th, -th, 0.0, 0.0]),
                 x_hat=np.array([th, -th, 0.0, 0.0]),
                 y=np.array([th, -th, 0.0, 0.0]), u=np.array([0.0]),
                 energy_J=12.0)
    return r.finalize()


def test_dashboard_renders_png(tmp_path):
    from dpend.viz.dashboard import dashboard

    out = tmp_path / "dash.png"
    state_labels = ("θ1 [rad]", "θ2 [rad]", "θ̇1 [rad/s]", "θ̇2 [rad/s]")
    dashboard(_tiny_telemetry(), state_labels=state_labels, input_labels=("u0 [N·m]",),
              save_path=out, show=False)
    assert out.exists() and out.stat().st_size > 5_000  # a real plot, not a stub file


# ---------------------------------------------------------------------------
# angle unwrap: a swing-up angle winding 0 -> 2pi sawtooths if plotted raw.
# _maybe_unwrap decides per-series from the same "[rad]"-suffix convention
# swingup_factory's angle_idx uses; unit-tested here, wired-in proof below.
# ---------------------------------------------------------------------------

def test_maybe_unwrap_removes_2pi_jump_for_angle_labels_only():
    from dpend.viz.dashboard import _maybe_unwrap

    # A ramp that wraps at the ±pi branch cut: raw values jump by ~-2pi at
    # the crossing (3.0 -> -3.0 is really "+0.28ish", not "-6ish").
    wrapped = np.array([3.0, 3.1, -3.1, -3.0, -2.9])
    assert np.any(np.diff(wrapped) < -np.pi)  # sanity: the input really does wrap

    unwrapped = _maybe_unwrap(wrapped, "θ [rad]")
    np.testing.assert_allclose(unwrapped, np.unwrap(wrapped))
    assert np.all(np.diff(unwrapped) > 0.0)  # monotone where the raw wraps

    # Rate / non-angle labels pass through unchanged, wrap-shaped or not —
    # endswith("[rad]") must not fire on "[rad/s]" (a real angular-velocity
    # jump is not a branch-cut artifact).
    for label in ("θ̇ [rad/s]", "x [m]", "ẋ [m/s]"):
        unchanged = _maybe_unwrap(wrapped, label)
        np.testing.assert_array_equal(unchanged, wrapped)


def _swingup_like_telemetry(n=60):
    """Synthetic telemetry whose theta ramps 0 -> 2pi but is stored wrapped to
    (-pi, pi] each sample — what an atan2-based angle convention reports and
    what the dashboard's unwrap must undo."""
    from dpend.telemetry.recorder import Recorder
    from dpend.util.angles import wrap_to_pi

    r = Recorder()
    for k in range(n):
        theta_true = 2 * np.pi * k / (n - 1)  # 0 -> 2pi over the run
        theta_wrapped = wrap_to_pi(theta_true)
        r.append(t_s=k * 0.02, x_true=np.array([0.0, theta_wrapped, 0.0, 0.0]),
                 x_hat=np.array([0.0, theta_wrapped, 0.0, 0.0]),
                 y=np.array([0.0, theta_wrapped, 0.0, 0.0]), u=np.array([0.0]),
                 energy_J=1.0)
    return r.finalize()


def test_dashboard_renders_wrapping_swingup_angle_without_error(tmp_path):
    """The real dashboard() call site fed a wrapping angle column ("θ [rad]"
    at index 1) must not raise and must produce a real (non-stub) PNG. The
    transform is unit-tested above; this proves dashboard() actually wires it in."""
    from dpend.viz.dashboard import dashboard

    out = tmp_path / "swingup_dash.png"
    state_labels = ("x [m]", "θ [rad]", "ẋ [m/s]", "θ̇ [rad/s]")
    dashboard(_swingup_like_telemetry(), state_labels=state_labels,
              input_labels=("F [N]",), save_path=out, show=False)
    assert out.exists() and out.stat().st_size > 5_000


def test_animation_saves_mp4(tmp_path):
    from dpend.model.plant import fixed_pivot_plant
    from dpend.viz.animation import animate

    tel = _tiny_telemetry()
    plant = fixed_pivot_plant()
    fk_points = np.stack([plant.fk(z)[1] for z in tel.x_true])  # (N,3,2): base,elbow,tip [m]

    out = tmp_path / "anim.mp4"
    animate(tel, fk_points, save_path=out, show=False, fps=25)
    assert out.exists() and out.stat().st_size > 5_000
