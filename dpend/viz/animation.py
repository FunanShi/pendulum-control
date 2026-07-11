"""Animate a Plant's kinematic chain from a Telemetry log.

Draws the ordered joint-chain polyline (plus an optional cart marker and
rail) in the world frame (x right, y up) via matplotlib FuncAnimation,
replaying a finished run; optional save to mp4/gif. Generic over any Plant:
the caller (batch.py) precomputes `fk_points` from `plant.fk(z)` per
telemetry row and passes plain arrays/floats, so this module never imports
model.

Depends on: telemetry, matplotlib. Never imports sim/controllers/model.
"""
from __future__ import annotations

import numpy as np
from matplotlib import animation as mpl_animation
from matplotlib import pyplot as plt

from dpend.telemetry.recorder import Telemetry


def forward_kinematics(l1: float, l2: float, theta1: np.ndarray, theta2: np.ndarray):
    """Joint positions in the world frame (x right, y up, origin at the pivot),
    meters. l1, l2 = link lengths [m]. Angles from the upward vertical, CCW+
    (θ₂ relative to link 1): a link at absolute angle θ points along (−sinθ, cosθ).

    Two-link helper kept for its frame-convention regression test; `animate()`
    consumes precomputed `fk_points` from `plant.fk` instead. The duplication
    is the price of the DAG rule (model and viz never import each other)."""
    a1 = np.asarray(theta1, dtype=float)
    a12 = a1 + np.asarray(theta2, dtype=float)
    elbow = np.stack([-l1 * np.sin(a1), l1 * np.cos(a1)], axis=-1)
    tip = elbow + np.stack([-l2 * np.sin(a12), l2 * np.cos(a12)], axis=-1)
    return elbow, tip


def animate(tel: Telemetry, fk_points: np.ndarray, rail: float | None = None,
            save_path=None, show: bool = True, fps: int = 50) -> None:
    """Replay a finished run from precomputed forward kinematics.

    fk_points: (N, k, 2) world-frame [m] joint-chain points per telemetry
    tick (N = len(tel.t_ns)), ordered base → … → tip. Precomputed by the
    caller via `plant.fk(z)` — never recomputed here. rail: rail half-length
    [m] if the plant has one; draws the rail line and a square cart marker
    at `fk_points[:, 0]`. None (default) draws the joint polyline only.
    Frames are subsampled from telemetry ticks to ~fps, so wall-clock
    playback ≈ real time. save_path (.mp4) uses ffmpeg.
    """
    t_s = tel.t_ns * 1e-9
    fk_points = np.asarray(fk_points, dtype=float)

    duration = float(t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0
    n_frames = max(2, min(len(t_s), int(duration * fps) + 1))
    idx = np.linspace(0, len(t_s) - 1, n_frames).astype(int)

    # Viewport derived from the geometry actually visited over the run:
    # farthest any chain point gets from the origin, plus the rail extent
    # if present, with a 10% margin.
    reach = float(np.max(np.linalg.norm(fk_points, axis=-1))) if fk_points.size else 1.0
    if rail is not None:
        reach = max(reach, float(rail))
    limit = 1.1 * max(reach, 1e-6)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)

    cart = None
    if rail is not None:
        ax.plot([-rail, rail], [0.0, 0.0], "-", color="0.6", lw=2, solid_capstyle="butt")
        (cart,) = ax.plot([], [], "s", markersize=14, color="0.3")
    (line,) = ax.plot([], [], "o-", lw=3, markersize=6)
    trace, = ax.plot([], [], "-", lw=1, alpha=0.4)
    clock = ax.set_title("")

    tip_xy = fk_points[:, -1, :]  # last chain point = the end-effector/tip

    def draw(k):
        i = idx[k]
        pts = fk_points[i]
        line.set_data(pts[:, 0], pts[:, 1])
        if cart is not None:
            cart.set_data([pts[0, 0]], [pts[0, 1]])
        j0 = idx[max(0, k - 25)]
        trace.set_data(tip_xy[j0 : i + 1, 0], tip_xy[j0 : i + 1, 1])
        clock.set_text(f"t = {t_s[i]:.2f} s")
        return [a for a in (line, trace, clock, cart) if a is not None]

    anim = mpl_animation.FuncAnimation(
        fig, draw, frames=n_frames, interval=1000.0 / fps, blit=False
    )
    if save_path is not None:
        anim.save(str(save_path), writer=mpl_animation.FFMpegWriter(fps=fps))
    if show:
        plt.show()
    plt.close(fig)
