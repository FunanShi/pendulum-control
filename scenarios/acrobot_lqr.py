"""Balance an Acrobot near the upright with LQR + full-state feedback.

The canonical phase-1 smoke scenario: small tip from upright, perfect sensing,
identity estimator, LQR catch. Run with ``python batch.py scenarios/acrobot_lqr.py``.
"""
from __future__ import annotations

from dpend.config import Scenario

scenario = Scenario(
    plant="fixed",
    actuation="acrobot",
    controller="lqr",
    estimator="identity",
    sensor="perfect",
    # x0 is the measured-convergent edge of this plant's region of attraction
    # at these Q/R: a larger tip is Hurwitz for the linearized loop but
    # diverges under the true nonlinear dynamics — real physics (a small
    # RoA), not a controller bug. Measurements:
    # docs/design-notes/lqr-riccati.md, "Bonus finding".
    x0=(0.02, -0.01, 0.0, 0.0),   # small tip from the upright origin [rad, rad/s]
    duration_s=8.0,
    sim_dt_s=1e-3,
    ctrl_dt_s=5e-3,
    seed=0,
    params={
        # LQR weights — starting point; tune from telemetry.
        "Q": [10.0, 10.0, 1.0, 1.0],   # diag, penalize [θ1, θ2, θ̇1, θ̇2]
        "R": [0.1],                     # diag, penalize elbow torque [N·m]
    },
)
