"""Scenario — names every swappable piece of one run (plant, sensor, estimator,
controller, reference, disturbance) plus timing, initial condition, and
horizon. ``batch.py`` resolves it into concrete objects via ``dpend.registry``.

Units: times in seconds; angles in rad; see docs/ARCHITECTURE.md conventions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np

    from dpend.reference import ReferenceSource


@dataclass
class Scenario:
    # --- plant ---
    plant: str = "cart"                 # key into dpend.registry.PLANTS
    actuation: str = "acrobot"          # "full" | "acrobot" | "pendubot" (sets B) —
                                         # fixed-plant only: consulted only when plant=="fixed";
                                         # the cart is force-on-cart (B=[1,0,0]ᵀ, fixed by cart_dynamics).

    # --- components (registry keys) ---
    controller: str = "lqr"             # key into the controllers registry
    estimator: str = "identity"         # "identity" | "kalman"
    sensor: str = "perfect"             # "perfect" | "encoder_angle_only" | "noisy"

    # --- initial condition & horizon ---
    # x0 length matches `plant`: 4 = [θ1,θ2,θ̇1,θ̇2] (rad, rad/s) for "fixed";
    # 6 = [x,θ1,θ2,ẋ,θ̇1,θ̇2] (m, rad, rad, m/s, rad/s, rad/s) for "cart".
    # Default: small tip from the cart's upright origin.
    x0: tuple[float, ...] = (0.0, 0.05, 0.0, 0.0, 0.0, 0.0)
    duration_s: float = 10.0

    # --- multi-rate clocks ---
    sim_dt_s: float = 1e-3              # plant integration step (1 kHz)
    ctrl_dt_s: float = 5e-3            # controller / estimator step (200 Hz)

    # --- reproducibility ---
    seed: int = 0

    # --- forcing & tracking beyond the controller ---
    # disturbance(t [s], x) -> τ_ext ∈ ℝ^(plant.n // 2) [generalized force/
    # torque, plant's own units]; None ⇒ zeros.
    disturbance: "Callable[[float, np.ndarray], np.ndarray] | None" = None
    # Tracking setpoint; None ⇒ tracking-controller factories self-default
    # (`params.get("reference") or ReferenceSource()`), keeping every
    # Scenario field a plain, diffable literal.
    reference: "ReferenceSource | None" = None

    # --- component-specific params (Q/R, MPC horizon, noise σ, gains, …) ---
    params: dict = field(default_factory=dict)
