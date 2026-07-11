"""Physical parameters of the double pendulum (SI units, per-field comments),
plus the actuation-matrix selector: fully-actuated arm, Acrobot, or Pendubot.

Frame: link angles from the upward vertical (see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    """Physical parameters, SI units. Defaults = uniform thin rods."""

    m1: float = 1.0          # link-1 mass [kg]
    m2: float = 1.0          # link-2 mass [kg]
    l1: float = 1.0          # link-1 length, joint→joint [m]
    l2: float = 1.0          # link-2 length, joint→tip [m]
    lc1: float = 0.5         # joint-1 → link-1 COM [m]
    lc2: float = 0.5         # joint-2 → link-2 COM [m]
    I1: float = 1.0 / 12.0   # link-1 inertia about its COM [kg·m²] (m·l²/12 for defaults)
    I2: float = 1.0 / 12.0   # link-2 inertia about its COM [kg·m²]
    g0: float = 9.81         # gravitational acceleration, world −y [m·s⁻²]
    b1: float = 0.0          # joint-1 viscous friction [N·m·s·rad⁻¹] (0 ⇒ energy conserved)
    b2: float = 0.0          # joint-2 viscous friction [N·m·s·rad⁻¹]


_ACTUATION = {
    "full": np.eye(2),                     # both joints driven
    "acrobot": np.array([[0.0], [1.0]]),   # elbow only (passive shoulder)
    "pendubot": np.array([[1.0], [0.0]]),  # shoulder only (passive elbow)
}


def actuation_matrix(config: str) -> np.ndarray:
    """Actuation matrix B (2×m): generalized torque τ = B u, u ∈ ℝᵐ [N·m]."""
    try:
        return _ACTUATION[config].copy()
    except KeyError:
        raise ValueError(
            f"unknown actuation config {config!r}; expected one of {sorted(_ACTUATION)}"
        ) from None
