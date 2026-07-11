"""Perfect full-state sensor: y = x, no noise.

Baseline for full-state-feedback runs, paired with the identity estimator.
"""
from __future__ import annotations

import numpy as np


class PerfectSensor:
    """y = x: full state [θ₁,θ₂,θ̇₁,θ̇₂] (rad, rad/s), no noise; rng unused
    (kept for the Sensor protocol so noisy sensors are drop-in swaps)."""

    def measure(self, t: float, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(x, dtype=float).copy()
