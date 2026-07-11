"""Identity estimator: x̂ = y. The default for full-state feedback.

Only valid when the paired sensor emits the full state (e.g. PerfectSensor);
asserts the shape so a mispaired scenario fails loudly instead of silently
feeding a short vector to a controller. Width `n` comes from the paired
plant's `plant.n` (4 fixed-pivot, 6 cart) so the guard scales with the plant.
State/units: x = [q, q̇] (rad, rad/s per joint; a cart plant adds x [m],
ẋ [m/s]); see docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import numpy as np


class IdentityEstimator:
    def __init__(self, n: int):
        self._n = n  # expected full-state width, from the paired plant's `n`

    def reset(self, t0: float, x0: np.ndarray) -> None:
        pass  # stateless

    def update(self, t: float, y: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        assert y.shape == (self._n,), (
            f"IdentityEstimator needs full-state measurements (shape ({self._n},)), got {y.shape}; "
            "pair it with a full-state sensor or use a real observer"
        )
        return y.copy()
