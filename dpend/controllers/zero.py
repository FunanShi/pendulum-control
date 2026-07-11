"""Zero controller: u = 0 ∈ ℝᵐ — the null law that validates the
sim/telemetry/viz pipeline with no controller in the loop.
"""
from __future__ import annotations

import numpy as np


class ZeroController:
    def __init__(self, m: int):
        self._m = m  # input dimension (= actuation matrix column count)

    def reset(self, t0: float, x0: np.ndarray) -> None:
        pass  # stateless

    def update(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        return np.zeros(self._m)
