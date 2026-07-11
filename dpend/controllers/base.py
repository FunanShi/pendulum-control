"""Controller interface — the swappable control-law contract.

A Controller maps a state estimate x̂ ∈ ℝⁿ to an input u ∈ ℝᵐ (N·m or N,
per ``plant.input_labels``). Controllers depend on ``model`` and ``util``
only — never ``sim`` — so the same law runs unchanged on hardware. Upright
equilibrium is the origin; ``t`` is the sim clock [s]. Conventions: docs/ARCHITECTURE.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Controller(Protocol):
    def reset(self, t0: float, x0: "np.ndarray") -> None:
        """Reset internal state at episode start (t0 [s], x0 = initial estimate ∈ ℝⁿ)."""
        ...

    def update(self, t: float, x_hat: "np.ndarray") -> "np.ndarray":
        """Return input u ∈ ℝᵐ for estimate x_hat ∈ ℝⁿ at time t [s]."""
        ...
