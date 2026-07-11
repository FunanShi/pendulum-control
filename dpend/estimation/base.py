"""Estimator interface — the swappable state-estimation contract.

An Estimator fuses a measurement ``y`` (from a Sensor) and the last applied
control ``u_prev`` into a full-state estimate x̂ ∈ ℝⁿ.

State/units/frame conventions: see docs/ARCHITECTURE.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Estimator(Protocol):
    def reset(self, t0: float, x0: "np.ndarray") -> None:
        """Reset at episode start (t0 [s], x0 = initial state estimate ∈ ℝⁿ)."""
        ...

    def update(self, t: float, y: "np.ndarray", u_prev: "np.ndarray") -> "np.ndarray":
        """Return estimate x̂ ∈ ℝⁿ from measurement y and last control u_prev at time t [s]."""
        ...
