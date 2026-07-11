"""Sensor interface — the swappable measurement-model contract.

A Sensor maps the true state x ∈ ℝ⁴ to a measurement ``y = h(x) + noise``.
Noise must be drawn from the injected numpy Generator so a run is reproducible
from the scenario seed. State/units/frame conventions: see docs/ARCHITECTURE.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Sensor(Protocol):
    def measure(self, t: float, x: "np.ndarray", rng: "np.random.Generator") -> "np.ndarray":
        """Return measurement ``y = h(x) + noise`` for true state x ∈ ℝ⁴ at time t [s]."""
        ...
