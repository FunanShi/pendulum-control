"""ReferenceSource — the setpoint a tracking controller aims for.

A plain component, independent of the `Controller.update(t, x̂)` protocol:
tracking laws close over a `ReferenceSource` at construction (via the
registry factories); regulation laws ignore it. The live UI mutates the
target in place (right-click); batch regulation runs never touch it.

Leaf module: no internal dependencies. Units/frame: the tracked signal's own
convention (e.g. cart x [m]) — this class is just a constant-hold store.
"""
from __future__ import annotations


class ReferenceSource:
    """Constant-hold reference: r(t) = target for every t, until
    `set_target()` changes it. Extension point: richer time-varying profiles
    drop in behind this same interface."""

    def __init__(self, target: float = 0.0):
        self._target = float(target)

    def r(self, t: float) -> float:
        """Reference value at time t [s]. Constant-hold: t is unused, kept in
        the signature so a time-varying profile needs no call-site changes."""
        return self._target

    def set_target(self, value: float) -> None:
        self._target = float(value)
