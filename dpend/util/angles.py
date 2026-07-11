"""Angle utilities (leaf module — no internal dependencies).

Conventions: radians, CCW-positive. See docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import math


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to (-pi, pi]. Units: rad in, rad out. Scalar form.

    The upper bound is inclusive (pi maps to pi, not -pi).

    Note: math.remainder(x, 2*pi) is NOT used here -- its round-half-to-even at
    the boundary sends 3*pi to -pi, breaking the (-pi, pi] contract.
    """
    # (pi - x) % 2pi lands in [0, 2pi); subtracting from pi gives (-pi, pi].
    return math.pi - (math.pi - angle_rad) % (2.0 * math.pi)


def angle_diff(a: float, b: float) -> float:
    """Signed shortest angular distance a-b, wrapped to (-pi, pi] [rad in, rad out].

    Plain subtraction breaks near +/-pi (179 deg vs -179 deg is a 2 deg error,
    not 358); wrapping the difference keeps control laws branch-cut-robust.
    """
    return wrap_to_pi(a - b)
