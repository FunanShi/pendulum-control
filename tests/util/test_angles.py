"""util.angles.angle_diff — branch-cut robustness; wrap_to_pi's own tests
live in test_smoke.py."""
from __future__ import annotations

import math

import pytest

from dpend.util.angles import angle_diff


def test_angle_diff_basic_no_wrap():
    """No branch-cut involved: angle_diff(a,b) is just a-b wrapped, and 0.2 rad
    is already inside (-pi, pi], so wrapping is a no-op here."""
    d = angle_diff(0.1, -0.1)
    assert d == pytest.approx(0.2, abs=1e-12)


def test_angle_diff_wraps_into_range():
    """a-b = 6.0 rad is outside (-pi, pi], so it must wrap by one 2*pi turn
    (6.0 - 2*pi ~= -0.283); assert the wrapped value and the contracted range."""
    d = angle_diff(3.0, -3.0)
    expected = 6.0 - 2.0 * math.pi
    print(f"\nangle_diff(3.0, -3.0) = {d:.6f} (expected {expected:.6f})")
    assert d == pytest.approx(expected, abs=1e-12)
    assert -math.pi < d <= math.pi


def test_angle_diff_across_branch_cut_stays_small():
    """pi and -pi+1e-9 are ~1e-9 apart on the circle despite being ~2*pi apart
    as plain reals — angle_diff must report the short way around. Sign is not
    asserted (it depends on which side of the cut the half-open interval
    favors); only the magnitude, which is what branch-cut-robust means."""
    d = angle_diff(math.pi, -math.pi + 1e-9)
    print(f"\nangle_diff(pi, -pi+1e-9) = {d:.3e}")
    assert abs(d) < 1e-8
