"""Numerical integration for the plant.

Fixed-step RK4 advancing ẋ = f(x, u) over one sim step, plus a zero-order-hold
helper so a control held across several sim steps is applied correctly.
Operates on a passed-in dynamics callable; a scipy reference integrator bounds
RK4 error in tests.

Units: time in seconds.
"""
from __future__ import annotations

import numpy as np


def rk4_step(dyn, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One classic RK4 step of ẋ = dyn(x, u), u held constant (ZOH) across
    the four stages. dt in seconds. Fixed-step by design: constant per-tick
    cost and byte-identical replays; energy drift of this non-symplectic
    scheme is measured in tests, not assumed away."""
    k1 = dyn(x, u)
    k2 = dyn(x + 0.5 * dt * k1, u)
    k3 = dyn(x + 0.5 * dt * k2, u)
    k4 = dyn(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_rollout_zoh(dyn, x: np.ndarray, u: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """Advance n_steps of size dt with the same u (one ZOH control interval)."""
    for _ in range(n_steps):
        x = rk4_step(dyn, x, u, dt)
    return x
