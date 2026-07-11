"""Mode switch: swing-up → LQR/MPC catch (phase 2).

A ``Controller`` that delegates to two injected children: the swing-up until
the state enters the catch's region of attraction, then the catch, with
hysteresis on the boundary to avoid chattering. Children are passed in (by
``swingup_factory``), never imported — ModeSwitch is generic over any
``base.Controller`` pair, and because it implements ``Controller`` itself,
the simulator and loop are untouched by phase 2.
"""
from __future__ import annotations

import numpy as np

from dpend.util.angles import angle_diff


class ModeSwitch:
    """Hybrid supervisor: children ``swingup``/``catch``, switched by a
    Lyapunov sublevel test with hysteresis.

    mode ∈ {"SWINGING", "CATCHING"}, starts SWINGING (construction and every
    reset()):

        SWINGING  --(V(x̂) < c_catch)-->   CATCHING
        CATCHING  --(V(x̂) > c_release)--> SWINGING

    ``c_release > c_catch`` is a required invariant, NOT checked here — the
    caller (``swingup_factory``) owns it. The gap is the dead zone that
    prevents chattering when a trajectory lingers near one threshold.

    V(x̂) = eᵀPe, e = x̂ − z_up, with the ``angle_idx`` components computed
    by ``angle_diff`` (S¹: a raw subtraction reports ~2π for a state
    essentially AT z_up across the ±π cut). P is the CATCH controller's own
    cost-to-go matrix (``catch.P``) — so "V < c_catch" means "inside the
    region the catch law's own stability argument certifies", not an
    arbitrary proxy.

    On every transition the child switched INTO is ``reset(t, x̂)`` (no
    stale memory, e.g. MPC's warm start); the child switched away from is
    left alone until its next activation.

    Angle re-wrapping before the catch child: LQR/MPC apply raw linear
    feedback about z_up, but a swing-up trajectory can arrive with θ wound
    by an extra 2π — physically at upright, numerically far from the
    linearization point (measured: one raw wound angle produced a single
    −395 N tick and an immediate bounce back to SWINGING). So on every
    catch consult (and reset), ``angle_idx`` components are re-expressed as
    ``z_up[i] + angle_diff(x̂[i], z_up[i])``. The swing-up child needs no
    wrapping — its law is built from cosθ and is 2π-invariant.
    Regression-tested in tests/test_mode_switch.py; full story in the
    swing-up design note.
    """

    #: The interactive RoA supervisor arms only for controllers that do NOT
    #: declare themselves hybrid: SWINGING deliberately roams far from
    #: upright, so an outer small-basin supervisor would fight it and revert
    #: to MANUAL on the first pumping tick. Declared, not inferred.
    IS_HYBRID = True

    def __init__(self, swingup, catch, P: np.ndarray, z_up: np.ndarray,
                 c_catch: float, c_release: float, angle_idx: tuple = ()):
        self.swingup = swingup
        self.catch = catch
        self.P = np.asarray(P, dtype=float)
        self.z_up = np.asarray(z_up, dtype=float)
        self.c_catch = float(c_catch)
        self.c_release = float(c_release)
        self.angle_idx = tuple(angle_idx)
        self.mode = "SWINGING"

    def reset(self, t0: float, x0: np.ndarray) -> None:
        """Reset both children (whichever activates first starts clean) and
        return to SWINGING — a fresh episode never assumes it is already
        caught. The catch child receives the angle-rewrapped x0, same
        contract as update()'s CATCHING branch."""
        self.swingup.reset(t0, x0)
        self.catch.reset(t0, self._wrap_for_catch(x0))
        self.mode = "SWINGING"

    def _lyapunov(self, x_hat: np.ndarray) -> float:
        x_hat = np.asarray(x_hat, dtype=float)
        e = x_hat - self.z_up
        for i in self.angle_idx:
            e[i] = angle_diff(x_hat[i], self.z_up[i])
        return float(e @ self.P @ e)

    def _wrap_for_catch(self, x_hat: np.ndarray) -> np.ndarray:
        """x̂ with `angle_idx` components re-expressed within ±π of z_up —
        see the class docstring, "Angle re-wrapping before the CATCH
        child." A no-op copy when angle_idx is empty."""
        x_wrapped = np.asarray(x_hat, dtype=float).copy()
        for i in self.angle_idx:
            x_wrapped[i] = self.z_up[i] + angle_diff(x_hat[i], self.z_up[i])
        return x_wrapped

    def update(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        x_hat = np.asarray(x_hat, dtype=float)
        V = self._lyapunov(x_hat)

        if self.mode == "SWINGING" and V < self.c_catch:
            self.mode = "CATCHING"
            self.catch.reset(t, self._wrap_for_catch(x_hat))
        elif self.mode == "CATCHING" and V > self.c_release:
            self.mode = "SWINGING"
            self.swingup.reset(t, x_hat)

        if self.mode == "CATCHING":
            return self.catch.update(t, self._wrap_for_catch(x_hat))
        return self.swingup.update(t, x_hat)
