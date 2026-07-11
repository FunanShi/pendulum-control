"""ModeSwitch hysteresis state machine, unit level (no sim): fakes stand in
for the swing-up/catch children, so only the mode/hysteresis/angle-wrapping
logic is under test."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.controllers.mode_switch import ModeSwitch


class _FakeController:
    """Stateless Controller stub: fixed output; records reset() calls and the
    x_hat handed to update(), so tests can assert which child was reset on a
    transition and what state array the catch child received."""

    def __init__(self, output: float):
        self._output = np.array([output])
        self.reset_calls: list = []
        self.update_calls: list = []

    def reset(self, t0: float, x0: np.ndarray) -> None:
        self.reset_calls.append((t0, np.asarray(x0, dtype=float).copy()))

    def update(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        self.update_calls.append(np.asarray(x_hat, dtype=float).copy())
        return self._output.copy()


def test_hysteresis_state_machine_crosses_all_four_regions():
    """V=x^T x (P=eye, z_up=0), c_catch=1.0 < c_release=4.0. Walks V through:
    large (SWINGING) -> below c_catch (CATCHING) -> between the thresholds
    (stays CATCHING, no chatter) -> above c_release (SWINGING), asserting
    mode and the delegated output at every step."""
    swingup = _FakeController(+1.0)
    catch = _FakeController(-1.0)
    n = 4
    P = np.eye(n)
    z_up = np.zeros(n)
    c_catch, c_release = 1.0, 4.0
    ms = ModeSwitch(swingup, catch, P, z_up, c_catch, c_release)

    ms.reset(0.0, np.array([5.0, 0.0, 0.0, 0.0]))
    assert ms.mode == "SWINGING"
    assert len(swingup.reset_calls) == 1
    assert len(catch.reset_calls) == 1

    # V=25 >> c_catch -> stays SWINGING, delegates to swingup (+1)
    x = np.array([5.0, 0.0, 0.0, 0.0])
    u = ms.update(0.0, x)
    assert ms.mode == "SWINGING"
    np.testing.assert_array_equal(u, [1.0])

    # V=0.25 < c_catch=1.0 -> transitions to CATCHING, delegates to catch (-1)
    x = np.array([0.5, 0.0, 0.0, 0.0])
    u = ms.update(0.1, x)
    assert ms.mode == "CATCHING"
    np.testing.assert_array_equal(u, [-1.0])
    assert len(catch.reset_calls) == 2  # reset on the SWINGING->CATCHING transition

    # V=2.25, between c_catch=1.0 and c_release=4.0 -> STAYS CATCHING (hysteresis)
    x = np.array([1.5, 0.0, 0.0, 0.0])
    u = ms.update(0.2, x)
    assert ms.mode == "CATCHING"
    np.testing.assert_array_equal(u, [-1.0])
    assert len(catch.reset_calls) == 2  # no additional reset -- no transition happened

    # V=9.0 > c_release=4.0 -> transitions back to SWINGING, delegates to swingup (+1)
    x = np.array([3.0, 0.0, 0.0, 0.0])
    u = ms.update(0.3, x)
    assert ms.mode == "SWINGING"
    np.testing.assert_array_equal(u, [1.0])
    assert len(swingup.reset_calls) == 2  # reset on the CATCHING->SWINGING transition


def test_wrapped_angle_error_uses_angle_diff_not_naive_subtraction():
    """angle_idx=(1,) with z_up's angle near +pi and x_hat's near -pi (a
    branch-cut crossing): angle_diff gives a tiny wrapped error and enters
    CATCHING; the naive unwrapped V (~(2pi)^2) is measured for contrast."""
    swingup = _FakeController(+1.0)
    catch = _FakeController(-1.0)
    n = 4
    P = np.eye(n)
    z_up = np.array([0.0, np.pi, 0.0, 0.0])
    c_catch, c_release = 0.05, 0.1
    ms = ModeSwitch(swingup, catch, P, z_up, c_catch, c_release, angle_idx=(1,))
    ms.reset(0.0, z_up.copy())

    x_hat = np.array([0.0, -np.pi + 1e-3, 0.0, 0.0])  # angle wrapped-close to z_up's pi
    naive_e = x_hat - z_up
    naive_V = float(naive_e @ P @ naive_e)
    print(f"\n[wrapped angle] naive (unwrapped) V = {naive_V:.4f} (>> c_release={c_release})")
    assert naive_V > c_release  # confirms this case WOULD have missed the basin unwrapped

    u = ms.update(0.0, x_hat)
    print(f"[wrapped angle] mode = {ms.mode}")
    assert ms.mode == "CATCHING"  # wrapped V ~ (1e-3)^2, tiny -- correctly enters the basin
    np.testing.assert_array_equal(u, [-1.0])


def test_catch_child_receives_angle_rewrapped_near_z_up_not_raw_x_hat():
    """Regression from a real e2e failure: a swing-up trajectory can arrive
    with theta wound an extra 2*pi (e.g. 6.20 rad, physically -0.08 from
    upright); V is fine (angle_diff), but a raw linear catch law sees a huge
    angle error (measured -395 N, instantly re-diverging). ModeSwitch must
    hand the catch child (on reset and update) the angle re-wrapped within
    ±pi of z_up; the swing-up child gets the raw x_hat (its cos/sin law
    tolerates — and its bookkeeping wants — the true accumulated angle)."""
    swingup = _FakeController(+1.0)
    catch = _FakeController(-1.0)
    n = 4
    P = np.diag([1.0, 100.0, 1.0, 1.0])  # theta-dominant weight, like the real LQR's P
    z_up = np.zeros(n)
    c_catch, c_release = 0.05, 0.1
    ms = ModeSwitch(swingup, catch, P, z_up, c_catch, c_release, angle_idx=(1,))
    ms.reset(0.0, z_up.copy())

    theta_wrapped_true = -0.02  # small physical error from upright
    theta_raw = theta_wrapped_true + 2 * np.pi  # same state, one extra revolution
    x_hat = np.array([0.0, theta_raw, 0.0, 0.0])

    u = ms.update(0.0, x_hat)
    assert ms.mode == "CATCHING"  # V uses angle_diff already -- correctly small, enters the basin

    # reset_calls[0] is from ms.reset() at episode start (x0=z_up, wrapping a
    # no-op); reset_calls[-1] is this update()'s SWINGING->CATCHING transition
    # reset — the one that must carry the wrapped angle.
    assert len(catch.reset_calls) == 2
    assert len(catch.update_calls) == 1
    reset_x0 = catch.reset_calls[-1][1]
    update_x_hat = catch.update_calls[0]
    for received, label in ((reset_x0, "reset"), (update_x_hat, "update")):
        print(f"\n[catch rewrap] catch.{label} theta: raw={theta_raw:.6f} -> received={received[1]:.6f}")
        assert received[1] == pytest.approx(theta_wrapped_true, abs=1e-9)
        assert received[0] == 0.0 and received[2] == 0.0 and received[3] == 0.0

    # the swing-up child must see the raw angle
    ms.reset(0.0, z_up.copy())
    x_hat_swinging = np.array([0.0, theta_raw, 0.0, 5.0])  # large thetadot -> stays SWINGING
    V_check = ms._lyapunov(x_hat_swinging)
    assert V_check > c_catch  # sanity: not in the basin (thetadot term)
    ms.update(0.0, x_hat_swinging)
    assert ms.mode == "SWINGING"
    swingup_x_hat = swingup.update_calls[-1]
    print(f"[catch rewrap] swingup.update theta: raw={theta_raw:.6f} -> received={swingup_x_hat[1]:.6f}")
    assert swingup_x_hat[1] == pytest.approx(theta_raw)  # raw, not rewrapped


def test_mode_starts_swinging_before_any_update():
    """mode is "SWINGING" immediately after construction, even before
    reset()/update() are called."""
    swingup = _FakeController(+1.0)
    catch = _FakeController(-1.0)
    ms = ModeSwitch(swingup, catch, np.eye(4), np.zeros(4), 1.0, 2.0)
    assert ms.mode == "SWINGING"
