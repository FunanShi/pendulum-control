"""Multi-rate loop semantics: rate-ratio guard, ZOH call schedule, telemetry
shape/cadence, determinism, and energy passthrough for the frictionless
uncontrolled run."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.controllers.zero import ZeroController
from dpend.estimation.identity import IdentityEstimator
from dpend.model.plant import fixed_pivot_plant
from dpend.sensors.perfect import PerfectSensor


def _run(duration_s=1.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3, controller=None, seed=0):
    from dpend.sim.simulator import simulate

    plant = fixed_pivot_plant()
    return simulate(
        x0=np.array([0.3, -0.2, 0.0, 0.0]),
        duration_s=duration_s,
        sim_dt_s=sim_dt_s,
        ctrl_dt_s=ctrl_dt_s,
        plant=plant,
        sensor=PerfectSensor(),
        estimator=IdentityEstimator(4),
        controller=controller or ZeroController(m=1),
        seed=seed,
    )


def test_rejects_non_integer_rate_ratio():
    from dpend.sim.simulator import simulate  # noqa: F401  (import check)

    with pytest.raises(ValueError, match="integer multiple"):
        _run(ctrl_dt_s=2.5e-3, sim_dt_s=1e-3)  # ratio 2.5 → refuse


def test_telemetry_cadence_and_shapes():
    tel = _run(duration_s=1.0, ctrl_dt_s=5e-3)
    assert tel.t_ns.shape == (200,)                      # 1 s / 5 ms, t=0 incl., t=1s excl.
    np.testing.assert_array_equal(tel.t_ns[:3], [0, 5_000_000, 10_000_000])
    assert tel.x_true.shape == (200, 4)
    assert tel.u.shape == (200, 1)


class SpyController:
    """Records update() call times — pins the ZOH schedule."""

    def __init__(self, m=1):
        self.m = m
        self.calls: list[float] = []

    def reset(self, t0, x0):
        self.calls.clear()

    def update(self, t, x_hat):
        self.calls.append(t)
        return np.zeros(self.m)


def test_controller_called_once_per_ctrl_tick():
    spy = SpyController()
    _run(duration_s=0.1, ctrl_dt_s=5e-3, controller=spy)  # 20 ticks
    assert len(spy.calls) == 20
    np.testing.assert_allclose(spy.calls[:4], [0.0, 5e-3, 1e-2, 1.5e-2], atol=1e-12)


def test_deterministic_given_seed():
    a, b = _run(seed=7), _run(seed=7)
    np.testing.assert_array_equal(a.x_true, b.x_true)
    np.testing.assert_array_equal(a.u, b.u)


def test_energy_constant_through_pipeline():
    """Frictionless + zero control ⇒ recorded energy is constant to RK4-drift
    level. Bound 1e-6 relative matches the integrator-level test."""
    tel = _run(duration_s=5.0)
    E = tel.energy_J
    drift = np.max(np.abs(E - E[0])) / max(abs(E[0]), 1.0)
    print(f"\npipeline energy drift over 5 s: {drift:.3e}")
    assert drift < 1e-6


def test_x_hat_equals_x_true_with_perfect_chain():
    """PerfectSensor + IdentityEstimator ⇒ x̂ ≡ x_true in telemetry (the
    'output feedback is free' invariant at its trivial point)."""
    tel = _run(duration_s=0.5)
    np.testing.assert_array_equal(tel.x_hat, tel.x_true)
