"""Protocol conformance (runtime_checkable isinstance) + behavior of the three
trivial loop components used by the uncontrolled-drop pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.controllers.base import Controller
from dpend.estimation.base import Estimator
from dpend.sensors.base import Sensor


def test_perfect_sensor():
    from dpend.sensors.perfect import PerfectSensor

    s = PerfectSensor()
    assert isinstance(s, Sensor)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = s.measure(0.0, x, np.random.default_rng(0))
    np.testing.assert_array_equal(y, x)
    assert y is not x  # defensive copy: sensor output must not alias sim state


def test_identity_estimator():
    from dpend.estimation.identity import IdentityEstimator

    e = IdentityEstimator(4)
    assert isinstance(e, Estimator)
    e.reset(0.0, np.zeros(4))
    y = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(e.update(0.0, y, np.zeros(1)), y)
    with pytest.raises(AssertionError):
        e.update(0.0, np.array([0.1, 0.2]), np.zeros(1))  # not full state → refuse


def test_zero_controller():
    from dpend.controllers.zero import ZeroController

    c = ZeroController(m=1)
    assert isinstance(c, Controller)
    c.reset(0.0, np.zeros(4))
    u = c.update(0.0, np.array([0.5, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(u, np.zeros(1))
