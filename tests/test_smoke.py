"""Scaffold smoke tests: package wiring and interface contracts only (no
numpy-dependent algorithm properties — those live with their implementations)."""
from __future__ import annotations


def test_package_imports() -> None:
    import dpend

    assert dpend.__version__


def test_interfaces_exist() -> None:
    from dpend.controllers.base import Controller
    from dpend.estimation.base import Estimator
    from dpend.sensors.base import Sensor

    for proto in (Controller, Estimator, Sensor):
        assert hasattr(proto, "__protocol_attrs__") or hasattr(proto, "_is_protocol")


def test_scenario_defaults() -> None:
    from dpend.config import Scenario

    s = Scenario()
    assert s.plant in {"cart", "fixed"}
    assert len(s.x0) in (4, 6)
    assert s.sim_dt_s <= s.ctrl_dt_s  # plant integrates at least as fast as control


def test_wrap_to_pi() -> None:
    import math

    from dpend.util.angles import wrap_to_pi

    assert math.isclose(wrap_to_pi(0.0), 0.0)
    assert math.isclose(wrap_to_pi(3.0 * math.pi), math.pi, abs_tol=1e-12)
    assert math.isclose(wrap_to_pi(-3.0 * math.pi), math.pi, abs_tol=1e-12)
