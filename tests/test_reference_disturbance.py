"""Scripted batch disturbances and ReferenceSource plumbing."""
from __future__ import annotations

import numpy as np

from dpend.controllers.zero import ZeroController
from dpend.estimation.identity import IdentityEstimator
from dpend.model.plant import fixed_pivot_plant
from dpend.sensors.perfect import PerfectSensor


def test_disturbance_reaches_plant_and_telemetry():
    from dpend.sim.simulator import simulate

    plant = fixed_pivot_plant()
    kick = lambda t, z: np.array([1.0, 0.0]) if t < 0.05 else np.zeros(2)  # 50 ms shoulder push [N·m]
    tel = simulate(x0=np.zeros(4), duration_s=0.2, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
                   plant=plant, sensor=PerfectSensor(),
                   estimator=IdentityEstimator(4), controller=ZeroController(m=1),
                   disturbance=kick)
    assert tel.tau_ext.shape == (40, 2)
    assert np.any(tel.tau_ext[:10] != 0) and np.all(tel.tau_ext[-10:] == 0)
    assert np.any(tel.x_true[-1] != 0)  # the push moved the plant off equilibrium


def test_no_disturbance_records_zeros():
    from dpend.sim.simulator import simulate

    plant = fixed_pivot_plant()
    tel = simulate(x0=np.array([0.1, 0, 0, 0]), duration_s=0.1, sim_dt_s=1e-3,
                   ctrl_dt_s=5e-3, plant=plant, sensor=PerfectSensor(),
                   estimator=IdentityEstimator(4), controller=ZeroController(m=1))
    np.testing.assert_array_equal(tel.tau_ext, np.zeros((20, 2)))


def test_reference_source():
    from dpend.reference import ReferenceSource

    ref = ReferenceSource(target=0.0)
    assert ref.r(0.0) == 0.0
    ref.set_target(0.8)
    assert ref.r(12.3) == 0.8  # constant-hold semantics


def test_build_threads_scenario_reference_into_factory_params():
    """batch.py::_build threads sc.reference into factory params under the
    same "reference" key app.py uses, so a tracking controller is a drop-in
    for both drivers with no wiring differences."""
    import batch as run_mod
    from dpend.config import Scenario
    from dpend.reference import ReferenceSource
    from dpend.registry import CONTROLLERS

    received = {}

    def _spy_factory(plant, params):
        received["reference"] = params.get("reference")
        return ZeroController(m=plant.m)

    CONTROLLERS["_run_reference_spy"] = _spy_factory  # registered for this test only
    try:
        ref = ReferenceSource(0.7)
        sc = Scenario(plant="fixed", controller="_run_reference_spy",
                      x0=(0.0, 0.0, 0.0, 0.0), reference=ref)
        run_mod._build(sc)
        assert received["reference"] is ref  # the same instance, not a copy
    finally:
        del CONTROLLERS["_run_reference_spy"]  # cleanup — don't leak into other tests
