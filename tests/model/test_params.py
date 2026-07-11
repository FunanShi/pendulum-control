"""Default parameter values and actuation-matrix shapes. Units: SI throughout."""
from __future__ import annotations

import numpy as np
import pytest


def test_default_params_match_spec():
    from dpend.model.params import Params

    p = Params()
    assert p.m1 == p.m2 == 1.0          # kg
    assert p.l1 == p.l2 == 1.0          # m
    assert p.lc1 == p.lc2 == 0.5        # m (COM at mid-rod)
    assert p.I1 == p.I2 == pytest.approx(1.0 / 12.0)  # kg·m² (uniform thin rod, m·l²/12)
    assert p.g0 == 9.81                 # m·s⁻²
    assert p.b1 == p.b2 == 0.0          # N·m·s·rad⁻¹ (frictionless default: clean energy test)


def test_params_frozen():
    from dpend.model.params import Params

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        Params().m1 = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "config, expected",
    [
        ("full", np.eye(2)),
        ("acrobot", np.array([[0.0], [1.0]])),   # elbow torque only, passive shoulder
        ("pendubot", np.array([[1.0], [0.0]])),  # shoulder torque only, passive elbow
    ],
)
def test_actuation_matrix(config, expected):
    from dpend.model.params import actuation_matrix

    B = actuation_matrix(config)
    assert B.shape == expected.shape
    np.testing.assert_array_equal(B, expected)


def test_actuation_matrix_rejects_unknown():
    from dpend.model.params import actuation_matrix

    with pytest.raises(ValueError, match="unknown actuation config"):
        actuation_matrix("cartpole")
