"""Component factory registries — the seam that turns a Scenario's string
keys into concrete objects.

Each registry maps a name to a factory `(plant, params) -> component`, so
construction can depend on the resolved `Plant` (dims, weights) and on
scenario `params` without the component modules knowing about plants or
scenarios. `PLANTS` lives in `dpend.model.plant`, re-exported here so callers
have one import surface. Imported by the drivers (`batch.py`,
`dpend.interactive.app`) only — never by the components it constructs.
"""
from __future__ import annotations

import numpy as np

from dpend.controllers.energy_swingup import EnergySwingUp
from dpend.controllers.lqr import lqr_factory
from dpend.controllers.mode_switch import ModeSwitch
from dpend.controllers.mpc import mpc_factory
from dpend.controllers.zero import ZeroController
from dpend.estimation.identity import IdentityEstimator
from dpend.model.plant import EnergyShapingCapable, PLANTS
from dpend.sensors.perfect import PerfectSensor

SENSORS = {
    "perfect": lambda plant, params: PerfectSensor(),
    # Additional sensor models (e.g. "encoder_angle_only", "noisy") register here.
}

ESTIMATORS = {
    "identity": lambda plant, params: IdentityEstimator(plant.n),
    # A future state estimator (e.g. "kalman") registers here.
}

CONTROLLERS = {
    "zero": lambda plant, params: ZeroController(m=plant.m),
    "lqr": lqr_factory,
    "mpc": mpc_factory,
    # Additional controllers (e.g. "pole_placement") register here — one line each.
}

# Swing-up method registry: {name: (capability, controller class)}.
# `swingup_factory` picks by capability (isinstance against a runtime_checkable
# Protocol — model/plant.py), not plant identity: a new method is one dict
# entry + one controller class.
SWINGUP_METHODS = {"energy": (EnergyShapingCapable, EnergySwingUp)}

# Catch-basin threshold, in units of a fresh default-Q/R LQR's V = eᵀPe
# (same number as tests/test_swingup.py's BASIN_V_THRESHOLD — cross-referenced,
# not imported across the production/test boundary; swingup_factory rescales it
# by λ_min(catch.P, P_lqr) so it names the same physical neighborhood for any
# catch controller). Sized to the catch's MEASURED region of attraction (~7×
# inside it — conservative, since the true basin is not exactly a V-sublevel
# set), not to how tightly the pump parks the pole: large enough that a
# near-upright nudge is caught rather than kicked into a full swing-up, and
# the from-hanging demo settles for any value in [0.25, 4.0]. History:
# docs/design-notes/energy-swingup.md.
BASIN_V_LQR_CALIBRATED = 2.0


def _min_generalized_eig(A: np.ndarray, B: np.ndarray) -> float:
    """Smallest generalized eigenvalue λ of ``A v = λ B v`` for symmetric ``A``,
    SPD ``B`` — numpy only. Via the Cholesky factor B = LLᵀ: the generalized
    eigenvalues of (A, B) are the ordinary eigenvalues of the symmetric
    ``L⁻¹ A L⁻ᵀ``. Used to size the catch basin (A = catch.P, B = P_lqr):
    λ_min is the constant for which the Rayleigh bound
    ``eᵀP_lqr e ≤ eᵀ(catch.P) e / λ_min`` holds.
    """
    L = np.linalg.cholesky(np.asarray(B, dtype=float))
    A = np.asarray(A, dtype=float)
    M = np.linalg.solve(L, np.linalg.solve(L, A).T).T   # L⁻¹ A L⁻ᵀ (symmetric)
    return float(np.linalg.eigvalsh(M)[0])


def swingup_factory(plant, params: dict) -> ModeSwitch:
    """Build a `ModeSwitch` wrapping a swing-up child and a catch child —
    registered as `CONTROLLERS["swingup"]`.

    1. Method: `params["method"]` if given, else the first entry in
       `SWINGUP_METHODS` whose capability the plant satisfies; ValueError
       (naming the available methods) if none resolves.
    2. Swing-up child: `cls(plant, **params.get("swingup_gains", {}))` —
       empty gains use the class's tuned defaults.
    3. Catch child: `CONTROLLERS[params.get("catch", "lqr")](plant, params)`
       — the existing LQR/MPC factory, unchanged, same `params` dict.
    4. angle_idx: every state whose label ends in "[rad]" — the S¹
       components ModeSwitch must wrap with `angle_diff` (cart-pole: `(1,)`).
    5. Thresholds: `c_catch = params.get("c_catch",
       BASIN_V_LQR_CALIBRATED * lam_min)`, `c_release = 2 * c_catch` (the
       hysteresis dead zone, scaled along). `lam_min = λ_min(catch.P, P_lqr)`
       makes the catch-engage set `{eᵀ(catch.P)e < c_catch}` provably sit
       inside the LQR-validated basin: the Rayleigh bound gives
       `eᵀP_lqr e ≤ eᵀ(catch.P)e / lam_min`, so the catch engages only where
       the basin certifies stability. For an LQR catch lam_min = 1 (no
       rescale); for MPC, whose DARE P is ≈200× the CARE P (per-tick vs
       per-second costs — mpc.py's scale note), the unrescaled threshold
       would demand a ~200× tighter basin and the catch would silently never
       engage. See BASIN_V_LQR_CALIBRATED above and the swing-up design note.
    6. Returns `ModeSwitch(child, catch, catch.P, plant.upright, c_catch,
       c_release, angle_idx=angle_idx)` — `catch.P` ties the switch test to
       the same cost-to-go the catch law's own stability argument uses.
    """
    method = params.get("method")
    if method is None:
        for name, (capability, _cls) in SWINGUP_METHODS.items():
            if isinstance(plant, capability):
                method = name
                break

    if method is None or method not in SWINGUP_METHODS:
        raise ValueError(
            f"swingup_factory: no swing-up method for plant {plant.name!r} "
            f"(explicit method={method!r}); available methods: {list(SWINGUP_METHODS)}"
        )

    _capability, cls = SWINGUP_METHODS[method]
    child = cls(plant, **params.get("swingup_gains", {}))

    catch_name = params.get("catch", "lqr")
    catch = CONTROLLERS[catch_name](plant, params)

    angle_idx = tuple(
        i for i, lbl in enumerate(plant.state_labels) if lbl.strip().endswith("[rad]")
    )

    # λ_min rescale (docstring point 5): keeps the catch-engage set inside the
    # LQR-validated basin; == 1 for an LQR catch.
    P_lqr_reference = CONTROLLERS["lqr"](plant, {}).P
    lam_min = _min_generalized_eig(catch.P, P_lqr_reference)

    c_catch = params.get("c_catch", BASIN_V_LQR_CALIBRATED * lam_min)
    c_release = params.get("c_release", 2 * c_catch)

    return ModeSwitch(child, catch, catch.P, plant.upright, c_catch, c_release,
                       angle_idx=angle_idx)


CONTROLLERS["swingup"] = swingup_factory

__all__ = ["PLANTS", "SENSORS", "ESTIMATORS", "CONTROLLERS", "SWINGUP_METHODS", "swingup_factory"]
