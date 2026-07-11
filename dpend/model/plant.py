"""Plant protocol + factories + registry — the seam every consumer sees.

A Plant bundles: state/input dims, forward dynamics ż = f(z, u, τ_ext),
total energy, forward kinematics for renderers, labels (with units),
geometry bounds, and the linearization surface (A, B_lin) controller
factories consume. New plant = new module + one registry line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from dpend.model import cart_dynamics as _cart
from dpend.model import cart_linearize as _cart_lin
from dpend.model import cart_pole_dynamics as _cart_pole
from dpend.model import cart_pole_linearize as _cart_pole_lin
from dpend.model import dynamics as _fixed
from dpend.model import linearize as _lin
from dpend.model.cart_params import CartParams
from dpend.model.cart_pole_params import CartPoleParams
from dpend.model.params import Params, actuation_matrix


@runtime_checkable
class Plant(Protocol):
    name: str
    n: int
    m: int
    state_labels: tuple
    input_labels: tuple
    reach: float
    rail: float | None
    upright: np.ndarray  # upright equilibrium state ∈ ℝⁿ (the origin), fresh copy
    hanging: np.ndarray  # hanging equilibrium state ∈ ℝⁿ, fresh copy
    lqr_weights: tuple  # (Q_diag: list, R_diag: list) -- factory-owned LQR/MPC defaults

    def f(self, z, u, tau_ext=None) -> np.ndarray: ...
    def energy(self, z) -> float: ...
    def fk(self, z): ...
    def linearize(self, z_eq) -> tuple[np.ndarray, np.ndarray]: ...
    # (A ∈ ℝⁿˣⁿ, B_lin ∈ ℝⁿˣᵐ) at an EQUILIBRIUM z_eq (q̇=0) — the plant-
    # generic surface: one controller file runs on every registered plant.


@runtime_checkable
class EnergyShapingCapable(Protocol):
    """Optional capability: the pole-subsystem energy-shaping primitives the
    swing-up controller needs. Structural (runtime_checkable) so consumers
    isinstance-gate on it instead of every Plant carrying these members."""

    energy_upright: float  # pendulum-subsystem energy at the upright equilibrium [J]

    def pendulum_energy(self, z) -> float: ...
    # pole-subsystem energy about its fixed pivot (Åström–Furuta shaping) [J]

    def accel_to_force(self, z, a_cmd) -> float: ...
    # collocated PFL inverse: desired cart accel [m/s²] -> cart force [N]


# eq=False: the `B` ndarray field makes generated __eq__/__hash__ raise;
# identity semantics are correct for a config/service object.
@dataclass(frozen=True, eq=False)
class _FixedPivotPlant:
    """Fixed-pivot double pendulum (adapter over model.dynamics)."""

    params: Params
    B: np.ndarray
    name: str = "fixed"
    n: int = 4
    state_labels: tuple = ("θ1 [rad]", "θ2 [rad]", "θ̇1 [rad/s]", "θ̇2 [rad/s]")
    reach: float = field(init=False)
    rail: None = None

    def __post_init__(self):
        object.__setattr__(self, "reach", self.params.l1 + self.params.l2)

    @property
    def m(self) -> int:
        return self.B.shape[1]

    @property
    def input_labels(self) -> tuple:
        return tuple(f"u{i} [N·m]" for i in range(self.m))

    def f(self, z, u, tau_ext=None) -> np.ndarray:
        zdot = _fixed.f(z, u, self.params, self.B)
        if tau_ext is not None:
            # generalized torque τ_ext ∈ ℝ²: q̈ += M⁻¹ τ_ext (units N·m)
            M = _fixed.mass_matrix(self.params, float(z[1]))
            zdot = zdot.copy()
            zdot[2:] += np.linalg.solve(M, np.asarray(tau_ext, float))
        return zdot

    def energy(self, z) -> float:
        return _fixed.energy(self.params, z)

    def fk(self, z):
        """(None, pts (3,2)): base at origin, elbow, tip — world frame [m]."""
        th1, th12 = float(z[0]), float(z[0]) + float(z[1])
        p = self.params
        base = np.zeros(2)
        elbow = base + [-p.l1 * np.sin(th1), p.l1 * np.cos(th1)]
        tip = elbow + [-p.l2 * np.sin(th12), p.l2 * np.cos(th12)]
        return None, np.vstack([base, elbow, tip])

    @property
    def upright(self) -> np.ndarray:
        """Upright equilibrium [rad, rad/s] — the origin; fresh copy per call."""
        return _lin.UPRIGHT.copy()

    @property
    def hanging(self) -> np.ndarray:
        """Hanging equilibrium [rad, rad/s]; fresh copy per call."""
        return _lin.HANGING.copy()

    @property
    def lqr_weights(self) -> tuple:
        """(Q_diag, R_diag) LQR/MPC defaults; override via params["Q"]/["R"]."""
        return ([10.0, 10.0, 1.0, 1.0], [0.1] * self.m)

    def linearize(self, z_eq) -> tuple[np.ndarray, np.ndarray]:
        """(A (4,4), B_lin (4,m)) at an equilibrium — delegates to the
        FD-verified model.linearize (valid at q̇=0 only; see its docstring)."""
        return _lin.linearize(self.params, self.B, z_eq)


def fixed_pivot_plant(params: Params = Params(), actuation: str = "acrobot") -> Plant:
    return _FixedPivotPlant(params=params, B=actuation_matrix(actuation))


# eq=False: convention symmetry with _FixedPivotPlant (identity semantics).
@dataclass(frozen=True, eq=False)
class _CartPlant:
    """Cart-mounted double pendulum (3 DOF: x, θ₁, θ₂)."""

    cart_params: CartParams
    name: str = "cart"
    n: int = 6
    state_labels: tuple = ("x [m]", "θ1 [rad]", "θ2 [rad]", "ẋ [m/s]", "θ̇1 [rad/s]", "θ̇2 [rad/s]")
    m: int = 1
    input_labels: tuple = ("F [N]",)
    rail: float = field(init=False)
    reach: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "rail", self.cart_params.L_rail)
        object.__setattr__(self, "reach", self.cart_params.pend.l1 + self.cart_params.pend.l2)

    def f(self, z, u, tau_ext=None) -> np.ndarray:
        return _cart.f(z, u, self.cart_params, tau_ext)

    def energy(self, z) -> float:
        return _cart.energy(self.cart_params, z)

    def fk(self, z):
        """(cart_xy [m], pts (k,2)): cart position + joint chain on cart base.

        cart_xy = (x, 0);  link chain = base + elbow + tip, world frame.
        """
        x, theta1, theta2 = float(z[0]), float(z[1]), float(z[2])
        p = self.cart_params.pend

        cart_xy = np.array([x, 0.0])
        base = cart_xy
        th1 = theta1
        th12 = theta1 + theta2
        elbow = base + np.array([-p.l1 * np.sin(th1), p.l1 * np.cos(th1)])
        tip = elbow + np.array([-p.l2 * np.sin(th12), p.l2 * np.cos(th12)])
        return cart_xy, np.vstack([base, elbow, tip])

    @property
    def upright(self) -> np.ndarray:
        """Upright equilibrium [m, rad, rad, m/s, rad/s, rad/s] — the origin;
        fresh copy per call."""
        return _cart_lin.UPRIGHT.copy()

    @property
    def hanging(self) -> np.ndarray:
        """Hanging equilibrium [m, rad, rad, m/s, rad/s, rad/s]; fresh copy per call."""
        return _cart_lin.HANGING.copy()

    @property
    def lqr_weights(self) -> tuple:
        """(Q_diag, R_diag) LQR/MPC defaults; override via params["Q"]/["R"]."""
        return ([10.0, 50.0, 50.0, 1.0, 5.0, 5.0], [0.1])

    def linearize(self, z_eq) -> tuple[np.ndarray, np.ndarray]:
        """(A (6,6), B_lin (6,1)) at an in-rail equilibrium — delegates to the
        FD-verified model.cart_linearize (valid at q̇=0, |x| < L_rail only)."""
        return _cart_lin.linearize(self.cart_params, z_eq)


def cart_plant(cart_params: CartParams = CartParams()) -> Plant:
    return _CartPlant(cart_params=cart_params)


# eq=False: convention symmetry with _FixedPivotPlant (identity semantics).
@dataclass(frozen=True, eq=False)
class _CartPolePlant:
    """Single-pole cart-pole: 2 DOF (x, θ), force-on-cart. Also
    EnergyShapingCapable (structurally, via the members below) — the
    swing-up controller isinstance-gates on that protocol, never on this
    class."""

    cart_pole_params: CartPoleParams
    name: str = "cartpole"
    n: int = 4
    state_labels: tuple = ("x [m]", "θ [rad]", "ẋ [m/s]", "θ̇ [rad/s]")
    m: int = 1
    input_labels: tuple = ("F [N]",)
    rail: float = field(init=False)
    reach: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "rail", self.cart_pole_params.L_rail)
        object.__setattr__(self, "reach", self.cart_pole_params.l)

    def f(self, z, u, tau_ext=None) -> np.ndarray:
        return _cart_pole.f(z, u, self.cart_pole_params, tau_ext)

    def energy(self, z) -> float:
        return _cart_pole.energy(self.cart_pole_params, z)

    def fk(self, z):
        """(cart_xy [m], pts (2,2)): cart position + pole tip, world frame.
        tip = cart_xy + (−l·sinθ, l·cosθ) — a single pole has no elbow point.
        """
        x, theta = float(z[0]), float(z[1])
        p = self.cart_pole_params

        cart_xy = np.array([x, 0.0])
        tip = cart_xy + np.array([-p.l * np.sin(theta), p.l * np.cos(theta)])
        return cart_xy, np.vstack([cart_xy, tip])

    @property
    def upright(self) -> np.ndarray:
        """Upright equilibrium [m, rad, m/s, rad/s] — the origin; fresh copy per call."""
        return _cart_pole_lin.UPRIGHT.copy()

    @property
    def hanging(self) -> np.ndarray:
        """Hanging equilibrium [m, rad, m/s, rad/s]; fresh copy per call."""
        return _cart_pole_lin.HANGING.copy()

    @property
    def lqr_weights(self) -> tuple:
        """(Q_diag, R_diag) starting values — tune from telemetry."""
        return ([1.0, 10.0, 1.0, 1.0], [0.05])

    def linearize(self, z_eq) -> tuple[np.ndarray, np.ndarray]:
        """(A (4,4), B_lin (4,1)) at an equilibrium — delegates to the
        FD-verified model.cart_pole_linearize (valid at q̇=0, |x| < L_rail only)."""
        return _cart_pole_lin.linearize(self.cart_pole_params, z_eq)

    # --- EnergyShapingCapable ---

    @property
    def energy_upright(self) -> float:
        return _cart_pole.energy_upright(self.cart_pole_params)

    def pendulum_energy(self, z) -> float:
        return _cart_pole.pendulum_energy(self.cart_pole_params, z)

    def accel_to_force(self, z, a_cmd) -> float:
        return _cart_pole.accel_to_force(self.cart_pole_params, z, a_cmd)


def cart_pole_plant(cart_pole_params: CartPoleParams = CartPoleParams()) -> Plant:
    return _CartPolePlant(cart_pole_params=cart_pole_params)


# Registry: fixed-pivot, double-pendulum-on-cart, and single-pole cart-pole
PLANTS = {"fixed": fixed_pivot_plant, "cart": cart_plant, "cartpole": cart_pole_plant}
