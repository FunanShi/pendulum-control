"""Physical parameters for the cart-mounted double pendulum.

Composition: pend (Params) + cart parameters (mass, damping, rail, stops).
All SI units.
"""
from __future__ import annotations

from dataclasses import dataclass

from dpend.model.params import Params


@dataclass(frozen=True)
class CartParams:
    """Cart + pendulum system parameters, SI units."""

    pend: Params = Params()  # double pendulum on the cart
    mc: float = 1.0  # cart mass [kg]
    bc: float = 0.0  # cart viscous friction [N·s/m]
    L_rail: float = 1.5  # half-rail length (total span 2*L_rail) [m]
    k_stop: float = 500.0  # end-stop spring constant [N/m]
    c_stop: float = 20.0  # end-stop damping [N·s/m]
