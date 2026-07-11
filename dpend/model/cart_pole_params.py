"""Physical parameters for the single-pole cart-pole plant. All SI units.

z = [x, theta, xdot, thetadot]; x [m] cart position (rail centered at 0),
theta [rad] pole angle from the upward vertical (upright = 0, hanging = pi).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CartPoleParams:
    """Cart + single-pole system parameters, SI units."""

    mc: float = 1.0    # cart mass [kg]
    mp: float = 0.5    # pole mass [kg]
    l: float = 0.5     # pivot -> pole COM distance [m]
    # Pole's own rotational inertia about its COM [kg*m^2] (0.0 = point mass).
    # Default mirrors Params.I1/I2 (m*length**2/12, thin-uniform-rod formula)
    # with `l` as the length scale -- a placeholder, not a rod-of-length-2l
    # derivation (that would give mp*l**2/3 about an end pivot); override I if
    # a specific rod geometry matters. The EOM use J = mp*l**2 + I, the inertia
    # about the pivot (parallel-axis theorem; see cart_pole_dynamics.py).
    I: float = 0.5 * 0.5**2 / 12
    g0: float = 9.81      # gravitational acceleration, world -y [m/s^2]
    bc: float = 0.0       # cart viscous friction [N*s/m]
    bp: float = 0.0       # pole viscous friction at the pivot [N*m*s/rad]
    L_rail: float = 1.5   # half-rail length (total span 2*L_rail) [m]
    k_stop: float = 500.0  # end-stop spring constant [N/m]
    c_stop: float = 20.0   # end-stop damping [N*s/m]
