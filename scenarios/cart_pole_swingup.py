"""Energy-shaping swing-up + LQR catch on the single-pole cart-pole plant:
hang -> pump -> catch -> hold.

x0 = plant.hanging = (0, pi, 0, 0) [m, rad, m/s, rad/s] (z = [x, theta,
xdot, thetadot], theta from the upward vertical -- upright=0, hanging=pi):
the pole starts hanging straight down. duration_s=12.0 covers the measured
~7.13 s swing-up-to-catch time (docs/design-notes/energy-swingup.md)
plus several seconds held upright.

Run with ``python batch.py scenarios/cart_pole_swingup.py``.
"""
from __future__ import annotations

import math

from dpend.config import Scenario

scenario = Scenario(
    plant="cartpole",
    actuation="",                        # N/A for cartpole (force-on-cart is fixed)
    controller="swingup",
    estimator="identity",
    sensor="perfect",
    x0=(0.0, math.pi, 0.0, 0.0),          # [x,θ,ẋ,θ̇] [m,rad,m/s,rad/s] -- plant.hanging
    duration_s=12.0,
    sim_dt_s=1e-3,
    ctrl_dt_s=5e-3,
    seed=0,
    params={"catch": "lqr"},   # swingup_factory auto-selects method="energy";
                                # c_catch/c_release default to the calibrated
                                # BASIN_V_LQR_CALIBRATED-derived values.
)
