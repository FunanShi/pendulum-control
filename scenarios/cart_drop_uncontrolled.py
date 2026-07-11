"""Uncontrolled drop of the cart-mounted double pendulum.

Cart at x=0 with small pendulum angular perturbations: the cart stays
centered in-rail, the pendulum falls and swings chaotically. No control
(u=0): the telemetry energy trace must stay flat (conservation validator).

Run: python batch.py scenarios/cart_drop_uncontrolled.py
"""
from __future__ import annotations

from dpend.config import Scenario

scenario = Scenario(
    plant="cart",
    actuation="",  # N/A for cart (force-on-cart is fixed)
    controller="zero",
    estimator="identity",
    sensor="perfect",
    x0=(0.0, 0.3, -0.2, 0.0, 0.0, 0.0),  # [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂] [m, rad, rad, m/s, rad/s, rad/s]
    duration_s=10.0,
    sim_dt_s=1e-3,  # 1 kHz plant
    ctrl_dt_s=5e-3,  # 200 Hz loop
    seed=0,
)
