"""Balance a cart-mounted double pendulum near upright with linear MPC
(condensed QP, osqp, hard input + rail constraints, DARE terminal cost).

Run with ``python batch.py scenarios/cart_mpc.py``.
"""
from __future__ import annotations

from dpend.config import Scenario

scenario = Scenario(
    plant="cart",
    actuation="",               # N/A for cart (force-on-cart is fixed)
    controller="mpc",
    estimator="identity",
    sensor="perfect",
    x0=(0.0, 0.15, 0.0, 0.0, 0.0, 0.0),  # [x,θ1,θ2,ẋ,θ̇1,θ̇2] [m,rad,rad,m/s,rad/s,rad/s]
    duration_s=8.0,
    sim_dt_s=1e-3,
    ctrl_dt_s=5e-3,
    seed=0,
    params={},   # factory defaults: N=40, u_max=150 N, x_max=rail-0.1 m,
                 # Q=diag(10,50,50,1,5,5), R=0.1 — tune from telemetry
)
