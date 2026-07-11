"""Uncontrolled drop from a tip near upright (fixed-pivot plant).

No control (u = 0): the pendulum falls and swings chaotically; with zero
friction the telemetry energy trace must stay flat (conservation validator).
Run: python batch.py scenarios/drop_uncontrolled.py
"""
from __future__ import annotations

from dpend.config import Scenario

scenario = Scenario(
    plant="fixed",
    actuation="acrobot",       # B irrelevant at u=0
    controller="zero",
    estimator="identity",
    sensor="perfect",
    x0=(0.3, -0.2, 0.0, 0.0),  # rad — a real tip, falls immediately
    duration_s=10.0,
    sim_dt_s=1e-3,             # 1 kHz plant
    ctrl_dt_s=5e-3,            # 200 Hz loop
    seed=0,
)
