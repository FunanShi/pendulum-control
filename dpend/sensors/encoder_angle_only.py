"""Angle-only sensor: measures (θ₁, θ₂) but not velocities.

Models joint encoders with no direct velocity readout, forcing an estimator
(observer/Kalman) to reconstruct the full state — the observability exercise.

Status: stub — implemented alongside the observer/LQG work.
"""
from __future__ import annotations
