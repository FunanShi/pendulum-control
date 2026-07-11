"""Kalman filter / Luenberger observer for output feedback (LQG).

Estimates the full state from partial, noisy measurements (e.g. angle-only)
so the controller can close the loop on x̂; pairing LQR with this gives LQG.

Status: stub — implemented with the observer/LQG work.
"""
from __future__ import annotations
