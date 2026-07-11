"""Pole-placement (full-state feedback) stabilizer.

Places the closed-loop eigenvalues of (A − BK) at chosen locations on the
linearized plant. A cheap baseline that makes controllability concrete:
poles on uncontrollable modes cannot be placed (Acrobot vs Pendubot differ).

Status: stub.
"""
from __future__ import annotations
