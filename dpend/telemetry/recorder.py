"""Telemetry record + recorder.

The Telemetry container (time-aligned numpy arrays; units on the class) and
a Recorder the simulator appends to each control tick.

Depends on: util. Never imports sim/controllers (it is a plain data type).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Telemetry:
    """Time-aligned per-control-tick arrays. N ticks; p = measurement dim,
    m = input dim, n_q = generalized-force dim (= plant.n // 2). Units: t_ns
    [ns, int64]; x_* [rad, rad/s] (+ [m, m/s] cart row); u [N·m] or [N];
    tau_ext [N·m] or [N] — same units as u, whatever generalized force the
    plant's second half of state responds to; energy_J [J]. Frame
    conventions: docs/ARCHITECTURE.md."""

    t_ns: np.ndarray      # (N,)  int64
    x_true: np.ndarray    # (N,n) true state
    x_hat: np.ndarray     # (N,n) estimate the controller consumed
    y: np.ndarray         # (N,p) measurement
    u: np.ndarray         # (N,m) control applied over [t_k, t_k + ctrl_dt)
    tau_ext: np.ndarray   # (N,n_q) external generalized force/torque that tick (batch: zeros)
    energy_J: np.ndarray  # (N,)  total energy of x_true


@dataclass
class Recorder:
    """Append one row per CONTROL tick; finalize() → immutable-ish Telemetry.
    The simulator owns the append cadence; this class is dumb storage."""

    _rows: list = field(default_factory=list)

    def append(self, *, t_s: float, x_true, x_hat, y, u, tau_ext=None, energy_J: float) -> None:
        x_true = np.asarray(x_true, float)
        if tau_ext is None:
            tau_ext = np.zeros(x_true.shape[0] // 2)  # no external forcing this tick
        self._rows.append(
            (int(round(t_s * 1e9)),  # s → ns, exact for tick-aligned times
             x_true.copy(),
             np.asarray(x_hat, float).copy(),
             np.asarray(y, float).copy(),
             np.asarray(u, float).copy(),
             np.asarray(tau_ext, float).copy(),
             float(energy_J))
        )

    def finalize(self) -> Telemetry:
        t, xt, xh, y, u, tau, e = zip(*self._rows)
        return Telemetry(
            t_ns=np.array(t, dtype=np.int64),
            x_true=np.stack(xt),
            x_hat=np.stack(xh),
            y=np.stack(y),
            u=np.stack(u),
            tau_ext=np.stack(tau),
            energy_J=np.array(e, dtype=float),
        )
