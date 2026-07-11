"""Persist / load a Telemetry record.

Save and load to ``.npz`` (primary; lossless numpy) with a CSV export for quick
inspection. Keeps the on-disk schema in one place so viz and analysis stay in
sync with the recorder.

Depends on: telemetry.recorder, util. Never imports sim/controllers.
"""
from __future__ import annotations

import numpy as np

from dpend.telemetry.recorder import Telemetry

_FIELDS = ("t_ns", "x_true", "x_hat", "y", "u", "tau_ext", "energy_J")


def save_npz(tel: Telemetry, path) -> None:
    """Lossless primary format: one array per field."""
    np.savez(path, **{f: getattr(tel, f) for f in _FIELDS})


def load_npz(path) -> Telemetry:
    with np.load(path) as z:
        return Telemetry(**{f: z[f] for f in _FIELDS})


def export_csv(tel: Telemetry, path) -> None:
    """Flat CSV for quick inspection; npz stays the source of truth."""
    cols = [("t_ns", tel.t_ns[:, None])]
    for name in ("x_true", "x_hat", "y", "u", "tau_ext"):
        arr = getattr(tel, name)
        cols += [(f"{name}_{i}", arr[:, i : i + 1]) for i in range(arr.shape[1])]
    cols.append(("energy_J", tel.energy_J[:, None]))
    header = ",".join(name for name, _ in cols)
    data = np.hstack([c for _, c in cols])
    np.savetxt(path, data, delimiter=",", header=header, comments="")
