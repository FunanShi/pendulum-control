"""Recorder append/finalize semantics + lossless npz round-trip + csv export.
Timestamps: ns int64 (docs/ARCHITECTURE.md convention)."""
from __future__ import annotations

import numpy as np


def _small_recording():
    from dpend.telemetry.recorder import Recorder

    r = Recorder()
    for k in range(3):
        t = k * 5e-3  # 200 Hz control ticks [s]
        r.append(
            t_s=t,
            x_true=np.array([0.1 * k, 0.0, 0.0, 0.0]),
            x_hat=np.array([0.1 * k, 0.0, 0.0, 0.0]),
            y=np.array([0.1 * k, 0.0, 0.0, 0.0]),
            u=np.array([0.5 * k]),
            tau_ext=np.zeros(2),
            energy_J=10.0 - k,
        )
    return r.finalize()


def test_recorder_shapes_and_ns_timestamps():
    tel = _small_recording()
    assert tel.t_ns.dtype == np.int64
    np.testing.assert_array_equal(tel.t_ns, [0, 5_000_000, 10_000_000])  # 5 ms = 5e6 ns
    assert tel.x_true.shape == (3, 4)
    assert tel.x_hat.shape == (3, 4)
    assert tel.y.shape == (3, 4)
    assert tel.u.shape == (3, 1)
    assert tel.tau_ext.shape == (3, 2)
    assert tel.energy_J.shape == (3,)


def test_npz_roundtrip_lossless(tmp_path):
    from dpend.telemetry.formats import load_npz, save_npz

    tel = _small_recording()
    path = tmp_path / "run.npz"
    save_npz(tel, path)
    back = load_npz(path)
    for field in ("t_ns", "x_true", "x_hat", "y", "u", "tau_ext", "energy_J"):
        np.testing.assert_array_equal(getattr(tel, field), getattr(back, field))
        assert getattr(tel, field).dtype == getattr(back, field).dtype


def test_csv_export(tmp_path):
    from dpend.telemetry.formats import export_csv

    tel = _small_recording()
    path = tmp_path / "run.csv"
    export_csv(tel, path)
    lines = path.read_text().strip().splitlines()
    assert lines[0].split(",")[:2] == ["t_ns", "x_true_0"]  # header present
    assert len(lines) == 1 + 3                              # header + 3 rows
