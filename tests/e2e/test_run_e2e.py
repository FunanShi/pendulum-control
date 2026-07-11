"""End-to-end: scenario file → batch.py main() → artifacts on disk. Headless
(Agg via MPLBACKEND env is NOT assumed here — --headless must force it)."""
from __future__ import annotations

import numpy as np


def test_run_headless_produces_artifacts(tmp_path):
    import batch as run_mod

    out = tmp_path / "drop"
    run_mod.main(["scenarios/drop_uncontrolled.py", "--headless", "--out", str(out)])

    npz = out / "telemetry.npz"
    png = out / "dashboard.png"
    assert npz.exists() and png.exists() and png.stat().st_size > 5_000

    from dpend.telemetry.formats import load_npz

    tel = load_npz(npz)
    assert tel.x_true.shape[0] == 2000            # 10 s at 200 Hz
    E = tel.energy_J
    assert np.max(np.abs(E - E[0])) / abs(E[0]) < 1e-6  # frictionless drop conserves E


def test_run_unknown_controller_fails_clearly(tmp_path, capsys):
    """Uses a controller name that will never be registered, so the test stays
    robust to whatever gets added to CONTROLLERS later."""
    import pytest

    import batch as run_mod

    bogus = tmp_path / "bogus_controller_scenario.py"
    bogus.write_text(
        "from dpend.config import Scenario\n"
        "scenario = Scenario(controller='_definitely_unregistered_xyz')\n"
    )

    with pytest.raises(SystemExit):
        run_mod.main([str(bogus), "--headless", "--out", str(tmp_path / "x")])
    assert "_definitely_unregistered_xyz" in capsys.readouterr().err  # names the missing key
