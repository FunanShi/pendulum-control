"""Dispatcher arg-routing via RUN_DRY_RUN — no Docker/X11 touched."""
import subprocess, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "run"

def _route(*args):
    env = {**os.environ, "RUN_DRY_RUN": "1"}
    out = subprocess.run(["bash", str(RUN), *args], capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    line = [l for l in out.stdout.splitlines() if l.startswith("DRYRUN:")]
    assert len(line) == 1, out.stdout
    return line[0]

def test_bare_opens_ui():                 assert "python ui.py" in _route()
def test_leading_flag_opens_ui():         assert "python ui.py --plant cartpole" in _route("--plant","cartpole")
def test_ui_subcommand():                 assert "python ui.py --start hanging" in _route("ui","--start","hanging")
def test_batch_maps_to_batch_py():        assert "python batch.py scenarios/cart_mpc.py" in _route("batch","scenarios/cart_mpc.py")
def test_test_maps_to_pytest():           assert "pytest -k mpc" in _route("test","-k","mpc")
def test_build_maps_to_compose_build():   assert "compose build" in _route("build")
def test_shell_maps_to_bash():            assert _route("shell").endswith("bash")
def test_unknown_shows_help():            assert "unknown command" in _route("frobnicate").lower()
