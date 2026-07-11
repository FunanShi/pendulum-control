"""CLI parsing tests: _parse_params coercion, _parse_ui_args/_selection_from_args,
and run_app's unknown-controller exit."""
from __future__ import annotations

import pygame
import pytest

from dpend.interactive.cli import _parse_params


def test_parse_params_coerces_numeric_and_keeps_strings():
    assert _parse_params(None) == {}
    assert _parse_params("") == {}
    got = _parse_params("k=1.5,mode=fast")
    assert got["k"] == pytest.approx(1.5)
    assert got["mode"] == "fast"


def test_parse_ui_args_bare_opens_menu():
    from dpend.interactive.cli import _parse_ui_args, _selection_from_args
    assert _selection_from_args(_parse_ui_args([])) is None   # bare ui.py → the launcher MENU


def test_parse_ui_args_any_flag_builds_direct_selection_with_defaults():
    from dpend.interactive.cli import _parse_ui_args, _selection_from_args
    sel = _selection_from_args(_parse_ui_args(["--plant", "cart"]))   # only plant given
    assert sel["plant"] == "cart"
    assert sel["controller"] == "zero"     # missing → default
    assert sel["start"] == "upright"       # missing → default
    assert sel["params"] == {}
    # plant default branch (`args.plant or "cart"`): a non-plant flag alone → plant falls back to "cart"
    assert _selection_from_args(_parse_ui_args(["--controller", "zero"]))["plant"] == "cart"


def test_parse_ui_args_parses_all_flags():
    from dpend.interactive.cli import _parse_ui_args, _selection_from_args
    sel = _selection_from_args(_parse_ui_args(
        ["--plant", "cartpole", "--controller", "swingup", "--start", "hanging",
         "--params", "k=1.5,name=foo"]))
    assert sel["plant"] == "cartpole"
    assert sel["controller"] == "swingup"
    assert sel["start"] == "hanging"
    assert sel["params"]["k"] == pytest.approx(1.5)
    assert sel["params"]["name"] == "foo"


def test_run_app_unknown_controller_exits_clearly(capsys):
    from dpend.interactive.cli import run_app
    with pytest.raises(SystemExit):
        run_app(["--controller", "nope-not-registered"])   # error surfaces at Shell/App build, before run()
    assert "nope-not-registered" in capsys.readouterr().err
    pygame.quit()
