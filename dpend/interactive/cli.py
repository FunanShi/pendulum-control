"""Live-UI CLI entry: arg parsing + `run_app`. Lives outside `app.py` so
`Shell` can be a top-level import without an app<->shell cycle (shell.py
imports App from app.py; app.py must not import shell back).
"""
from __future__ import annotations

import argparse
import sys

from dpend.interactive.shell import Shell
from dpend.interactive.ui_config import InteractiveConfig
from dpend.model.plant import PLANTS


def _parse_params(spec: "str | None") -> dict:
    """`--params k=v,k2=v2` -> `{"k": v_as_float_or_str, ...}`; `None`/`""` ->
    `{}`. Minimal parser — a richer format can replace it with no other
    call-site changes."""
    if not spec:
        return {}
    out: dict = {}
    for pair in spec.split(","):
        k, _, v = pair.partition("=")
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def _parse_ui_args(argv=None):
    """Parse the live-UI CLI. --plant/--controller/--start default to None so
    run_app can distinguish a bare invocation (all None → open the menu) from
    a direct launch (any given → skip the menu, boot the sim)."""
    ap = argparse.ArgumentParser(description="dpend — live interactive UI")
    ap.add_argument("--plant", default=None, choices=sorted(PLANTS))
    ap.add_argument("--controller", default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--start", default=None, choices=["upright", "hanging"])
    return ap.parse_args(argv)


def _selection_from_args(args) -> "dict | None":
    """None if the user gave no plant/controller/start (→ open the menu);
    else a full selection dict for a direct sim launch (missing fields
    default to cart/zero/upright)."""
    if args.plant is None and args.controller is None and args.start is None:
        return None
    return {
        "plant": args.plant or "cart",
        "controller": args.controller or "zero",
        "start": args.start or "upright",
        "params": _parse_params(args.params),
    }


def run_app(argv=None) -> None:
    """Entry point for ui.py: bare invocation opens the launcher menu; any
    --plant/--controller/--start boots that sim directly. The Shell owns the
    pygame window + loop for both."""
    sel = _selection_from_args(_parse_ui_args(argv))
    try:
        shell = Shell(InteractiveConfig(), sorted(PLANTS), start_selection=sel)
    except ValueError as exc:
        sys.stderr.write(str(exc) + "\n")
        raise SystemExit(1) from None
    shell.run()


if __name__ == "__main__":
    run_app()
