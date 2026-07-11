"""Entry point: load a scenario, simulate, render, and log telemetry.

Usage:
    python batch.py scenarios/drop_uncontrolled.py [--headless] [--out DIR] [--save-anim]

A *scenario* is a small module exposing a ``scenario`` object
(:class:`dpend.config.Scenario`). ``batch.py`` resolves it into concrete
plant/sensor/estimator/controller objects via ``dpend.registry``, runs the
simulator to produce a Telemetry log, then hands that log to ``viz`` for
animation + plots.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from dpend.config import Scenario


def _die(msg: str) -> None:
    """Print msg to stderr and exit(1) — the one exit path every failure site
    in this module funnels through."""
    sys.stderr.write(msg + "\n")
    raise SystemExit(1) from None


def _load_scenario(path: str) -> Scenario:
    """Import a scenario module by file path and return its `scenario` object."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if spec is None or spec.loader is None:
        _die(f"cannot import scenario file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        sc = mod.scenario
    except AttributeError:
        _die(f"{path} defines no `scenario` object")
    if not isinstance(sc, Scenario):
        _die(f"{path}: `scenario` is not a dpend.config.Scenario")
    return sc


def _build(sc: Scenario):
    """Resolve string keys → concrete objects via ``dpend.registry`` (new
    components register in registry.py, not here)."""
    import numpy as np

    from dpend.registry import CONTROLLERS, ESTIMATORS, PLANTS, SENSORS

    try:
        plant_factory = PLANTS[sc.plant]
    except KeyError:
        _die(f"plant {sc.plant!r} not implemented yet; available: {sorted(PLANTS)}")

    # fixed_pivot_plant needs its actuation config (selects B); the other
    # factories take no argument — plant factories are not required to share
    # a call signature, only the resulting Plant object is uniform.
    plant = plant_factory(actuation=sc.actuation) if sc.plant == "fixed" else plant_factory()

    # Thread sc.reference into factory params under the same "reference" key
    # App uses, so a tracking controller factory receives it identically
    # from either driver. setdefault, not overwrite: an explicit
    # params["reference"] wins, same precedence App applies. Copy sc.params
    # first so _build never mutates the Scenario itself.
    params = dict(sc.params)
    if sc.reference is not None:
        params.setdefault("reference", sc.reference)

    def resolve(registry, key, kind):
        try:
            return registry[key](plant, params)
        except KeyError:
            _die(f"{kind} {key!r} not implemented yet; available: {sorted(registry)}")

    components = (resolve(SENSORS, sc.sensor, "sensor"),
                  resolve(ESTIMATORS, sc.estimator, "estimator"),
                  resolve(CONTROLLERS, sc.controller, "controller"))
    return plant, components, np.asarray(sc.x0, float)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="dpend scenario runner: simulate → log → render")
    ap.add_argument("scenario", help="path to a scenario module, e.g. scenarios/drop_uncontrolled.py")
    ap.add_argument("--headless", action="store_true",
                    help="no windows: force Agg backend, only write artifacts")
    ap.add_argument("--out", default=None,
                    help="artifact dir (default: artifacts/<scenario-stem>)")
    ap.add_argument("--save-anim", action="store_true",
                    help="also render animation.mp4 (needs ffmpeg — in the image)")
    args = ap.parse_args(argv)

    if args.headless:
        import matplotlib

        matplotlib.use("Agg")  # must precede any pyplot import

    sc = _load_scenario(args.scenario)
    plant, (sensor, estimator, controller), x0 = _build(sc)

    from dpend.sim.simulator import simulate

    tel = simulate(plant=plant, x0=x0, duration_s=sc.duration_s, sim_dt_s=sc.sim_dt_s,
                   ctrl_dt_s=sc.ctrl_dt_s, sensor=sensor, estimator=estimator,
                   controller=controller, seed=sc.seed, disturbance=sc.disturbance)

    out = Path(args.out) if args.out else Path("artifacts") / Path(args.scenario).stem
    out.mkdir(parents=True, exist_ok=True)

    from dpend.telemetry.formats import save_npz

    save_npz(tel, out / "telemetry.npz")

    import numpy as np

    from dpend.viz.animation import animate
    from dpend.viz.dashboard import dashboard

    fk_points = np.stack([plant.fk(z)[1] for z in tel.x_true])  # (N,k,2) [m], precomputed once

    dashboard(tel, state_labels=plant.state_labels, input_labels=plant.input_labels,
              save_path=out / "dashboard.png", show=not args.headless)
    if args.save_anim:
        animate(tel, fk_points, rail=plant.rail, save_path=out / "animation.mp4", show=False)
    if not args.headless:
        animate(tel, fk_points, rail=plant.rail, show=True)

    print(f"artifacts: {out}/telemetry.npz, dashboard.png"
          + (", animation.mp4" if args.save_anim else ""))


if __name__ == "__main__":
    main()
