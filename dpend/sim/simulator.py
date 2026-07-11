"""Multi-rate simulation loop.

Drives the closed loop and produces a Telemetry log:

    plant integrates at sim_dt (fast);
    sensor → estimator → controller run at ctrl_dt (slower), with the control
    held (ZOH) between updates.

A Plant + concrete components go in, a Telemetry record comes out — no
plotting, no global state — so runs are deterministic given the seed. The
per-tick cadence lives in `sim.ticker.ControlTicker` (shared with the live
interactive loop); this module owns the outer tick-count loop and the
scripted-disturbance hook. Sensors/estimators/controllers arrive as
already-built objects.

Units: seconds; telemetry timestamps in ns.
"""
from __future__ import annotations

import numpy as np

from dpend.sim.integrators import rk4_step
from dpend.sim.ticker import ControlTicker
from dpend.telemetry.recorder import Recorder, Telemetry


def simulate(
    *,
    plant,
    x0: np.ndarray,
    duration_s: float,
    sim_dt_s: float,
    ctrl_dt_s: float,
    sensor,
    estimator,
    controller,
    seed: int = 0,
    step_fn=rk4_step,
    disturbance=None,
) -> Telemetry:
    """Run the closed loop; return per-control-tick Telemetry.

    One tick at t = k·ctrl_dt_s (k = 0 … N−1, N = duration/ctrl_dt):
        y = sensor.measure(t, x, rng)         # measurement (+ injected noise)
        x̂ = estimator.update(t, y, u_prev)    # estimate
        u = controller.update(t, x̂)           # control, then held (ZOH)
        τ = disturbance(t, x)                 # scripted external force, None ⇒ 0
        x ← n_sub × step_fn(·, u, sim_dt_s)   # plant at the fast rate, τ_ext applied

    step_fn (default `rk4_step`): the injected integrator — a swap point,
    never rewritten per-plant.
    disturbance: callable (t, x) -> τ_ext ∈ ℝ^(plant.n // 2) [generalized
    force/torque, plant's own units], None ⇒ zeros. Recorded verbatim in
    telemetry's `tau_ext` column.

    Times in s (telemetry converts to ns). Deterministic given seed: the only
    randomness is the numpy Generator handed to the sensor.
    """
    rng = np.random.default_rng(seed)
    ticker = ControlTicker(plant=plant, sensor=sensor, estimator=estimator,
                            controller=controller, ctrl_dt_s=ctrl_dt_s,
                            sim_dt_s=sim_dt_s, step_fn=step_fn, rng=rng)

    x = np.asarray(x0, dtype=float).copy()
    ticker.reset(x)

    rec = Recorder()
    n_ticks = round(duration_s / ctrl_dt_s)
    n_q = plant.n // 2
    for k in range(n_ticks):
        t = k * ctrl_dt_s
        tau = disturbance(t, x) if disturbance is not None else np.zeros(n_q)
        x, row = ticker.tick(t, x, tau)
        rec.append(t_s=t, x_true=row["x_true"], x_hat=row["x_hat"], y=row["y"],
                   u=row["u"], tau_ext=row["tau_ext"], energy_J=row["energy"])

    return rec.finalize()
