"""RealtimeLoop — wall-clock fixed-timestep accumulator over the shared
`ControlTicker` (dpend.sim.ticker): live and batch call the exact same
`tick()`, so they cannot drift apart. Owns only the "wall time → control
ticks" bookkeeping — no pygame, no rendering. The clock is injected
(`now_fn`, default `time.perf_counter`), so the class is unit-testable with
a scripted fake clock.

Algorithm — fixed-timestep accumulator at ctrl_dt granularity: each
`advance()` adds wall-clock elapsed to an accumulator and drains whole
`ctrl_dt_s`-sized ticks from it; the sub-tick remainder carries to the next
call, so ticks stay exactly `ctrl_dt_s` apart in sim time regardless of
frame jitter. A pathological hiccup is capped at
`max_substeps_per_frame // ticker.n_sub` ticks per call; the rest of that
backlog is discarded into `dropped_s`, not carried — carrying it would make
catch-up itself late, needing more catch-up, forever (spiral of death).
`max_substeps_per_frame` counts sim_dt-rate integrator substeps
(`ticker.n_sub` per tick), not control ticks, so the substep budget holds
regardless of the ctrl_dt/sim_dt ratio a scenario picks.

Units: seconds throughout; `dropped_s` is wall-clock seconds of physics
requested but never simulated. Depends on the injected ticker by duck
typing only (never imports it). Never pygame.
"""
from __future__ import annotations

import time

import numpy as np


class RealtimeLoop:
    """Drives an injected `ticker` at fixed `ctrl_dt_s` ticks, paced by
    wall-clock time read through `now_fn`.

    Public, read-after-`advance()`/`reset()` state: `x` (current true state,
    the plant's own ℝⁿ — not UI/pixel state), `t` [s] (sim time reached so
    far), `dropped_s` [s] (cumulative wall-clock time discarded to hiccup
    caps, non-decreasing between resets — the HUD's drift indicator).
    """

    def __init__(self, *, ticker, x0, ctrl_dt_s: float,
                 max_substeps_per_frame: int = 100, now_fn=time.perf_counter):
        self._ticker = ticker
        self._ctrl_dt_s = float(ctrl_dt_s)
        self._now_fn = now_fn
        # cfg-derived cap (see module docstring): substeps -> ticks via n_sub.
        self._max_ticks_per_advance = max(1, max_substeps_per_frame // ticker.n_sub)
        self.reset(x0)

    def reset(self, x0) -> None:
        """Re-arm at true state `x0`, sim time t=0, `dropped_s`=0, and
        re-anchor the wall clock so the next `advance()` measures elapsed
        time from this instant — a mid-session reset must not produce a
        catch-up burst from wall-clock time (or an accumulator remainder)
        that predates it.
        """
        self.x = np.asarray(x0, dtype=float).copy()
        self._n_ticks_done = 0
        self.t = 0.0
        self.dropped_s = 0.0
        self._accum_s = 0.0
        self._ticker.reset(self.x)
        self._last_now = self._now_fn()

    def advance(self, tau_ext_fn) -> list[dict]:
        """Call once per rendered frame.

        Consumes wall-clock time elapsed since the last `advance()`/`reset()`
        into whole `ctrl_dt_s` ticks (remainder carried), running each
        through `ticker.tick(t, x, tau_ext_fn(t, x))`, capped per the module
        docstring.

        Returns one dict per tick actually run, shaped for
        `Recorder.append(**record)`: `{t_s, x_true, x_hat, y, u, tau_ext,
        energy_J}` — `ticker.tick`'s raw row with "energy" relabeled
        "energy_J" and this tick's `t_s` stamped on.
        """
        now = self._now_fn()
        elapsed = now - self._last_now
        self._last_now = now
        self._accum_s += elapsed

        # +eps guard: an elapsed time meant to land exactly on a tick
        # boundary (e.g. a scripted clock ticking by exactly ctrl_dt_s) must
        # not be shorted a tick by sub-ULP float noise. Mirrors
        # ControlTicker's own 1e-9-relative tolerance.
        eps = 1e-9 * self._ctrl_dt_s
        n_ticks_needed = int((self._accum_s + eps) // self._ctrl_dt_s)
        # Drain the whole backlog now, not just what we're about to run:
        # ticks beyond the cap are dropped below, so their time must not
        # linger in the accumulator either — else the drop becomes a carry
        # and the spiral of death returns one call later.
        self._accum_s -= n_ticks_needed * self._ctrl_dt_s

        n_ticks = min(n_ticks_needed, self._max_ticks_per_advance)
        self.dropped_s += (n_ticks_needed - n_ticks) * self._ctrl_dt_s

        records = []
        for _ in range(n_ticks):
            t = self._n_ticks_done * self._ctrl_dt_s  # k*ctrl_dt_s — same
            tau = tau_ext_fn(t, self.x)                # form simulate() uses,
            x_next, row = self._ticker.tick(t, self.x, tau)  # for bit-identical t
            records.append({
                "t_s": t,
                "x_true": row["x_true"],
                "x_hat": row["x_hat"],
                "y": row["y"],
                "u": row["u"],
                "tau_ext": row["tau_ext"],
                "energy_J": row["energy"],
            })
            self.x = x_next
            self._n_ticks_done += 1
        self.t = self._n_ticks_done * self._ctrl_dt_s
        return records
