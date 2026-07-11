"""ControlTicker — one control tick, shared by every time-stepping driver.

Bundles the multi-rate cadence (sensor → estimator → controller at ctrl_dt,
plant integration at sim_dt with u held ZOH) into a single reusable core so
the batch `simulate()` loop and the live UI loop cannot drift apart: both
call the same `tick()`. The integrator is injected via `step_fn` (default
`rk4_step`). Telemetry-free: `tick()` returns plain dict rows; the caller
records.

Units: seconds; state/inputs per the plant's own convention (rad/rad·s⁻¹ per
joint, N·m or N per actuator — see the concrete Plant).
"""
from __future__ import annotations

import numpy as np

from dpend.sim.integrators import rk4_step


class ControlTicker:
    """Drives one Plant through repeated control ticks.

    Construction validates that `ctrl_dt_s` is an integer multiple of
    `sim_dt_s` (the ZOH substep count `n_sub`).

    `controller` is optional if `controller_provider` (a zero-arg callable
    returning the `Controller` to consult this tick) is supplied instead:
    a live mode toggle can then swap which controller a shared ticker
    consults without rebuilding the ticker — and so without resetting the
    shared sensor/estimator/physics state (which must not reset on a mode
    switch). Callers passing a fixed `controller=` are unaffected:
    `.controller` resolves to exactly that object. See the `controller`
    property for the resolution rule and a known limitation.
    """

    def __init__(self, *, plant, sensor, estimator, controller=None, ctrl_dt_s: float,
                 sim_dt_s: float, step_fn=rk4_step, rng=None, controller_provider=None):
        ratio = ctrl_dt_s / sim_dt_s
        n_sub = round(ratio)
        if abs(ratio - n_sub) > 1e-9 * ratio or n_sub < 1:
            raise ValueError(
                f"ctrl_dt_s ({ctrl_dt_s}) must be an integer multiple of sim_dt_s ({sim_dt_s})"
            )
        if controller is None and controller_provider is None:
            raise ValueError(
                "ControlTicker requires either `controller` (a fixed Controller) or "
                "`controller_provider` (a zero-arg callable returning the Controller to "
                "consult THIS tick) — got neither"
            )
        if controller is not None and controller_provider is not None:
            raise ValueError(
                "ControlTicker accepts `controller` or `controller_provider`, not both — "
                "ambiguous which one tick() should consult"
            )
        self.plant = plant
        self.sensor = sensor
        self.estimator = estimator
        self._controller = controller
        self._controller_provider = controller_provider
        self.ctrl_dt_s = ctrl_dt_s
        self.sim_dt_s = sim_dt_s
        self.step_fn = step_fn
        self.rng = np.random.default_rng() if rng is None else rng
        self._n_sub = n_sub
        self._u_prev = np.zeros(plant.m)
        self._primed = None  # (y0, x_hat0) from reset(), consumed by the next tick()

    @property
    def controller(self):
        """The Controller consulted this access: `controller_provider()` if
        one was supplied at construction, else the fixed `controller`
        object. Resolved fresh on every access — never cached — so a live
        mode toggle takes effect on the very next `tick()`, not the next
        `reset()`.

        Known limitation: switching which controller answers `update()` does
        NOT call `.reset()` on the newly-active one — only
        `ControlTicker.reset()` does that, to whichever controller is active
        at that moment. Harmless for stateless laws; a stateful one
        (integral action, MPC warm-start) needs an explicit on-activation
        reset hook.
        """
        if self._controller_provider is not None:
            return self._controller_provider()
        return self._controller

    @property
    def n_sub(self) -> int:
        """Sim substeps per control tick (ctrl_dt_s / sim_dt_s, integer)."""
        return self._n_sub

    def reset(self, x0) -> None:
        """Prime the loop at t=0 from the true initial state x0 ∈ ℝⁿ.

        Measures and estimates once so the controller can settle on its
        first estimate before the first tick, then caches that (y0, x̂0) so
        `tick()`'s first call reuses it instead of drawing a second, different
        measurement at t=0 (determinism: the RNG must advance exactly once
        per control step, not twice on the reset tick).

        `estimator.reset(0.0, x0.copy())` seeds the estimator with the true
        state (ℝⁿ), not the raw measurement y0 — the `Estimator.reset(t0, x0)`
        contract takes a full-state prior, and matters once a partial/angle-
        only a partial/angle-only sensor lands, where y0 wouldn't even have the right shape.
        """
        x0 = np.asarray(x0, dtype=float).copy()
        self._u_prev = np.zeros(self.plant.m)
        y0 = self.sensor.measure(0.0, x0, self.rng)
        self.estimator.reset(0.0, x0.copy())
        x_hat0 = self.estimator.update(0.0, y0, self._u_prev)
        self.controller.reset(0.0, x_hat0)
        self._primed = (y0, x_hat0)

    def tick(self, t: float, x: np.ndarray, tau_ext=None) -> tuple[np.ndarray, dict]:
        """Advance one control tick starting at true state x, time t [s].

        measure → estimate → control at t (reusing the reset() priming on
        the very first call), then `n_sub` substeps of `step_fn` at
        `sim_dt_s` with u held (ZOH) and `tau_ext` (ℝ^(n//2) generalized
        force/torque, None ⇒ zeros) applied through `plant.f`.

        Seam guard: a non-None `tau_ext` must have shape `(plant.n // 2,)` —
        checked here (not left to numpy) because a wrong-shape array would
        otherwise silently broadcast across every generalized coordinate
        instead of raising. Every caller goes through this one seam, so the
        guard is enforced exactly once.

        Returns (x_next, record) with record = {x_true, x_hat, y, u,
        tau_ext, energy} — everything the caller needs to log this tick.
        """
        if self._primed is not None:
            y, x_hat = self._primed
            self._primed = None
        else:
            y = self.sensor.measure(t, x, self.rng)
            x_hat = self.estimator.update(t, y, self._u_prev)
        u = self.controller.update(t, x_hat)
        n_q = self.plant.n // 2
        if tau_ext is None:
            tau = np.zeros(n_q)
        else:
            tau = np.asarray(tau_ext, dtype=float)
            expected = (n_q,)
            if tau.shape != expected:
                raise ValueError(
                    f"tau_ext has shape {tau.shape}, expected {expected} for "
                    f"plant {self.plant.name!r} (n={self.plant.n}, n//2={n_q})"
                )

        record = {
            "x_true": x,
            "x_hat": x_hat,
            "y": y,
            "u": u,
            "tau_ext": tau,
            "energy": self.plant.energy(x),
        }

        dyn = lambda z, uu: self.plant.f(z, uu, tau)
        x_next = x
        for _ in range(self._n_sub):
            x_next = self.step_fn(dyn, x_next, u, self.sim_dt_s)

        self._u_prev = np.asarray(u, dtype=float)
        return x_next, record
