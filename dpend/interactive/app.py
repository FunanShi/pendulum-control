"""`App` — the live pygame driver (the CLI entry, `run_app`, lives in cli.py).

Wires the same core the batch driver uses (`sim.ticker.ControlTicker` via
`interactive.loop.RealtimeLoop`) to a pygame event pump + renderer, building
components through the same `dpend.registry` factories as `batch.py` — the
live session is the verified physics/control code plus a human at the
mouse/keyboard, never a second implementation.

Controller selection IS the mode (no toggle key): one `ControlTicker` is
built once with a `controller_provider` callable that resolves the currently
selected controller fresh every tick. Selecting "None" (the zero controller)
applies zero force — manual drag; a real law controls. `App.mode` is a
derived read-only property of that selection. Swapping (`set_controller`)
leaves the shared sensor/estimator/physics state untouched; drag/keys stay
live for any selection, as a disturbance.

Force composition: the hand force (mouse-spring + keys, computed by
`interactive.input.hand_force`, already clamped to ±cfg.f_max) plus an
optional scenario disturbance (`scenario_params["disturbance"]` — scripted
physics, unclamped). The sum becomes `tau_ext = [F, 0, ...]`: the cart/base
channel only.

DAG: this file and `render.py` are the only pygame importers in `dpend/`;
nothing in `dpend` imports `interactive` (a driver/leaf consumer, like `sim`).
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from dpend.interactive.input import hand_force
from dpend.interactive.loop import RealtimeLoop
from dpend.interactive.menu import compatible_controllers, controller_label
from dpend.interactive.render import WorldToScreen, cart_rect_px, draw_scene
from dpend.interactive.ui_config import InteractiveConfig
from dpend.interactive.widgets import Button, ButtonGroup
from dpend.model.plant import EnergyShapingCapable
from dpend.reference import ReferenceSource
from dpend.registry import CONTROLLERS, ESTIMATORS, SENSORS
from dpend.sim.ticker import ControlTicker
from dpend.telemetry.formats import save_npz
from dpend.telemetry.recorder import Recorder

# Multi-rate clocks: the repo's standard rates (Scenario's own defaults).
# App takes a built plant + scenario_params, not a full Scenario, so the
# pair is fixed here rather than exposed as a constructor param.
DEFAULT_SIM_DT_S = 1e-3    # plant integration step (1 kHz)
DEFAULT_CTRL_DT_S = 5e-3   # controller/estimator step (200 Hz)

_MANUAL, _CONTROLLER = "MANUAL", "CONTROLLER"

# RoA (region-of-attraction) supervisor calibration: 0.30 rad is the
# tightest measured-convergent theta1 tip of the cart-plant LQR basin — the
# conservative choice for a safety trigger (measurements and the live
# failure that motivated it: docs/design-notes/lqr-riccati.md).
ROA_CALIBRATION_TIP_RAD = 0.30  # rad, theta1 — cart-plant state index 1
                                # ([x, th1, th2, xdot, th1dot, th2dot], n=6)


def _ui_controller(plant, key: str, params: dict):
    """UI controller-building policy. On a swing-up-capable plant, a linear
    catch law (lqr/mpc) is wrapped into a swing-up + that-catch `ModeSwitch`,
    so selecting it is robust from any state: swings up from hanging, catches
    immediately if already balanced, and wraps θ for the catch so a wound-up
    angle never trips a naked catch. Naked lqr/mpc remain on non-swing-up
    plants and in batch; anything else builds straight from the registry."""
    if key in ("lqr", "mpc") and isinstance(plant, EnergyShapingCapable):
        return CONTROLLERS["swingup"](plant, {**params, "catch": key})
    return CONTROLLERS[key](plant, params)


class App:
    """One live session: renders into a pygame `screen` Surface owned and
    torn down by the caller, and owns the ticker/loop, the live controller
    selection, and the telemetry recorder.

    `now_fn` (keyword-only, default `time.perf_counter`) is a test seam:
    injecting the clock lets tests pin an exact tick count per `step_once()`
    call instead of depending on real wall-clock speed.
    """

    def __init__(self, plant, cfg: InteractiveConfig, screen, controller_name: str = "zero",
                scenario_params: dict | None = None, start: str = "upright",
                *, now_fn=time.perf_counter):
        self.plant = plant
        self.cfg = cfg
        self.controller_name = controller_name
        self.scenario_params = dict(scenario_params) if scenario_params else {}

        self.screen = screen
        pygame.display.set_caption("dpend — live cart")   # harmless; the caller has already set_mode()
        self.clock = pygame.time.Clock()
        self.fps_measured = 0.0
        self.w2s = WorldToScreen(plant, cfg.window_px)

        # Reference target: an App-owned ReferenceSource the UI mutates in
        # place (right-click), handed to the controller factory too so a
        # tracking controller closes over the same instance.
        self.reference = ReferenceSource(0.0)
        self.scenario_params.setdefault("reference", self.reference)
        # Scripted disturbance: optional callable under
        # scenario_params["disturbance"] (the live analog of
        # Scenario.disturbance).
        self.disturbance = self.scenario_params.get("disturbance")

        # u_max default (UI policy): unsaturated LQR out-muscles the 500 N/m
        # end-stops once outside the linear basin. 150 N is >3x the ~45 N
        # measured at the basin edge (normal recovery never clips) and
        # bounds static intrusion past the rail to 150/500 = 0.3 m. Applied
        # for any controller name (uniform policy): one dict entry, harmless
        # for a factory that ignores it.
        self.scenario_params.setdefault("u_max", 150.0)

        # sensor/estimator pinned to "perfect"/"identity", the only entries
        # that exist today.
        sensor = SENSORS["perfect"](plant, {})
        estimator = ESTIMATORS["identity"](plant, {})
        try:
            self._active_controller = _ui_controller(plant, controller_name, self.scenario_params)
        except KeyError:
            raise ValueError(
                f"controller {controller_name!r} not implemented yet; "
                f"available: {sorted(CONTROLLERS)}"
            ) from None

        # RoA supervisor: arms only when the active controller exposes a
        # Lyapunov/cost-to-go P (LQR/MPC) and does not declare IS_HYBRID.
        # V(x) = xᵀPx; z_tip is the measured-convergent basin-edge tip
        # (ROA_CALIBRATION_TIP_RAD) on theta1 alone. A UI safeguard, not a
        # certified region of attraction — deliberately conservative (see
        # docs/design-notes/lqr-riccati.md); fires in `step_once`.
        # Hybrid (ModeSwitch) controllers opt out via the explicit IS_HYBRID
        # class flag (default False): they borrow the catch child's P, and
        # this small-tip supervisor would trip on the first tick from
        # hanging (V(hanging) ≫ V_lim), defeating the swing-up — ModeSwitch
        # is a supervisor in its own right; SWINGING is supposed to roam far
        # from upright.
        self._arm_supervisor()

        ticker = ControlTicker(
            plant=plant, sensor=sensor, estimator=estimator,
            controller_provider=self._current_controller,
            ctrl_dt_s=DEFAULT_CTRL_DT_S, sim_dt_s=DEFAULT_SIM_DT_S,
        )
        # start: "upright" (the origin) or "hanging" (the swing-up demo's
        # start). plant.upright/.hanging already return fresh copies, but
        # .copy() again is explicit: reset paths consult self.x0 by
        # reference, so it must be App-owned.
        self.x0 = plant.hanging.copy() if start == "hanging" else plant.upright.copy()
        self.loop = RealtimeLoop(ticker=ticker, x0=self.x0, ctrl_dt_s=DEFAULT_CTRL_DT_S,
                                 max_substeps_per_frame=cfg.max_substeps_per_frame, now_fn=now_fn)

        self.recorder = Recorder()

        # Hand-input state (mouse/keys) — plain data, no pygame types; the
        # force law itself lives in `interactive.input.hand_force`.
        self.dragging = False
        self._keys_held: set[str] = set()
        self._mouse_world_x = 0.0
        self._last_force_n = 0.0

        # Tip-trace history: the last `maxlen` tip positions, world frame
        # [m], one appended per frame that advanced physics; lives here so
        # `draw_scene` stays stateless. maxlen=60 ≈ 1 s at the 60 fps
        # target; a bounded deque so a long session can't grow it.
        self.tip_trace: deque = deque(maxlen=60)

        self.running = True
        self.want_menu = False

        # In-sim control strip along the bottom edge: a radio of the plant's
        # compatible controllers (live-swap via set_controller) + a Menu
        # button (sets want_menu; the Shell returns to the launcher). Built
        # once here — button rects readable before the first draw, and the
        # factory probe never runs per frame.
        _STRIP_W, _STRIP_H, _STRIP_GAP = 84, 28, 6
        strip_y = cfg.window_px[1] - _STRIP_H - 10
        strip_buttons = [
            Button(controller_label(k), (10 + i * (_STRIP_W + _STRIP_GAP), strip_y, _STRIP_W, _STRIP_H),
                  value=k)
            for i, k in enumerate(self.compatible_controllers())
        ]
        self._controller_strip = ButtonGroup("controller", strip_buttons, selected_value=self.controller_name)
        self._strip_origin = (10, strip_y - 20)   # anchors only the group's title text
        self._menu_button = Button("Menu", (cfg.window_px[0] - _STRIP_W - 10, strip_y, _STRIP_W, _STRIP_H),
                                   value="menu")

    # -- RoA supervisor + controller swap ---------------------------------

    def _arm_supervisor(self) -> None:
        """Set self.notice / self._v_limit for the current active controller —
        armed (V_lim finite) iff it exposes `.P` and does not declare
        IS_HYBRID (see the supervisor note at __init__'s call site). Single
        source of truth for __init__ and set_controller."""
        self.notice = ""
        self._v_limit = None
        if hasattr(self._active_controller, "P") and not getattr(
                self._active_controller, "IS_HYBRID", False):
            z_tip = np.zeros(self.plant.n)
            z_tip[1] = ROA_CALIBRATION_TIP_RAD
            self._v_limit = float(z_tip @ self._active_controller.P @ z_tip)

    def set_controller(self, key: str) -> None:
        """Live-swap the active controller, keeping the current plant state
        (the point is to watch different laws handle the same state). Build
        via the registry with the App's scenario_params, reset at the current
        (t, x), re-arm the RoA supervisor (IS_HYBRID-gated: a hybrid
        ModeSwitch never arms it; a naked lqr/mpc does)."""
        self._active_controller = _ui_controller(self.plant, key, self.scenario_params)
        self.controller_name = key
        self._active_controller.reset(self.loop.t, self.loop.x)
        self._arm_supervisor()

    def compatible_controllers(self) -> list[str]:
        return compatible_controllers(self.plant)

    # -- mode machine ---------------------------------------------------

    @property
    def mode(self) -> str:
        """Derived, read-only — the selected controller is the mode: MANUAL
        when "None" (the zero controller) is selected, CONTROLLER when a real
        law runs. Change it via set_controller."""
        return _MANUAL if self.controller_name == "zero" else _CONTROLLER

    def _current_controller(self):
        """`controller_provider` handed to `ControlTicker`: the currently
        selected controller, resolved fresh every tick."""
        return self._active_controller

    def _v(self, x) -> float:
        """V(x) = xᵀPx — the active controller's Lyapunov/cost-to-go, at the
        current true state. Callers must guard on `self._v_limit is not None`
        first (only meaningful when the active controller exposes `P`)."""
        x = np.asarray(x, dtype=float)
        return float(x @ self._active_controller.P @ x)

    @property
    def key_dir(self) -> int:
        """-1/0/+1 from the held-key set — a set (not two booleans) so
        LEFT+RIGHT both held cancels to 0 and a stale KEYUP for a key that
        was never "down" is a no-op."""
        left = "LEFT" in self._keys_held
        right = "RIGHT" in self._keys_held
        if left and not right:
            return -1
        if right and not left:
            return 1
        return 0

    # -- event handling ---------------------------------------------------

    def _handle_event(self, ev) -> None:
        if ev.type == pygame.QUIT:
            self.running = False
        elif ev.type == pygame.WINDOWFOCUSLOST:
            # Alt-tab away while a key is physically held: the matching
            # KEYUP fires on whichever window has focus afterward (if ever),
            # so without this the key would stay "held" — and its force
            # applied — forever. Mouse-drag has its own release path
            # (MOUSEBUTTONUP) and is left alone here.
            self._keys_held.clear()
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self.running = False
            elif ev.key == pygame.K_r:
                self.loop.reset(self.x0)
                self.tip_trace.clear()  # fresh session: a stale trace would draw
                                        # a bogus teleport line from the old tip
                self.dragging = False   # stale grab: R mid-drag must not keep
                                        # pulling the just-reset cart toward a
                                        # pre-reset mouse position
                self._mouse_world_x = float(self.x0[0])  # clear the stale target too
                self.notice = ""  # fresh session: clear any stale supervisor notice
            elif ev.key == pygame.K_LEFT:
                self._keys_held.add("LEFT")
            elif ev.key == pygame.K_RIGHT:
                self._keys_held.add("RIGHT")
        elif ev.type == pygame.KEYUP:
            if ev.key == pygame.K_LEFT:
                self._keys_held.discard("LEFT")
            elif ev.key == pygame.K_RIGHT:
                self._keys_held.discard("RIGHT")
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:
                # In-sim controls take precedence over the cart grab: a click
                # on the strip or Menu button must not also begin a drag.
                if self._menu_button.hit(ev.pos):
                    self.want_menu = True
                elif self._controller_strip.click(ev.pos):
                    self.set_controller(self._controller_strip.selected_value)
                else:
                    rect = cart_rect_px(self.plant, self.loop.x, self.w2s)
                    if rect is not None and rect.collidepoint(ev.pos):
                        self.dragging = True
                        self._mouse_world_x = self.w2s.to_world(ev.pos)[0]
            elif ev.button == 3:  # right-click: set the reference target
                world_x = self.w2s.to_world(ev.pos)[0]
                if self.plant.rail is not None:
                    world_x = float(np.clip(world_x, -self.plant.rail, self.plant.rail))
                self.reference.set_target(world_x)
        elif ev.type == pygame.MOUSEBUTTONUP:
            if ev.button == 1:
                self.dragging = False
        elif ev.type == pygame.MOUSEMOTION:
            if self.dragging:
                self._mouse_world_x = self.w2s.to_world(ev.pos)[0]

    # -- per-frame step (the test seam) ------------------------------------

    def step_once(self, synthetic_events=None) -> list[dict]:
        """Advance exactly one rendered frame: pump events (`synthetic_events`
        if given — the test seam — else pygame's real queue), compute this
        frame's hand force, advance the shared physics/control loop by
        however many control ticks the wall clock says elapsed, record each
        tick, draw, pace to `cfg.fps`. Returns the tick records produced
        this frame (each `Recorder.append(**record)`-shaped; may be empty).
        """
        events = pygame.event.get() if synthetic_events is None else synthetic_events
        for ev in events:
            self._handle_event(ev)

        if not self.running:
            return []

        x_cart = float(self.loop.x[0])
        xdot_cart = float(self.loop.x[self.plant.n // 2])
        F_hand = hand_force(dragging=self.dragging, x_mouse=self._mouse_world_x,
                            x_cart=x_cart, xdot_cart=xdot_cart,
                            key_dir=self.key_dir, cfg=self.cfg)
        self._last_force_n = F_hand

        def tau_fn(t, x):
            tau = np.zeros(self.plant.n // 2)
            tau[0] = F_hand  # clamped hand force: cart/base channel only
            if self.disturbance is not None:
                tau = tau + np.asarray(self.disturbance(t, x), dtype=float)  # unclamped, summed
            return tau

        records = self.loop.advance(tau_fn)
        for r in records:
            self.recorder.append(**r)
        if records:  # physics advanced: log the new tip position (world [m]);
            #          a stalled frame appends nothing — duplicates would
            #          just burn trace-history slots
            _, pts = self.plant.fk(self.loop.x)
            self.tip_trace.append((float(pts[-1][0]), float(pts[-1][1])))

        # RoA supervisor: only when the active controller exposes a V (see
        # __init__). Fires on a V-sublevel exit (V(x) > V_lim), not entry —
        # one-sided by design.
        if self._v_limit is not None:
            if self._v(self.loop.x) > self._v_limit:
                disengaged = self.controller_name
                self.set_controller("zero")   # drop the selection to None (manual); this also
                                              # clears _v_limit + notice via _arm_supervisor…
                self.notice = (f"{disengaged.upper()} disengaged: left the "
                              "basin (V > V_lim) — R to reset")   # …so set the notice AFTER

        self._draw()
        self.clock.tick(self.cfg.fps)
        self.fps_measured = self.clock.get_fps()
        return records

    def _draw(self) -> None:
        draw_scene(self.screen, self.plant, self.loop.x, self.hud(), self.w2s)
        self._draw_controls()
        pygame.display.flip()

    def _draw_controls(self) -> None:
        """The in-sim control strip (controller radio + Menu button), drawn on top
        of the scene each frame. Font is created here, not cached (a pygame Font
        doesn't survive a quit()->init() cycle — see render._font)."""
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.Font(None, 20)
        self._controller_strip.selected_value = self.controller_name  # highlight follows the source of truth
        self._controller_strip.draw(self.screen, font, self._strip_origin)
        self._menu_button.draw(self.screen, font, selected=False)

    def hud(self) -> dict:
        """The `hud` dict `render.draw_scene` consumes — also directly
        useful to tests.

        `notice`: one-line supervisor/session message, "" when nothing to
        show. `v`/`v_limit`: the RoA supervisor's V(x)/V_lim, or `None` when
        the active controller exposes no P — always freshly evaluated at the
        current true state, never stale. `swing_mode`: the active
        controller's own `.mode` (`ModeSwitch`'s "SWINGING"/"CATCHING"), or
        `None` for controllers without one. A separate key from `mode` —
        `mode` is the UI's own MANUAL/CONTROLLER interaction state,
        `swing_mode` the swing-up child's energy-shaping-vs-catch state; the
        two are orthogonal and render on distinct HUD lines.
        """
        return {
            "mode": self.mode,
            "controller": self.controller_name,
            "t": self.loop.t,
            "energy": float(self.plant.energy(self.loop.x)),
            "fps": self.fps_measured,
            "dropped_s": self.loop.dropped_s,
            "force": self._last_force_n,
            "swing_mode": getattr(self._active_controller, "mode", None),
            "f_max": self.cfg.f_max,
            "target": self.reference.r(self.loop.t),
            "tip_trace": self.tip_trace,  # bounded deque of world (x,y) [m]
            "notice": self.notice,
            "v": self._v(self.loop.x) if self._v_limit is not None else None,
            "v_limit": self._v_limit,
        }

    # -- lifecycle ----------------------------------------------------------

    def run(self, out_dir: "Path | None" = None) -> None:
        """Blocking real-time loop: real pygame events, real wall clock
        (unless `now_fn` was overridden), until ESC/QUIT/window-close, then
        `close()`. `out_dir` is forwarded to `close()` — the same test seam
        one level up; `run_app` never passes it. `step_once()` is this
        loop's entire body and is tested directly with a scripted clock;
        real wall-clock/event-queue pacing stays manual-only verification.
        """
        while self.running:
            self.step_once()
        self.close(out_dir)

    def close(self, out_dir: "Path | None" = None) -> "Path | None":
        """Finalize telemetry -> `artifacts/live_<runstamp>/telemetry.npz`
        (runstamp = `time.strftime("%Y%m%d-%H%M%S")`), print a ready-to-run
        dashboard command. `out_dir` overrides the default path (a test seam
        so tests don't write into the real `artifacts/` tree; `run()` never
        passes it). Requires >=1 recorded tick (`Recorder.finalize()`'s
        contract) — an empty session skips finalize/save instead of hitting
        that ValueError: prints a one-line notice, creates no artifacts dir,
        returns None. Pygame teardown is the caller's job — it owns the
        `screen` it handed to `App.__init__`.
        """
        if not self.recorder._rows:
            print("session ended: no telemetry recorded (0 control ticks)")
            return None
        tel = self.recorder.finalize()
        if out_dir is None:
            runstamp = time.strftime("%Y%m%d-%H%M%S")
            out_dir = Path("artifacts") / f"live_{runstamp}"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "telemetry.npz"
        save_npz(tel, npz_path)

        state_labels = tuple(self.plant.state_labels)
        input_labels = tuple(self.plant.input_labels)
        cmd = (
            "python -c \"from dpend.telemetry.formats import load_npz; "
            "from dpend.viz.dashboard import dashboard; "
            f"dashboard(load_npz({str(npz_path)!r}), state_labels={state_labels!r}, "
            f"input_labels={input_labels!r})\""
        )
        print(f"session ended: {npz_path}")
        print(f"dashboard: {cmd}")
        return out_dir
