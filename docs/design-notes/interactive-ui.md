# Design note — `./run` dispatcher + in-UI launcher menu & live controller swap

**Date:** 2026-07-10. **Status:** shipped (suite 239 → 256; real-display self-drive passed).
This is a **driver-layer** feature (no control/estimation/plant code changed); the design notes usually
cover controllers/estimators, but this nontrivial UI work earned one too.

## What shipped

1. **`./run`** (repo-root bash dispatcher): one front door. Bare `./run` opens the UI; `./run
   ui|batch|test|build|shell|help` wrap the `docker compose` incantations (writes `.env` UID/GID,
   grants X11, auto-builds if the image is missing). `RUN_DRY_RUN=1` is the test seam (print, don't exec).
2. **`run.py` → `batch.py`** rename (the launcher had to be the file `run`).
3. **`interactive/widgets.py`** — `Button`/`ButtonGroup` (absolute-rect geometry + draw, pygame-only).
4. **`interactive/menu.py`** — `MenuScreen` (plant · controller · start groups + Start ▶) and the shared
   `compatible_controllers(plant)` helper.
5. **`App` surface refactor** — `App(plant, cfg, screen, …)` draws into a caller-owned Surface (no longer
   owns `pygame.init/set_mode/quit`); gains `set_controller(key)` (live swap) and `_arm_supervisor()`.
6. **`interactive/shell.py`** — `Shell` owns the pygame window + main loop, switching between `MenuScreen`
   and a running `App`. `run_app` (the `ui.py` entry) opens the menu by default; flags boot a sim directly.

## Key decisions & alternatives

- **Launcher name `./run`, not `./dpend`.** The originally-proposed `dpend` file collides with the
  `dpend/` package directory (a file and a directory can't share a name). `./run` is unambiguous once the
  old `run.py` is renamed — hence `batch.py`. *Alternative:* keep `run.py`, name the launcher something
  else (`go`, `x`) — rejected as less discoverable than the verb "run".
- **Bare `./run` = the launcher MENU.** `_parse_ui_args` defaults `--plant/--controller/
  --start` to `None`; `_selection_from_args` returns `None` (→ menu) iff all three are `None`, else a full
  selection (missing fields → cart/zero/upright) for a scripted direct launch. *Alternative:* bare = a
  default cart/zero sim (the old behavior) — rejected: the whole point was "pick by clicking".
- **`Shell` owns the pygame lifecycle; `App` became surface-based.** A two-screen app (menu ↔ sim) needs
  one window owner; making `App` take a `screen` also made it construct headlessly and step deterministically
  (the whole interactive test suite builds `App`s under the dummy driver). *Alternative:* `App` keeps
  owning pygame and the menu is a separate window — rejected: two windows + an awkward destroy/recreate
  hand-off on every menu trip.
- **Compatibility by try-build.** `compatible_controllers(plant)` = "the registry factory constructs
  without raising", computed once per plant change. So `swingup` self-excludes on non-`EnergyShapingCapable`
  plants, `lqr`/`mpc` would self-exclude on any future non-controllable plant, and a newly-registered
  controller needs no menu edit. *Alternative:* a hard-coded per-plant table — rejected as brittle and
  duplicative of the factories' own gating.
- **Live swap keeps plant state** (`set_controller`): the point is to watch different laws handle the
  *same* state (lqr→mpc while balanced; →swingup while hanging). It's **mode-orthogonal** — it does not
  force CONTROLLER mode (in CONTROLLER it live-swaps the running law; in MANUAL it stages the law for the
  next `M`). It re-arms the RoA supervisor via the shared `_arm_supervisor()` (IS_HYBRID-gated, so →swingup
  doesn't arm the small-tip linear supervisor; →lqr/mpc does). *[Superseded by the follow-up below:
  there is no MANUAL/CONTROLLER toggle or `M` key anymore — selecting a controller runs it; "None" is manual.]*
- **`_build_app` retired.** `run_app` now goes through the `Shell`; keeping `_build_app`'s own argparse
  alongside the new `_parse_ui_args` would be two parsers for one CLI. Its test coverage migrated to
  parser tests + Shell tests. **`App.run()` was retained** — it's the isolated single-App exit-path loop
  that two e2e tests drive (real-queue ESC → `close()` → npz); `Shell.run()`'s multi-state loop can't
  isolate that, so it isn't dead duplication.

## Failure modes (the things that bite)

1. **Wound-angle strip-swap (surfaced live by the self-drive).** The Åström–Furuta swing-up winds `θ` past
   ±π (to ~2π at "upright"). `ModeSwitch` handles this by wrapping `θ` to ±π of the catch equilibrium before
   calling its LQR/MPC child (`_wrap_for_catch`). But a **strip live-swap to a *naked* `lqr`/`mpc`** feeds
   the raw wound angle straight in → `V(x)=xᵀPx` is enormous → the RoA supervisor disengages to MANUAL and
   the pole falls (observed: `cos θ` fell to −0.951, notice "LQR disengaged: left the basin"). The *clean*
   live-swap case (lqr↔mpc while genuinely balanced near `θ≈0`) is fine and unit-tested. **RESOLVED
   (see the follow-up below):** the UI no longer builds a *naked* lqr/mpc on the cart-pole — selecting
   lqr/mpc there builds the swing-up+catch `ModeSwitch`, which wraps `θ` internally, so there is no
   naked-catch path to feed a wound angle. (Batch scenarios can still build a naked catch; wrapping *those*
   is a separate control-code change, not done.)
2. **Dispatcher real paths untestable in-container.** The pytest suite runs *inside* the dev container,
   which can't launch Docker, so `test_dispatch.py` only exercises the `RUN_DRY_RUN` arg-routing. The actual
   `./run build` / bare `./run` / `./run shell` (real `docker compose`, `.env` write, `xhost` grant) need a
   one-time host smoke — done manually; a human should re-run after any dispatcher edit.
3. **Menu-return finalizes telemetry.** Returning to the menu calls `App.close()`, which finalizes the
   sim's telemetry and in production writes `artifacts/live_<stamp>/telemetry.npz`. A long menu-hopping
   session accumulates many `live_*` dirs. Tests redirect this via `Shell`'s `out_dir` seam.

### Accepted cosmetic gaps (candidate follow-ups)

- The **CLI bypass doesn't validate compatibility**: `./run ui --controller swingup` with the default
  `--plant cart` fails loud via `App.__init__ → ValueError → SystemExit` (swingup isn't
  `EnergyShapingCapable` on the plain cart). The *menu* filters incompatible controllers; the flag path
  promises no validation and fails loud rather than silently — acceptable.
- After a **direct `--flag` launch**, the in-sim Menu button returns to a `MenuScreen` showing its *default*
  selections (cart/…), not the flag-launched plant/controller/start — the Shell builds the menu with
  defaults, independent of the launch selection. Cosmetic; nothing requires menu↔CLI sync. A tidy
  follow-up would seed `MenuScreen` from the launch selection.

## DAG (unchanged, verified whole-set)

`interactive` stays a leaf driver: `widgets`→pygame only; `menu`→pygame + registry + model.plant; `shell`
→ app + menu + config + model.plant; `app`→(model, sim.ticker, registry, telemetry, reference, pygame) +
lazily `shell`/`menu`. Nothing outside `interactive` imports it. The one cycle risk (`app` ↔ `shell`) is
broken by importing `Shell` lazily inside `run_app`.

## Follow-up (2026-07-10) — selection *is* the mode; robust lqr/mpc

Two behavior changes after the first live use of the UI.

**The selected controller IS the mode (no `M` key).** The independent MANUAL/CONTROLLER toggle is
gone. Whatever controller is selected runs, always; `App.mode` is a **derived read-only property** — MANUAL
iff the `zero` ("None") controller is selected (you drag the cart), CONTROLLER otherwise. `_current_controller`
always returns `_active_controller`. The RoA safety supervisor's disengage (a naked lqr/mpc on the plain
cart leaving its basin) now **drops the selection to `"zero"`** (None) instead of flipping a mode flag — the
strip visibly falls back to None. *Why:* the toggle was a hidden second state that could desync from the
selection ("controller: lqr, mode: MANUAL" ran nothing — the exact confusion reported). Governing mode by
selection removes that whole failure class.

**lqr/mpc are robust on a swing-up-capable plant.** In the UI, selecting lqr/mpc on the cart-pole now
builds `ModeSwitch(energy swing-up + that catch law)` (`app._ui_controller`, gated on `isinstance(plant,
EnergyShapingCapable)` — never `plant.name`), so it swings up from hanging then catches, or catches
immediately if already balanced. That is what "select LQR, it figures out the state" means, and it
**resolves the wound-angle failure mode #1** above: the ModeSwitch wraps `θ` for its catch, and the UI no
longer exposes a naked catch on the cart-pole. The standalone `swingup` menu entry is dropped (subsumed by
lqr/mpc). Naked lqr/mpc remain on the plain cart / fixed pivot (not EnergyShapingCapable) and in batch — so
the App-level RoA supervisor still guards *those*.

**Labels.** The menu shows friendly plant names (`Cart 2 Pendulum` / `Cart 1 Pendulum` / `2 Pendulum`) and
the `zero` controller reads **"None"**; button *values* stay the registry keys, so the CLI/config/Shell
plumbing is untouched (`menu.plant_label`/`menu.controller_label`, shared with the in-sim strip).

**Verified:** suite 256 → 262; real-display self-drive on x11 drove menu → select lqr (Cart 1 Pendulum,
hanging) → Start → swung up `cos θ −1.000→1.000` with **no `M` pressed** → strip None→MANUAL, mpc→CONTROLLER.

**New thing to keep in mind:** with the App supervisor no longer arming for the cart-pole's (hybrid)
lqr/mpc, a hard disturbance there is handled by the ModeSwitch's own SWINGING re-entry (re-pump, re-catch)
rather than a disengage-to-None — different, generally better, but worth remembering when reasoning about
"what stops a blow-up" on the cart-pole vs the plain cart.
