# Architecture

## Signal flow (one control tick)

```
x_true ─▶ Sensor ─▶ Estimator ─▶ Controller ─▶ ZOH ─▶ Plant (model + integrator) ─▶ x_true'
          y=h(x)+n   x̂=est(y,u)   u=ctrl(t,x̂)          ẋ = f(x, u, B)
              └───────────────── every signal tapped ─────────────────▶ Telemetry
```

The **controller always consumes an estimate `x̂`, never ground truth.** With no observer
the estimator is the identity (`x̂ = x_true`); an LQG observer swaps in a Kalman filter. So
output feedback is "free" — adding estimation is a config swap, not a rewrite. The swing-up
hand-off is the same idea: a `ModeSwitch` is just a `Controller` that delegates to
sub-controllers.

## Module map

```
dpend/
├── model/        what the system IS:  params, dynamics (M, C, g, B), linearize (A, B, ctrb/obsv)
├── sim/          how we roll it forward:  integrators (RK4, ZOH), the multi-rate simulator loop
├── sensors/      measurement models  y = h(x) + noise
├── controllers/  swappable laws:  lqr, mpc, energy_swingup, mode_switch, pole_placement
├── estimation/   state estimators:  identity (default), kalman (LQG)
├── viz/          animation + dashboard, rendered FROM telemetry
├── telemetry/    recorder + formats (npz/csv): the sim↔viz contract
├── interactive/  the live pygame UI (a second driver of the same interfaces)
├── util/         leaf helpers (angle wrapping); no internal deps
├── registry.py   component factories (plants, controllers, sensors, estimators)
├── reference.py  a setpoint source a tracking controller reads
└── config.py     Scenario — the config object one batch run is built from
```

## Dependency rules (acyclic)

- `util` — leaf; depends on nothing internal.
- `model` → `util`.
- `sensors`, `estimation`, `controllers` → `model`, `util` only — **never `sim`** (plus
  `controllers` → `reference`, a leaf, so a tracking law can read the setpoint).
- `sim` → `model`, `sensors`, `estimation`, `controllers`, `telemetry` (the orchestrator).
- `viz` → `telemetry` only.  `telemetry` → `util`.
- `interactive` (a driver, like `sim`/`viz`) → model, sensors, estimation, controllers,
  telemetry, registry, reference, `sim` (the control-tick loop only), and pygame. Nothing
  else in `dpend/` imports it.
- `config` — a leaf dataclass; `batch.py` (the composition root) wires a `Scenario` into
  concrete components via `registry`.
- `ui.py` → `interactive.cli` → `interactive.shell` → `interactive.app`.

**Key invariant:** because `controllers` / `estimation` / `sensors` never import `sim`, the
same control + estimation code would run unchanged on hardware. The simulator is just one
driver of those interfaces.

**Enforced, not aspirational:** the rules above are `import-linter` contracts
(`[tool.importlinter]` in `pyproject.toml`), checked against the real import graph by
`tests/test_architecture.py` — a violating import fails the test suite.

## Conventions (units + frames)

- **World frame:** x right, y up, planar; gravity `g = −y`, `9.81 m·s⁻²`.
- **Time:** seconds for dynamics/controllers; telemetry timestamps in nanoseconds.
- **Energy** in joules. Every physical field carries its units + frame in a docstring or
  inline comment.

**Fixed-pivot double pendulum** — `x = [θ₁, θ₂, θ̇₁, θ̇₂]`:
- `θ₁` = link-1 angle from the **upward** vertical (absolute), CCW-positive [rad];
  `θ₂` = link-2 angle relative to link 1 [rad].
- **Upright = the origin** `(0,0,0,0)`; hanging = `(π,0,0,0)`. Origin-at-upright keeps the
  LQR/MPC linearization clean.
- Actuation: generalized torque `τ = B u` [N·m]; `B` selects the actuated joints
  (fully-actuated / Acrobot / Pendubot).

**Single-pole cart-pole** — `z = [x, θ, ẋ, θ̇]`: `x` = cart position [m]; `θ` = pole angle
from the upward vertical [rad] (upright = 0, hanging = π). Actuation: force `u` [N] on the cart.

**Cart-mounted double pendulum** — `z = [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]`, force on the cart.

Every `Plant` exposes its `upright`/`hanging` equilibria and default LQR weights; the
cart-pole additionally implements an `EnergyShapingCapable` protocol (the energy primitives
the swing-up controller consumes), so a swing-up method picks its plant by capability, not
by name.

## Where does X live?

| To find / change… | Go to |
|---|---|
| Fixed-pivot equations of motion | `model/dynamics.py` |
| Cart / cart-pole dynamics | `model/cart_dynamics.py`, `model/cart_pole_dynamics.py` |
| Plant protocol + factories + registry | `model/plant.py`, `dpend/registry.py` |
| Linearization + controllability/observability | `model/*_linearize.py` |
| Integration (RK4) + zero-order hold | `sim/integrators.py` |
| The time-stepping loop + multi-rate clocks | `sim/simulator.py`, `sim/ticker.py` |
| LQR gain — hand-rolled CARE (eigenvector seed + Newton–Kleinman) | `controllers/riccati.py` |
| DARE solve + `dlqr_gain` (MPC terminal cost) | `controllers/riccati.py` |
| Linear MPC — condensed QP, osqp, warm start | `controllers/mpc.py` |
| Exact ZOH discretization | `controllers/discretize.py` |
| Energy-shaping swing-up | `controllers/energy_swingup.py` |
| Hybrid swing-up → catch mode switch | `controllers/mode_switch.py` |
| Interactive UI (pygame live loop) | `interactive/` |
| Plots / animation | `viz/` |
| What gets logged | `telemetry/recorder.py` |
| CLI: batch scenarios / interactive UI | `batch.py` / `ui.py` (via `./run`) |
| A runnable experiment | `scenarios/<name>.py` |

`tests/` mirrors this package layout (`tests/model/`, `tests/controllers/`,
`tests/interactive/`, …), so the tests for any module are easy to find.

## Notes on rigor

- **Units & frames everywhere** — every physical quantity carries SI units + a frame.
- **Verify, don't assert** — dynamics checked by energy conservation, equilibria, and
  finite-difference-vs-analytic linearization; LQR gains cross-checked against `scipy`; MPC
  checked against its unconstrained LQR limit.
- **Design notes** — each nontrivial controller has a short note (decision, alternatives,
  why, failure modes) under [`design-notes/`](design-notes/).
