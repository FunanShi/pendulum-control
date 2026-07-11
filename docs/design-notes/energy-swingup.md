# Design note — Energy-shaping swing-up + hybrid mode-switch catch

*The decision, the alternatives, why, and the failure modes —
sibling to [`lqr-riccati.md`](lqr-riccati.md) and
[`linear-mpc.md`](linear-mpc.md). Every number below is
measured — from the tests and instrumented runs cited inline — never invented.*

## The decision, in one paragraph

`dpend/controllers/energy_swingup.py` (`EnergySwingUp`) drives the single-pole
cart-pole plant from hanging to upright by **energy shaping**: pump the pole
subsystem's own mechanical energy `E = ½Jθ̇² + m_p g l·cosθ` toward its
upright value `E_up = m_p g l`, realized exactly via **collocated partial
feedback linearization** (command the actuated cart's acceleration directly;
invert the passive θ-row for the force). `dpend/controllers/mode_switch.py`
(`ModeSwitch`) is a **hybrid supervisor**: it runs the energy pump
(`SWINGING`) until a Lyapunov/cost-to-go sublevel test `V(x̂) = eᵀPe < c_catch`
says the state has entered the existing LQR/MPC catch's region of attraction,
then hands off (`CATCHING`) — with hysteresis (`c_release > c_catch`) against
chatter. `dpend/registry.py`'s `swingup_factory` wires an
`EnergyShapingCapable` plant + a chosen catch controller into one `ModeSwitch`,
registered as `CONTROLLERS["swingup"]`. The alternative for swing-up —
**trajectory optimization** (offline iLQR/DDP or direct collocation, tracked
online by TVLQR) — is the honest, general method, but is left as future work
here: energy shaping is closed-form, needs no offline solve, and is provably
global on the **single** pole (see the S¹-topology section below) — but, as that
same section explains, it does **not** generalize to the double pendulum, which
is exactly why extending swing-up to the double would reach for trajectory
optimization instead.

---

## Energy shaping: the `Ė` derivation, and why it drives `E → E_up`

**Setup.** The pole-subsystem energy about its own (fixed) pivot —
`cart_pole_dynamics.pendulum_energy` — is `E(θ,θ̇) = ½Jθ̇² + m_p g l·cosθ`,
`J = m_p l² + I` (parallel-axis theorem; `I` is the pole's own inertia about
its COM — see the "honest scope" section below for the `I` modeling choice).
This is **not** the plant's total mechanical energy (that also counts the
cart's kinetic energy and any end-stop spring PE) — it is the pole-only
quantity the swing-up law pumps toward `E_up = m_p g l` (`pendulum_energy` at
upright rest, per `cart_pole_dynamics.energy_upright`).

**Differentiate along the closed loop.** `Ė = Jθ̇θ̈ − m_p g l·sinθ·θ̇ =
θ̇(Jθ̈ − m_p g l·sinθ)`. Now substitute the ACTUAL `θ̈` produced when the cart's
acceleration is commanded to exactly equal `a_cmd` (frictionless, in-rail, no
`tau_ext`/`tau_stop` — the same exactness `accel_to_force`'s own test
verifies to ~1e-9). From the θ-row of `Mq̈ + Cq̇ + g = τ` (unactuated row,
`C`'s second row is identically zero — `Cq̇ = [b·sinθ·θ̇², 0]ᵀ`, `b = m_p l`):

```
−b·cosθ·a_cmd + J·θ̈ − m_p g l·sinθ = 0     ⟹     θ̈ = (b·cosθ·a_cmd + m_p g l·sinθ) / J
```

Substituting back: `Jθ̈ − m_p g l·sinθ = b·cosθ·a_cmd` (the `m_p g l·sinθ`
terms cancel exactly), so

```
Ė = b·θ̇·cosθ·a_cmd
```

— exactly the identity `EnergySwingUp`'s own docstring states, now re-derived
from the closed forms rather than asserted. This identity holds for
**whatever** `a_cmd` actually is — it comes purely from correctly inverting
the passive θ-row, independent of where `a_cmd` came from. Substituting the
**pump term alone**, `a_cmd = k_E(E_up − E)·θ̇·cosθ`:

```
Ė = k_E·b·(E_up − E)·θ̇²·cos²θ
```

Let `Ẽ := E − E_up` (energy error). Then `Ẽ̇ = Ė = −k_E·b·Ẽ·θ̇²·cos²θ` — the
identity stated above. **Precisely what this buys** (worth stating exactly,
not loosely): `θ̇²cos²θ ≥ 0` always, so `Ẽ̇` has the **opposite sign of `Ẽ`**
whenever `θ̇²cos²θ > 0` — i.e. `Ẽ̇ ≤ 0` when `Ẽ ≥ 0` and `Ẽ̇ ≥ 0` when `Ẽ ≤ 0`.
That is *not* the same claim as "`Ẽ̇ ≤ 0` unconditionally" (which is false —
`Ẽ` can be negative, e.g. right after release from hanging where
`E(hanging) = −m_p g l < E_up`). The clean, unconditional Lyapunov statement
uses `V := ½Ẽ²`:

```
V̇ = Ẽ·Ẽ̇ = −k_E·b·Ẽ²·θ̇²·cos²θ ≤ 0   for ALL (θ, θ̇), unconditionally
```

(product of three non-negative terms — `Ẽ²`, `θ̇²`, `cos²θ` — times positive
constants `k_E, b > 0`). `|Ẽ|` is monotonically non-increasing regardless of
which side of `E_up` the state starts on — the "homoclinic orbit is
attractive" claim: the level set `{E = E_up}` (which, for a pendulum, is the
separatrix through the upright equilibrium — the boundary between oscillating
and rotating motion) becomes an attracting set under this law.

**A precise caveat on the SHIPPED law** (checked directly, not assumed): the above is exact for the **pump term in isolation**. The shipped
`a_cmd` additionally adds the cart-centering correction, `a_cmd = a_pump −
k_x·x − k_d·ẋ`; since `Ė = b·θ̇·cosθ·a_cmd` holds for *whatever* `a_cmd` is,
the centering piece contributes its own additive term
`b·θ̇·cosθ·(−k_x·x − k_d·ẋ)` to `Ė`, which is **not** sign-definite. Checked numerically (three random in-flight states): `Ė` computed from
the pump term alone vs. the full shipped `a_cmd` differ — in one sampled
case the full-`a_cmd` `Ė` even has the **opposite sign** from the pump-only
`Ė` (`+0.1176` vs `−0.2353`). This is a slightly stronger statement than
`EnergySwingUp`'s own module docstring makes ("the centering term ... does
not touch the pole subsystem's energy balance ... not a perturbation of the
energy argument") — that phrasing is best read as "the pump term's *own*
computation doesn't reference `x`/`ẋ`" (true), not "the *realized* `Ė` is
unaffected by centering" (measured false above). None of this contradicts the
measured convergence in "Final measured results" below — the TUNED gains
demonstrably swing the plant up and hold it, repeatedly — but that
convergence, for the shipped (pump + centering) law, is an **empirical**
result (the grid search + measured basin-entry), not a closed-form Lyapunov
proof the way the pump-term-alone argument above is. The clean proof covers
the idealized law; the shipped law's convergence is verified, not derived.

**The honest gap, and where it shows up empirically.** `V̇ = 0` not only at the
target (`Ẽ = 0`) but also whenever `θ̇ = 0` **or** `cosθ = 0` — a strict
Lyapunov-decrease argument alone does not rule out the trajectory getting
stuck in one of those larger sets; the rigorous next step is a **LaSalle
invariance** argument (show the *only* trajectory that can stay inside
`{θ̇=0} ∪ {cosθ=0}` forever, under the FULL nonlinear closed-loop dynamics —
not just `V̇`'s formula — is the target itself), not carried out in full here.
This is not a hand-wave: it is *exactly* what the shipped controller's own
measured "dead time" is. Starting from **exact** `hanging = [0, π, 0, 0]`,
`θ̇ = 0` exactly, so `a_cmd = k_E(E_up−E)·0·cos(π) = 0` exactly, and then
`θ̈ = (b·cos(π)·0 + m_p g l·sin(π))/J = 0` too (`sin(π) = 0` exactly in exact
arithmetic) — **exact hanging is itself a genuine fixed point of the closed
loop**, exactly the kind of point a pure-decrease argument cannot evict a
trajectory from. The measurement (`tests/test_swingup.py`): "starting
exactly at `hanging=[0,π,0,0]` ... the trajectory sits there (up to float64
noise from `sin(pi)!=0` exactly) for ~5 of the 8s budget before the swing
becomes visible ... regardless of `k_E`" — the *only* thing that breaks the
exact fixed point in the simulator is `np.sin(np.pi) ≈ 1.2×10⁻¹⁶` (π is not
exactly representable in float64), not the control law. On real hardware (or
with any sensor noise/friction) this symmetry breaks immediately; it is a
property of a noiseless, frictionless simulator started at an exact unstable
equilibrium, fully deterministic and reproducible (not flaky).

## Collocated partial-feedback-linearization: the `accel_to_force` inversion

`accel_to_force(z, a_cmd)` is the algebraic step that makes the derivation
above exact, not approximate. Write both EOM rows (`Mq̈ + Cq̇ + g = τ`,
`τ = [u, 0]ᵀ` — only the cart is actuated):

```
x-row (actuated):   m_t·ẍ − b·cosθ·θ̈ + b·sinθ·θ̇² = u        (m_t = m_c + m_p)
θ-row (passive):   −b·cosθ·ẍ + J·θ̈ − m_p g l·sinθ = 0
```

Substitute the DESIRED `ẍ = a_cmd` into the θ-row and solve for `θ̈` (shown
above), then substitute both `a_cmd` and that `θ̈` into the x-row and solve
for `u`:

```
θ̈ = (b·cosθ·a_cmd + m_p g l·sinθ) / J
u  = m_t·a_cmd − b·cosθ·θ̈ + b·sinθ·θ̇²
```

This is called **collocated** PFL because the coordinate whose acceleration
is being commanded (`x`, the cart) is the *same* coordinate that is actuated
— inverting the x-row for `u` given a desired `ẍ` is a direct, always-defined
substitution (`m_t > 0` always). Contrast the classical **Acrobot** swing-up
(elbow-actuated, shoulder passive): there the coordinate you want to shape
(the shoulder) is *not* the actuated one, so its PFL must invert through the
coupling between joints — well-defined only away from a mass-matrix
determinant singularity. The cart-pole's collocated case has no such
singularity (`m_t = m_c + m_p > 0` unconditionally), which is *why* this
plant's realization is exact for every in-rail state, verified to ~1e-9 by
round-tripping through `f` (`tests/test_cart_pole_dynamics.py`).

## S¹ topology: why the mode-switch is not a convenience, it's necessary

**The coordinate artifact.** An angle lives on the circle `S¹`, not on `ℝ`.
Representing it as a scalar `θ ∈ (−π, π]` (this repo's convention,
`wrap_to_pi`) forces a **branch cut**: `θ = π − ε` and `θ = −π + ε` are the
SAME physical angle but are far apart as real numbers. Two escapes exist for
a control LAW built from `θ`:

1. **`angle_diff(a, b) = wrap_to_pi(a − b)`** (`util/angles.py`): *handles*
   the cut algebraically (shortest signed distance) — used in `ModeSwitch`'s
   `V(x̂) = eᵀPe` (the angle component of `e` is `angle_diff`'d, never
   subtracted raw) and in the catch-child re-wrap (see the precondition
   section below).
2. **Any smooth function of `(cosθ, sinθ)`** instead of `θ` directly has
   **no cut at all** — `θ ↦ cosθ` is smooth and exactly `2π`-periodic on the
   whole circle, so anything built from it inherits that smoothness
   globally, with no artificial jump anywhere.

**Why this matters for THIS controller.** The pole's potential energy
`m_p g l·cosθ` — and therefore `pendulum_energy`, `energy_upright`, and the
entire `a_cmd = k_E(E_up − E)θ̇cosθ` law — is *exactly* such a function: it is
smooth and well-defined for **every** `θ ∈ S¹`, with no preferred branch. This
is *why* energy-shaping swing-up is **global** (works from any starting
angle, not just near one lift of it), while `u = −K(x̂ − z_ref)` (LQR/MPC) is
built from the raw SCALAR `θ` via linearization at one point — it is only
ever locally valid near that point, by construction.

**The deeper, topological point** (not just a coordinate/representation
issue): moving to the `(cosθ, sinθ)` embedding removes the *coordinate*
singularity, but **not** a *topological* one underneath it. `S¹` is not
contractible — informally, a continuous vector field on the circle that has
a stable zero (an attracting equilibrium) at one point cannot avoid having at
least one more zero somewhere else (a hairy-ball/Poincaré–Hopf-index style
obstruction restricted to `S¹`): you cannot continuously "comb" every
direction on the circle to point toward a single target with no other
equilibrium surviving. Consequently: **no continuous state-feedback law can
make upright the unique, globally attracting equilibrium on all of
`S¹ × ℝ`** (angle × angular velocity) — some second equilibrium (typically
hanging) necessarily persists for ANY continuous control law, full stop, not
a shortcoming of this particular design. Swing-up-and-catch therefore
**must** be a hybrid (mode-switched) controller — `ModeSwitch`'s
discontinuity (a jump in *which law* answers `update()`, triggered by a
discrete test on `V(x̂)`) is not an engineering convenience layered on top of
a nicer continuous solution; it is the only way to evade an obstruction that
rules out a continuous one existing at all.

**Hysteresis / Zeno-avoidance.** `ModeSwitch` uses `c_catch < c_release` (a
strict dead zone), not one threshold: `SWINGING → CATCHING` at `V < c_catch`,
`CATCHING → SWINGING` only once `V > c_release`. A trajectory sitting exactly
at a single threshold could, in principle, cross it infinitely often in
finite time (Zeno behavior) under measurement/integration noise; with a
dead zone of strictly positive width, the state must traverse the finite gap
`(c_catch, c_release)` — which, since it evolves continuously with locally
bounded velocity, takes strictly positive time — before it can switch back,
ruling out infinite switches in finite time.

## `c_catch`: the calibrated basin, and two ways to scale it across catches

> **Superseded:** the base value `0.05` below was later found too tight — it
> made a near-upright nudge spin the pole forever — and was widened to `2.0`,
> sized to the catch's true region of attraction. See the **"Widening
> `c_catch` to the true catch basin"** section that follows. The `0.05`
> measurement and the λ_min *scaling* argument in this section remain accurate
> for the value as it stood; only the base changed.

**The calibrated number** (`registry.BASIN_V_LQR_CALIBRATED = 0.05`): measured, the
tuned `EnergySwingUp` (`k_E=1.5, k_x=4.0, k_d=3.0`) drives
`V(z) = eᵀP_lqr e` (a FRESH `CONTROLLERS["lqr"](plant, {})`'s CARE `P`) down
to a **measured minimum of 0.015842** within an 8s run from hanging, holding
`V < 0.05` for a real ~0.5s+ dwell near the end of that run (not a one-tick
fluke). `0.05` sits **~3.2× above** that measured minimum — comfortably
inside the basin so the switch reliably triggers, not right at the numerical
edge — while remaining small relative to `P_lqr`'s dominant θ-θ entry
(~22.9): ignoring cross-terms, `V < 0.05` bounds `|θ_error| ≲
√(0.05/22.9) ≈ 0.047 rad` (~2.7°) — a genuinely tight neighborhood of
upright, confirmed by test (`swingup_factory` auto-selects `angle_idx=(1,)`,
`c_catch=0.05`, `c_release=0.1`).

**The scale problem.** `c_catch` is defined in terms of `P_lqr`'s own metric,
but `ModeSwitch` needs a threshold in whatever `P` the ACTUAL catch uses —
which differs for MPC. Measured: MPC's DARE terminal cost sums
UNDISCOUNTED per-step costs (`mpc.py`'s "scale note"), so for this plant
`‖P_mpc‖ ≈ 200.2·‖P_lqr‖` (elementwise ratio uniform to within 0.3%). Without correcting for this, the raw `0.05` against `P_mpc`
demands a basin ~200× tighter than the swing-up ever measures reaching — the
MPC catch never triggers (measured before the fix: `catch_time=None` over a
12s run).

**Two honest formulations, presented side by side:**

1. **The earlier norm-ratio heuristic** (the first version shipped; since
   replaced — see the decision below; not what
   `registry.swingup_factory` computes today). It set
   `scale = ‖catch.P‖_F / ‖P_lqr_reference‖_F` (Frobenius norm) and
   `c_catch = 0.05 · scale`. This is **exact** — reproduces the true "same
   physical neighborhood of upright" rescaling — *only* when `P_catch` is a
   scalar multiple of `P_lqr` (`P_catch = α·P_lqr`), because then the
   Frobenius-norm ratio equals `α` exactly, for any `α`.

2. **Shipped: the generalized-eigenvalue form**,
   computed numpy-only by `registry._min_generalized_eig` (Cholesky + `eigvalsh`).
   `c_catch = 0.05 · λ_min(P_catch, P_lqr)`, where `λ_min` is the smallest
   generalized eigenvalue of the pencil `P_catch v = λ P_lqr v`. By the
   variational characterization of generalized eigenvalues,
   `eᵀP_catch e ≥ λ_min·eᵀP_lqr e` for **every** `e` (not just on average) —
   so `eᵀP_catch e < c_catch ⟹ eᵀP_lqr e < 0.05` **unconditionally**, i.e.
   `basin_catch ⊆ basin_lqr` **regardless of how P_catch's shape (not just
   its overall scale) differs from P_lqr's**. This is the guarantee the
   norm-ratio heuristic does *not* have in general: a single global scalar
   cannot correctly rescale a basin whose eigenstructure (directional
   weighting) differs from `P_lqr`'s, even if the two matrices' overall
   "size" (Frobenius norm) matches.

**Measured** (an independent scipy cross-check): for `cart_pole_plant()`'s factory-default MPC catch,

```
||P_mpc||_F / ||P_lqr||_F  (shipped norm-ratio scale) = 200.222898
lambda_min(P_mpc, P_lqr)   (provably-correct scale)    = 200.126240
lambda_max(P_mpc, P_lqr)                               = 206.618930
spread (lambda_max - lambda_min) / lambda_min          = 3.2443%
c_catch (shipped, norm-ratio)          = 10.011145
c_catch (provably-correct, lambda_min) = 10.006312
shipped vs exact relative difference   = +0.0483%  (shipped is LARGER/less conservative)
```

Read this precisely — it is subtler than a flat "the heuristic is a few
percent conservative": the two `c_catch` VALUES themselves differ by only
**0.048%** (the shipped heuristic is extremely close in absolute terms), but
the pencil's eigenvalues themselves SPREAD by **3.24%** (`λ_max` vs `λ_min`)
— i.e. `P_mpc` is close to, but not exactly, a scalar multiple of `P_lqr`
(elementwise ratio measured in `[200.01, 200.56]`, mean `200.11`). Because
the shipped scale (200.223) sits **slightly above** `λ_min` (200.126), the
shipped heuristic is, in the single worst eigendirection, very slightly
(~0.05%) on the **non-conservative** side of the exact guarantee — a state
right at the shipped threshold in that one direction could correspond to
`eᵀP_lqr e` up to ~0.05% above the nominal `0.05`, not strictly under it.
Everywhere else (every other direction, up to the best eigendirection at
`λ_max`), the shipped heuristic is comfortably conservative (up to ~3.2%
so). **Practically the difference is negligible** (0.05% on a threshold that
is itself a ~3× safety margin over the measured minimum) — but the norm-ratio
is not a mathematical proof, and the `λ_min` form is. **Decision: ship the `λ_min` form.** `registry.swingup_factory` now sets
`c_catch = 0.05·λ_min(catch.P, P_lqr)` via the numpy-only `_min_generalized_eig`
(Cholesky + `eigvalsh`). Two committed tests lock it in:
`test_swingup_factory_mpc_catch_c_catch_guarantees_basin_containment` pins the
containment worst case at **exactly 1.000000** (the norm-ratio gave ~1.0005, a
hair OUTSIDE the basin), and `test_min_generalized_eig_matches_scipy_oracle`
cross-checks the numpy λ_min against scipy's generalized `eigh`. For an LQR catch
`λ_min = 1` identically, so `c_catch = 0.05` is unchanged — the guarantee costs
the LQR path nothing.

## Widening `c_catch` to the true catch basin — the near-upright-nudge fix

**The bug (found in live-UI testing).** Selecting LQR/MPC on the cart-pole
while the pole was *near upright but moving* — a mouse nudge, or a live
controller-swap from `None` after tapping the cart — made the pole spin over
the top indefinitely instead of being caught ("uncontrolled, swings back and
forth passively"). A clean start (`start=upright`, or a full swing-up from
hanging) still worked — which is the tell: the difference is whether the state
carries velocity when the catch is asked to engage.

**Root cause (measured, not asserted).** `ModeSwitch.reset()` always begins in
SWINGING and only hands off when `V = eᵀPe < c_catch`. With the old
`BASIN_V_LQR_CALIBRATED = 0.05`, that threshold was **~300× tighter than the
catch's true region of attraction**. Measured on `cart_pole_plant()` (default
Q/R), a *naked* LQR catch stabilizes pole tilts out to ~0.8 rad (`V ≈ 15`) and
pole-velocity kicks to ~5 rad/s (`V ≈ 25`). But *any* nudge exceeds 0.05 (a
θ̇ = 0.5 rad/s kick alone is `V = 0.25`), so the ModeSwitch committed to
energy-pumping a pole that already had ≈ `E_up`. The pump drove it onto the
`E ≈ E_up` separatrix and it rotated forever, skimming the catch basin (min
`V ≈ 0.06`) but never dipping under 0.05. The `0.05` was only ever calibrated
as a "the from-hanging pump has parked the pole slow at the top" detector (min
V ~0.0158, ~3.2× headroom) — the right number for THAT job, but the wrong
quantity to gate "is this state catchable?", which is the much larger RoA.

**The fix: size `c_catch` to the catch's actual RoA.**
`BASIN_V_LQR_CALIBRATED: 0.05 → 2.0` — ~7× inside the measured basin (~15),
conservative because the true basin is not exactly a V-sublevel set, yet large
enough that a real near-upright nudge is CAUGHT, not pumped. Measured safe
window: the from-hanging swing-up still enters and settles for any threshold in
`[0.25, 4.0]` (its min V ~0.0158 is far below all of them, so the demo is
untouched); the nudge (θ̇ = 1.5 rad/s) settles cleanly at every value ≥ 0.25 and
spins forever at 0.05. Regression-locked by
`test_swingup_catches_nudged_near_upright_pole` (RED at 0.05: tail |θ error| ≈
π, i.e. spinning; GREEN at 2.0: caught, final |θ error| ≈ 0, mode CATCHING).

**Alternatives considered.**
- *Decouple: keep 0.05 for the in-swing hand-off, add a separate wide
  "already catchable" seed on `reset()`.* Rejected (YAGNI): the measured
  `[0.25, 4.0]` window shows the tight in-swing threshold is unnecessary even
  for the from-hanging demo, so one widened threshold serves both the hand-off
  and the catchability gate — less code, one notion of "the basin."
- *Minimal bump to ~0.15 (just above the pump-skim min V ≈ 0.06).* Rejected:
  fragile (tuned to one skim value) and still ~100× tighter than the real RoA,
  so it would catch a nudge only after a wasteful pump excursion.

**Knock-on: the disturbance-recovery e2e was re-tuned (measured).** The wider,
correctly-sized basin makes the *unclipped* LQR catch genuinely robust — it now
ABSORBS cart kicks up to ~100 N/20 ms without ever dropping to SWINGING (the
old 20 N kick no longer leaves the basin — the LQR just recovers it). To still
exercise the `CATCHING → SWINGING → CATCHING` re-swing path,
`test_disturbance_recovery_re_swings_and_recatches` now uses a **200 N/20 ms**
shove (measured window: ~150 N through at least ~450 N all give the clean round
trip, re-catching ~1 s after the kick — a brief knock-out, not a fall to
hanging). This is not a regression: absorbing a moderate disturbance instead of
doing a full re-swing is the *better* behavior a right-sized basin buys.

**The λ_min containment argument (previous section) is unchanged** — it is
scale-invariant. With base 2.0: an LQR catch still has `λ_min = 1` so
`c_catch = 2.0`; the default MPC catch has `λ_min ≈ 200.1` so `c_catch ≈ 400`
(was ~10.0 at base 0.05) — the same certified physical neighborhood
`eᵀP_lqr e < 2.0`, just expressed in `P_mpc`'s units. The old-vs-new figures in
"Final measured results" below (catch at ~7.13 s, 20 N disturbance) are the
original record at base 0.05; the fix's numbers are those quoted here.

## Principal-branch precondition — a measured necessity

**The failure, measured (the end-to-end run).** `LQRController`/
`MPCController` apply raw linear feedback (`u = −K(x̂ − z_ref)` /
`u = −K_d x̃₀`) with **no angle wrapping of their own** — a contract only ever
exercised, before swing-up existed, by states that stay within one branch of
`θ`. A swing-up trajectory routinely arrives at the catch basin with `θ`
having wound through an EXTRA multiple of `2π` (e.g. `θ ≈ 6.20 rad` —
physically `~-0.082 rad` from upright, but numerically nowhere near the
origin the catch law was linearized at). Handing that raw `θ` to LQR
(`θ`-gain ≈ 63) is catastrophic: measured, this produced a single **−395 N**
tick that itself threw `V` from `0.05` to `3.7` in one step — an immediate
`CATCHING → SWINGING` bounce, not a catch.

**The fix.** `ModeSwitch` re-wraps the angle components (`angle_idx`) to
within `±π` of `z_up` — `z_up[i] + angle_diff(x̂[i], z_up[i])` — on **every**
tick it is about to consult (or reset) the CATCH child, never the swing-up
child (`EnergySwingUp`'s own law is built from `cosθ`/`sinθ`, already exactly
invariant to any multiple of `2π` — see the S¹-topology section — so it needs
no such fix; confirmed by `tests/test_mode_switch.py`'s regression test,
which fails without the fix and passes with it).

**The general precondition, stated for any future direct consumer:**
`LQRController`/`MPCController` assume their input state is already in the
**principal angle branch** (within `±π` of the corresponding `z_ref`/upright
component) — a raw, un-rewrapped feed from a controller/estimator that winds
through multiple revolutions (any future swing-up-adjacent code, or a real
sensor that free-runs an encoder count rather than wrapping it) MUST wrap
first. A one-sentence docstring note on both classes states
exactly this (`dpend/controllers/lqr.py`, `dpend/controllers/mpc.py`).

## A second, measured interaction: the RoA supervisor must not arm for a hybrid controller

Found by a test going unexpectedly red, then root-caused. The live UI's
region-of-attraction (RoA) supervisor (`interactive/app.py`) arms whenever the
active controller exposes a `.P`
— but `ModeSwitch` **also** exposes `.P` (borrowed from its catch child, so
its OWN switching threshold means the same neighborhood of upright as the
catch law). Measured: from `start="hanging"`,
`V(hanging) = 226.4` vs the supervisor's own `V_lim = 2.065` (factory
cart-pole Q/R) — hanging is, correctly, far outside the LINEAR catch's
basin, so pre-fix the supervisor tripped on the very first CONTROLLER-mode
tick and silently reverted to MANUAL before the energy pump ever got to act,
defeating the entire swing-up demo. **Fix:** the supervisor arms only for a
controller that does NOT declare itself hybrid — it checks
`not getattr(ctrl, "IS_HYBRID", False)`, and `ModeSwitch` sets a class-level
`IS_HYBRID = True`. A hybrid is a supervisor in its own right (roaming far from
upright during `SWINGING` is the intended behavior, not a fault), so a second,
small-tip-calibrated linear supervisor must not be layered on top of it. The
flag DEFAULTS to False, so LQR/MPC are supervised with zero change — their own
"must fire"/"must not fire" tests stay green.

**A refinement.** The initial fix
keyed off `not hasattr(ctrl, "mode")` — reusing the very attribute the HUD
reads to display the swing mode. That is fragile *duck typing of a safety
policy*: a future non-hybrid controller that exposes `.mode` for any unrelated
reason would silently lose its supervisor, and a future hybrid not named `.mode`
would be wrongly supervised. An explicit `IS_HYBRID` flag makes the supervision
policy a **declared** property of the controller, not one inferred from an
incidental attribute — and its default-False means the *safe* direction (plain
controllers keep the net) is automatic, while a forgotten flag on a new hybrid
fails LOUDLY (immediate spurious disengage in testing), never silently.
Regression-tested two ways:
`tests/test_interactive_app.py::test_swingup_controller_does_not_arm_the_roa_supervisor`
(the hybrid is not armed) and
`test_roa_supervisor_gate_uses_IS_HYBRID_flag_not_mode_attribute` (a non-hybrid
that DOES expose `.mode` is still armed — the guard against reverting to duck
typing). Both fail without the respective code, pass with it.

## Honest scope

**Single-pole swing-up is reliable** (measured, repeatable — see "Final
measured results" below): both catch controllers (LQR, MPC) swing up from
exact hanging, catch within milliseconds of each other, and hold. **The
double-pendulum-on-cart swing-up is deliberately deferred** — energy
shaping does **not** generalize to the double pendulum (there is no single
scalar "energy" whose gradient collocated-PFL-inverts as cleanly for a
2-DOF passive subsystem the way it does for one passive angle here); the
honest method for the double is **trajectory optimization** (an offline
hang→upright trajectory via iLQR/DDP or direct collocation, tracked online
by TVLQR, catching to the existing LQR/MPC once close) — a real, separate,
future build (a separate `controllers/trajopt_swingup.py`),
not a small extension of this one.

**The `I = m_p·l²/12` pole-inertia default** (`cart_pole_params.py`) is a
documented MODELING CHOICE, not a physical measurement: it mirrors this
repo's existing double-pendulum convention (`m·length²/12`, the thin-
uniform-rod-about-its-own-COM formula), using `l` (pivot-to-COM distance) as
the length scale. It is a compact placeholder, not a first-principles
derivation for a rod of length `2l` pivoted at one end — that geometry would
give `m_p·l²/3` by the parallel-axis theorem instead. `I` is freely
overridable (e.g. `I=0` models an idealized point mass at the COM). This
choice is self-consistent everywhere it enters (`J = m_p l² + I` appears
identically in the mass matrix, the linearization, and the energy — never
inconsistently in one place and not another), and — as the `Ė` derivation
above shows directly — **the swing-up law's convergence argument never
assumes a particular value of `J`, only that `J > 0`**: `J` only rescales how
fast `θ̈` responds to a given `a_cmd`/torque, not the sign argument that
makes `E → E_up`. The inertia choice is therefore a genuine free physical
parameter, correctly decoupled from the control law's correctness.

## Final measured results

> **Note:** the figures in this section are the original records at `c_catch`
> base `0.05`. After the `0.05 → 2.0` widening (see the near-upright-nudge fix
> section above) the catch fires a touch earlier and the disturbance-recovery
> kick was re-tuned to 200 N; refreshed figures are quoted in that section.
> These historical records are left as-is.

**Swing-up reachability** (`tests/test_swingup.py`):
```
velocity reversals (~swings) over the full 8s run: 5
time E first within 5% of E_up: 6.58 s
time-to-basin (V<0.05): 7.13 s
MEASURED min V = 0.015842 at t=7.490 s
max|x| = 0.8541 m (rail=1.5 m)
peak|u| = 21.2303 N
final E = 2.4517 J, E_up = 2.4525 J, final frac diff = 0.0003
```
Gains (`EnergySwingUp` FINAL defaults): `k_E=1.5, k_x=4.0, k_d=3.0,
u_max=None`. Tuning method: a 140-combination grid
(`k_E∈{1.0,1.2,1.5,1.8,2.0} × k_x∈{2,4,6,8,10,15,20} × k_d∈{1,2,3,5}`), 71
passing both filters (rail-safe AND basin-reaching), ranked by smallest
measured min-V; `k_E=1.5/k_x=4.0/k_d=3.0` chosen over the single-best min-V
candidate (`k_E=2.0`, min-V=0.0095) for two measured reasons: far more rail
margin (`max|x|=0.8541 m` vs `1.4425 m` — 57% vs 96% of the available rail)
and a visibly cleaner, more monotonic `V` descent into the basin (vs
`k_E=2.0`'s wild pre-basin overshoots of `V` into the hundreds).

**Real-display self-drive** (live on the x11 window).
The full UI cycle passed end-to-end: boots hanging with the RoA supervisor
correctly **not** armed (`v=v_limit=None` — see below); engaging the controller pumps up
and catches (`min‖z−z_up‖=0.0320`) with the outer supervisor **never** firing a
spurious MANUAL disengage (the fix); a knockout impulse ejects the pole past
`c_release`→SWINGING; it re-catches to `‖z−z_up‖=0.0000`; `R` resets to hanging;
`ESC` exits. One concrete finding worth recording, and it is the **live
corollary of the non-sign-definite centering term above**: the knockout that
re-catches cleanly is a *pole-torque impulse that leaves the cart centered* —
even a violent one (post-knockout `‖z−z_up‖≈2.9`, essentially back to hanging)
recovers, because swing-up from hanging **at center** is exactly the in-scope
case. A hand-*drag*, by contrast, both flings the pole AND parks the cart
off-center; the `−k_x·x` term then injects a large standing acceleration bias
(`x≈0.9 m ⇒ −k_x·x ≈ −3.6 m/s²`) that fights the pump, and recovery stalls
within a 12 s budget. So the reliable-recovery guarantee is "pole knocked, cart
near center," not "arbitrary state" — the cornered-cart swing-up shares the
double-pendulum's out-of-scope status for the same underlying reason (the
single scalar energy law has no authority to independently re-home the cart).

**Headline e2e** (`tests/test_swingup_e2e.py`; `x0` =
hanging, `T=12s`):
```
[catch=lqr] catch time (first mode->CATCHING): 7.13 s
[catch=lqr] final ||z-z_up|| (wrapped) = 0.000866
[catch=lqr] max ||z-z_up|| over last 2s = 0.009792
[catch=lqr] final z = [-5.39448417e-04  6.28327526e+00  6.62615351e-04 -1.09370833e-04]

[catch=mpc] catch time (first mode->CATCHING): 7.135 s
[catch=mpc] final ||z-z_up|| (wrapped) = 0.000855
[catch=mpc] max ||z-z_up|| over last 2s = 0.009638
[catch=mpc] final z = [-5.34139089e-04  6.28327362e+00  6.52803539e-04 -1.07262998e-04]
```
Both catches converge within ~0.005s of each other and hold (last-2s max
norm ~0.0096-0.0098, well under the 0.1 requirement); `θ_final ≈ 6.2833 rad
≈ 2π` in both — one full revolution during swing-up, correctly recognized as
upright by the angle-wrapped norm.

**Disturbance recovery** (`tests/test_swingup_e2e.py`;
hang → swing → catch → hold → 20N/20ms cart kick at t=9.0s → re-swing → re
-catch, one continuous `simulate()` call):
```
mode timeline (t, new_mode): [(0.0, 'SWINGING'), (7.13, 'CATCHING'), (9.02, 'SWINGING'), (10.92, 'CATCHING')]
final ||z-z_up|| (wrapped) = 0.000582
max ||z-z_up|| over last 2s = 0.005334
final z = [-3.72699626e-04  1.25664260e+01  4.39920374e-04 -6.15332781e-05]
```
Exact 4-phase round trip (`θ_final ≈ 12.566 ≈ 4π` — two full revolutions
accumulated across the initial swing-up and the re-swing). Disturbance
sizing (measured): kicks below ~18N/20ms are fully absorbed without
ever leaving the basin; kicks above ~25N/20ms dump enough energy to send the
pole into a **sustained full rotation** (`E` pushed well above `E_up` into a
near-separatrix state that does not recover within any window tested up to
30s — real pendulum physics, not a bug); 20N/20ms sits in the narrow window
that reliably crosses `c_release` yet stays inside the basin the swing-up law
recovers from in ~1.9s.

**`c_catch` scale correction** (measured — see the dedicated section above for the full table): shipped norm-ratio
scale `200.223` vs provably-correct `λ_min = 200.126` (0.048% apart in
`c_catch` terms; 3.24% generalized-eigenvalue spread).
