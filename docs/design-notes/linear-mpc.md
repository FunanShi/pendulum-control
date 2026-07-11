# Design note — Linear MPC via a condensed QP

*The decision, the alternatives, why, the failure modes —
and the numerical walls we hit on the way. A design note, sibling to
[`lqr-riccati.md`](lqr-riccati.md). Every number below was
measured (Docker container, `osqp==1.1.3`, `cvxpy==1.9.2`), not asserted.*

## The decision, in one paragraph

`dpend/controllers/mpc.py` implements receding-horizon linear MPC as a
**hand-condensed, dense quadratic program**: eliminate the predicted state
sequence algebraically (`X̃ = Sx x̃₀ + Su U`), leaving an unconstrained-
dimension QP in the input sequence alone, `J(U) = ½UᵀHU + fᵀU`, `H = SuᵀQ̄Su +
R̄`. **OSQP solves it** (never formulates it — the condensing, the cost
assembly, and the constraint bookkeeping are pure numpy); OSQP is `setup()`
once at construction (`H`, the constraint rows `A_c` are structurally fixed)
and re-`update()`+`solve()`d every 5 ms tick with a warm start. Terminal cost
`P_f` = the **discrete** Riccati (DARE) solution, not the continuous CARE `P`
— the decision this note spends the most time defending, because it is what
makes unconstrained finite-horizon MPC ≡ infinite-horizon discrete LQR an
**exact theorem**, not a fast-sampling approximation. `cvxpy` (declarative
formulation oracle) and `scipy.linalg.expm`/`solve_discrete_are` are
strictly TEST-ONLY; neither is imported inside `dpend/`.

## Why hand-rolled at all

Same rationale as the LQR note: hand-roll the
**formulation** — condensing, cost assembly, constraint stacking, exact ZOH
discretization — and leave the **numerical optimization**
(OSQP's ADMM iterations) in the library layer. Wrapping an already-fully-
formed dense `H`/`A_c` in `scipy.sparse.csc_matrix(...)` for OSQP's C core is
a data-format adapter, not "using a library to do the formulation" — no more
than calling `np.ndarray()` would be (see `mpc.py`'s own note).

## Condensing — the derivation summary (dimensions)

Cart plant: `n=6` (`[x, θ1, θ2, ẋ, θ̇1, θ̇2]`), `m=1` (cart force). Factory
default `N=40` (200 ms lookahead at `ctrl_dt=5 ms`). Stacking:

- `U ∈ ℝ^{N·m} = ℝ⁴⁰` (decision variable — OSQP's whole problem).
- `X̃ ∈ ℝ^{N·n} = ℝ²⁴⁰` (eliminated, not a decision variable).
- `Sx ∈ ℝ^{240×6}` (block `i` = `A_d^{i+1}`), `Su ∈ ℝ^{240×40}` (lower-block-
  triangular, block `(i,j) = A_d^{i−j}B_d` for `j≤i` — causality: `u_j` can
  only affect states from step `j+1` on).
- `H ∈ ℝ^{40×40} = SuᵀQ̄Su + R̄` — **this** is OSQP's whole problem size, not
  `240×240`: condensing trades a bigger, sparse, structured QP (the
  KKT/sparse formulation, one block per stage) for a smaller, dense,
  unstructured one. That trade is the entire condensed-vs-sparse story
  below.

## Condensed vs sparse — the measured conditioning wall

**Chosen: dense-small condensed**, for three reasons: (1) at `N=40`, `H` is
`40×40` — trivially small for a dense factorization, and OSQP caches that
factorization once at `setup()`; (2) the condensing algebra (`Sx`, `Su`,
`H=SuᵀQ̄Su+R̄`) is hand-derivable linear algebra worth keeping explicit,
whereas a sparse KKT formulation's block-elimination is
materially more solver-engineering, not more control theory; (3) OSQP's
dense-`P`/`A` path is simpler to reason about and to test (the `cvxpy`
declarative oracle rebuilds the SAME QP a totally different way — sparse,
one block per stage — and agrees with the condensed solve to atol 1e-5,
`tests/test_mpc.py`).

**The honest limit, measured** (`cart_plant()`, factory Q/R,
sweeping `N`, reading the controller's own shipped `H`):

| N | cond(H) | H's smallest eigenvalue |
|---|---|---|
| 40 (factory default) | **1.626×10²** | 0.1001 |
| 60 | 1.223×10³ | 0.1001 |
| 80 | 7.862×10³ | 0.1001 |
| 100 | 4.607×10⁴ | 0.1001 |
| 150 | 3.128×10⁶ | 0.1001 |
| 200 | 1.881×10⁸ | 0.1001 |
| 300 | 6.201×10¹¹ | 0.1001 |
| 400 | 7.726×10¹⁵ | 0.05434 (below the 0.1 floor — eroding) |
| 500 | 1.338×10²⁰ | **−2.239×10²  (H is INDEFINITE)** |
| 600 | 5.193×10²³ | −6.628×10⁵ |
| 700 | 8.109×10²⁷ | −4.285×10⁹ |
| 800 | 5.703×10³¹ | −1.778×10¹³ |

**Historical data point, for context:**
the FIRST version of this controller used the continuous CARE `P` as
terminal cost (see "DARE terminal cost" below for why that was rejected),
and an earlier investigation measured the SAME conditioning wall for THAT
(superseded) formulation: **cond(H) = 6.1 @ N=40 → 7.7×10² @ N=100 →
8.8×10⁹ @ N=300 → 7.8×10²⁹ @ N=800.** The DARE terminal cost documented
here makes `N=40`'s conditioning ~27× WORSE in exchange for exactness
(`6.1→162.6`) — the terminal block `P_f` is now ~200× larger in magnitude
(the "scale note" below), and `H`'s late rows scale with it. Both walls are
real, measured, and grow at roughly the same rate in `N`; they differ in
WHERE they start, not in shape.

Reading this table: `cond(H)` grows roughly geometrically in `N` because
`Su`'s late-horizon blocks are powers of the open-loop-**unstable** `A_d`
(an eigenvalue `>1` in magnitude) — `SuᵀQ̄Su` inherits that exponential
dynamic range. `mpc.py`'s own regression test
(`test_condensed_hessian_well_conditioned_at_factory_default_horizon`) pins
the `N=40`/`N=100` rows above (1.626e2 / 4.607e4) as a guard so nobody
lengthens the horizon "for a longer lookahead" blind to this cliff.
**A sharper finding**: the `1e-9·I` regularization is not
just an ill-conditioning nuisance past some point — by `N=500` the
*symmetrized, regularized* `H` OSQP would actually receive is **no longer
positive semi-definite** (measured smallest eigenvalue `−223.9`). That is a
correctness failure, not merely a numerical-precision one: OSQP's assumption
that `P` is PSD is violated, and its behavior on an indefinite `P` is
undefined, not just imprecise. `N=40` sits **more than 18 orders of
magnitude** of headroom below where this starts (`cond(H)` grows from `1.6e2`
to `1.3e20` between `N=40` and `N=500`) — comfortably safe, but the wall is
real and much closer in `N`-space than intuition suggests (`N=100→500` is
"only" a 12.5× horizon increase for a 10¹⁸× conditioning increase).
**Sparse, KKT-based (non-condensed) MPC never forms `SuᵀQ̄Su` at all** — each
stage's dynamics stay a separate equality constraint, so the factorized
matrix's conditioning tracks the *per-stage* system, not the horizon's
compounded instability. That is the documented scaling fix, not built here.

## DARE terminal cost — why it makes the equivalence EXACT

**The theorem.** The finite-horizon backward Riccati recursion (`P_N=P_f`,
then `P_k = Q + A_dᵀP_{k+1}A_d − A_dᵀP_{k+1}B_d(R+B_dᵀP_{k+1}B_d)⁻¹B_dᵀP_{k+1}A_d`)
has the DARE solution `P*` as its **fixed point**. Seed `P_f = P*` and every
stage's cost-to-go is exactly `P*` — the stage-0 optimal gain equals the
infinite-horizon discrete LQR gain `K_d` for **any** `N≥1`. This is why
unconstrained finite-`N` MPC ≡ infinite-horizon discrete LQR is a *theorem*,
not an `N→∞` limit, and it is also the textbook stability argument (Rawlings,
Mayne & Diehl): `P_f` as the exact unconstrained cost-to-go beyond the
horizon is a control-Lyapunov terminal cost.

**Measured, to the QP solver's own tolerance floor:** `max|u_mpc −
(−K_d·x̃0)|` over 20 random small states = **1.371×10⁻⁶** (re-running
`tests/test_mpc.py`'s exactness test) — matching OSQP's
`eps_abs=eps_rel=1e-6` almost exactly. There is no better number to report
here: the *theorem* is exact; the only daylight is the solver's own
convergence tolerance, and the measurement lands right on it.

**The rejected alternative, and why (mirrors the LQR note's "fork" structure):**
the first version used the **continuous CARE** `P` here — the standard
fast-sampling shortcut, with the attraction that LQR/MPC/the UI supervisor
would share one bit-identical `P`. Measured, this plant does not sit deep
enough in the fast-sampling regime for that shortcut to be free: the cart's
fastest closed-loop pole is `−15.95 rad/s`, so `dt·|λ| ≈ 0.08`, and the
continuous-vs-discrete gain gap measured **6–9% per entry** — five orders of
magnitude above the ~1e-4 the original LQR-limit test demanded, at *any* `N`
(the conditioning wall above rules out "just take N huge" as an escape:
`cond(H)` is already at `4.6e4` by `N=100`, long before the discretization
gap would meaningfully close). The DARE terminal cost replaces an
`O(dt·|λ|)` approximation with an exact equivalence at the cost of `P_f`
having a **different scale** than LQR's continuous `P` (`‖P_dare‖ ≈
‖P_care‖/dt`, ~200× larger here, confirmed below) — `K_lqr` (continuous)
stays exposed as the cross-reference/fallback gain; `K_d` (discrete) is the
gain the equivalence theorem actually holds for:

```
K_d   = [  9.26, -112.10, -362.15,  16.61, -75.86, -69.22]   (discrete, DARE)
K_lqr = [ 10.00, -122.92, -384.67,  17.91, -81.59, -73.74]   (continuous, CARE)
```

max per-entry relative gap = **8.81%** — right where the `dt·|λ_fast|≈8%`
estimate predicts.

## Constraint semantics

**Hard input** (`u_max`, every plant): box rows `±u_max` on every one of the
`N` predicted inputs — the same actuator limit at every step. **Hard rail**
(railed plants only, `x_max = plant.rail − 0.1 m` margin by default): the
cart-position rows of `Su`, bounding the **ABSOLUTE** cart position, not the
tracking-shifted error coordinate — a rail is a fixed physical object; it
does not move when a tracking target does. Concretely, wanting `−x_max ≤ x̃_k[0]
+ z_ref ≤ x_max`, the bound on `(Su U)`'s position rows must subtract off
`z_ref` and the propagated initial condition `(Sx x̃0)` explicitly (`mpc.py`'s
"Constraints" section derives this in full) — getting this wrong (bounding
`x̃_k[0]` as if it were absolute) is flagged in the module docstring as *the*
easiest subtle bug here, and is exactly what
`tests/test_mpc.py::test_tracking_shift_uses_absolute_rail_bounds_not_shifted_coordinates`
exists to catch open-loop, and
`tests/test_mpc_e2e.py::test_cart_mpc_tracks_with_rail_genuinely_active_near_target`
exists to catch **closed-loop**: with the target at `+0.8 m` and `x_max`
tightened to `0.812 m`, the trajectory measurably **rides the wall** (peak
cart position `0.8120007 m` — `0.7 µm` past `x_max`, inside OSQP's own
`1e-6` solver tolerance) while still converging cleanly.

**Fallback as the documented failure mode.** Any OSQP status other than
`"solved"`/`"solved inaccurate"` sets `status="fallback"` and returns, in
order: (1) the shifted previous plan's new first entry, if one exists; else
(2) the clipped continuous-LQR law — a safe, always-defined last resort,
**not** Lyapunov-certified. Two honest, measured illustrations (not
hidden): the rail-showdown headline (below) hits 8/1600 ticks of
`"maximum iterations reached"` (not infeasibility) right at its tightest
moment — the promised bound still held (fallback absorbed it); and probing
just **0.5 mm** below the tight-tracking test's `x_max=0.812` (to `x_max=
0.8115`) collapses the SAME scenario into ~800/1600 fallback ticks and loses
the pendulum entirely (final `|θ1|~1 rad`) — a real, sharp cliff between
"the horizon can still certify recovery" and "it cannot," not a gradual
degradation. **A hard constraint is only as hard as the horizon can
certify**; this is the single biggest caveat about this
controller.

## Warm start

Every tick: `osqp.update(q=f, l=l, u=u)` (values only; `H`/`A_c` sparsity is
untouched, so no re-factorization), then `warm_start(x=shift(U_prev))` — drop
`u_0*`, keep `u_1*..u_{N-1}*`, repeat `u_{N-1}*` as the new last entry (the
standard receding-horizon warm start: "yesterday's plan, one step later" is
much closer to the true optimum than a cold start). **Measured benefit**
(a clean, unconfounded test: solve the *identical* QP instance
twice, once with the production warm start and once forced to a zero
`(x,y)` start, with **no** intervening simulation step, so there is no
closed-loop trajectory divergence to confound the comparison):

| scenario | warm mean iters/tick | forced-cold mean iters/tick | reduction |
|---|---|---|---|
| calm regulation (tip 0.15) | 25.03 | 40.12 | **1.60×** |
| rail-showdown instance | 40.66 | 54.47 | **1.34×** |

Per-tick (calm case): warm strictly fewer iterations on 967/1600 ticks, tied
on 632/1600, worse on only 1/1600. (A first, naive attempt at this
measurement — patching `_U_prev=None` inside the `solve()` wrapper instead
of neutralizing `warm_start()` itself — showed **no** difference at all,
because OSQP's own persistent internal iterate carries over between
`solve()` calls on the same object regardless; that dead end is worth
remembering as a lesson in itself, see below.) `U_prev` does double duty
beyond speed, though: it is also the **fallback law's** basis (the shifted
previous plan), so it is load-bearing even where the iteration-count benefit
were negligible.

## The KKT view

OSQP solves `min ½UᵀHU+fᵀU` s.t. `l ≤ A_c U ≤ u` by (approximately)
satisfying, at convergence: stationarity `HU* + f + A_cᵀy* = 0`, primal
feasibility `l ≤ A_c U* ≤ u`, and **complementary slackness** — a row's dual
`y_i` is nonzero *only if* that row's constraint is active (its slack is
exactly zero) at the optimum. Measured, directly (the tight-tracking
scenario, `x_max=0.812`): at the tick with the single largest rail
dual, **exactly 1 of the 40 rail rows** has a nonzero multiplier (row 39 —
the *terminal* predicted step, exactly where the wall was found to bind) —
`y=56.98` — while the other 39 rail rows and all 40 box rows read `y≈0`.
That single nonzero multiplier *is* the shadow price: it says relaxing
`x_max` by one more metre at that predicted step would reduce the optimal
cost by ≈57 (cost units) — a concrete, measured instance of "the dual tells
you which constraint is buying its keep, and how much."

## Four numerics lessons (measured)

**1. The conditioning wall goes past "hard to invert" to "wrong."** See the
table above: `cond(H)` crosses from merely enormous to **H is no longer
PSD** between `N=400` (min eig `+0.054`) and `N=500` (min eig `−224`). The
factory default sits 12.5×-in-`N` / ~18-orders-of-magnitude-in-`cond`
below that line.

**2. DARE exactness, measured to the solver floor:** `1.371×10⁻⁶` (above) —
the theorem is exact; OSQP's own `eps` is the only gap.

**3. The ADMM objective-scaling trap, including the over-normalization
failure** (near-rail instance, `x0=[x_max−0.02, 0.02,−0.01,0.4,0,0]`; our
production `J(U*) = −1694.5062287`; declarative cvxpy/OSQP oracle,
`max_iter=500000`, `eps_abs=eps_rel=1e-9`, at three objective scales):

   | scale | status | iterations | max\|U_oracle−U_ours\| | objective gap |
   |---|---|---|---|---|
   | 1.0 (raw, `P_f` entries ~2.6e5), eps=1e-9, cap 500k | `user_limit` (**stalled at this cap**) | 500000 | 3.56×10⁻⁴ | −1.32×10⁻² |
   | 1×10⁻³ (shipped) | `optimal` | 121475 | 4.61×10⁻⁷ | 1.09×10⁻⁵ |
   | 3.826×10⁻⁶ (`1/max\|P_f\|` — "normalize all the way") | `optimal` (**falsely**) | 61775 | 3.578 | 4.671 |

   Row 1 is genuinely slow, not fundamentally infeasible — an
   investigation (same instance) pushed the cap to 1M at
   a slightly looser `eps=1e-8` and found scale=1.0 DOES eventually reach
   `"optimal"`, at **423225 iterations** (gap `7.31×10⁻⁷`, honestly tiny once
   it gets there) — a ~3.5× larger iteration budget than the shipped scale
   needs. Raw-scale costs (`~2.6e5`) make `eps_abs` (absolute) negligible
   relative to the objective, so ADMM must grind toward `eps_rel` almost
   unaided; short of that iteration count, the *unconverged* iterate can
   even look spuriously better in cost (`−1.3e-2` at 500k here) because ADMM
   iterates need not be feasible pre-convergence. Row 3: scaling
   all the way down to `O(1)` makes `eps_abs` **dominate** termination —
   OSQP reports `"optimal"` after only 61775 iterations, but the answer is
   off by **3.578 in raw force units** (physically meaningless — nowhere
   near this problem's ~10–150 N range) and 4.67 in true objective. Row 2 is
   the sweet spot chosen for `tests/test_mpc.py`: large enough that `eps_abs`
   never binds early, small enough that `eps_rel` alone gets there in a
   reasonable iteration count. **A solver's own "optimal" status is only as
   trustworthy as the scale it was told to converge at.**

**4. The oracle-stall lesson, arbitrated by a third, algorithmically
independent solver.** A harder near-rail-while-tracking instance
(`x0=[x_max−0.05, 0.02,−0.01,1.5,0,0]`, `z_ref=0.8` — the exact instance
`tests/test_mpc.py`'s own comment describes as stalling the oracle) was
re-run fresh, three ways:

   | solver | status | objective | gap vs ours (abs / rel) | max\|U − U_ours\| |
   |---|---|---|---|---|
   | our production (dense condensed, OSQP) | `optimal` | 22375.228273 | — | — |
   | cvxpy/OSQP declarative oracle (shipped config: scale 1e-3, `max_iter=200000`) | **`user_limit` (stalled)** | 22368.180435 | −7.048 / 3.15×10⁻⁴ | 3.75 |
   | cvxpy/CLARABEL declarative oracle (interior-point; same formulation) | `optimal` | 22374.9344 | −0.2939 / **1.313×10⁻⁵** | 0.904 |

   The shipped ADMM oracle genuinely does not converge on this instance
   within its 200k-iteration budget — and its unconverged iterate, rolled
   forward, **violates the rail by `9.30×10⁻⁵ m`** (measured directly;
   matches the existing code comment's `9.4×10⁻⁵` almost exactly, confirming
   the reproduction), i.e. it is not just slow, it is *infeasible*. CLARABEL
   — a different algorithm family (interior-point, not first-order ADMM) —
   converges cleanly and agrees with our production controller's objective
   to **1.3×10⁻⁵ relative**, ~24× tighter than the stalled oracle's own
   agreement. **The lesson an oracle only certifies the instances it can
   converge on** — bring in a third, independent solver when the usual one
   stalls, rather than concluding the thing under test is wrong.

   A bonus wrinkle, directly measured, that motivates the question
   below: comparing the two *oracles* to each other (OSQP-declarative vs
   CLARABEL-declarative, both nominally "done" — one stalled but still
   returns an iterate, one genuinely converged) gives `max|U_osqp −
   U_clarabel| = 3.26` (a big **raw** disagreement) alongside `|J_osqp −
   J_clarabel| = 6.75` (**3.0×10⁻⁴ relative** — comparatively tight). Even
   between our production solve and CLARABEL, both plausibly genuinely
   optimal, the objective agrees to `1.3×10⁻⁵` relative while the raw `U`
   vectors differ by up to `0.90` (in a ~10–150 N problem). **This is why
   you compare objective values across solver families, not raw iterates**:
   the QP's optimal *value* is unique; a near-flat cost direction (e.g. a
   redundant/degenerate active set right at a constraint boundary) can leave
   the optimal *point* only loosely pinned down, and two correct solvers can
   land in different corners of that flat region.

## The rail showdown (`tests/test_mpc_e2e.py`)

x0 search, documented (a first candidate `x0=(1.2, 0.05, 0, 0.3, 0, 0)`
measured **identical** for MPC and LQR — peak `|x_cart|=1.3674 m` for both,
nowhere near `x_max=1.4` — LQR's own regulate-to-origin law already arrests
that drift in time by luck of the numbers, so it does not separate them). A
grid sweep (`x_start∈[1.20,1.32]`, `v_start∈[0.28,0.45]`) found: push harder
and **both** fail (MPC's own 200 ms horizon cannot certify recovery either,
falling into sustained fallback); the genuinely-separating regime is a
narrow band where MPC's rail-aware horizon still finds a feasible plan and
LQR's rail-blind law does not back off in time.

**x0 = (1.24, 0.05, 0, 0.30, 0, 0)** — the same tip/velocity, cart
start nudged 4 cm closer to the wall:

| | peak\|x_cart\| | peak\|u\| | final ‖x‖ | fallback ticks |
|---|---|---|---|---|
| **MPC** | **1.40005 m** (within `x_max+1e-3`) | 10.862 N | 2.504×10⁻⁴ | 8/1600 (`"maximum iterations reached"`, not infeasible) |
| **LQR** (`u_max=150`) | **1.41089 m** (1.09 cm PAST `x_max`) | 11.626 N | 2.490×10⁻⁴ | n/a |

Both fully stabilize the pendulum; neither saturates `u_max` (peak ~11 N of
150 N available) — this is a **position/horizon-awareness** story, not a
torque-authority one. MPC starts braking earlier because its 200 ms
lookahead can already see the wall; LQR's instantaneous law reacts to state
error alone, with no notion that a wall exists. Mean/max per-tick solve time
(regulation run): **0.0862 ms / 0.2938 ms** — inside the `<2 ms` gate by
~23×/~7×, and inside the full 5 ms `ctrl_dt` budget by ~58×/~17×.

**What bound is actually being tested here — and what isn't.** Read
quickly, "LQR exceeds the bound MPC respects" can sound like MPC is dodging
a collision. It isn't: `x_max=1.4 m` is MPC's own SOFT, self-imposed safety
margin (`plant.rail − 0.1 m`), not the physical end-stop. The rail's actual
hard stops sit at `±1.5 m` and only exert a restoring force once
`|x|>1.5 m` (`k_stop=500 N/m`, `cart_dynamics.py`'s `stop_force`). Neither
controller gets anywhere near that physical wall here: MPC peaks at
`1.40005 m` (`9.995 cm` short of `1.5 m`), LQR at `1.41089 m` (`8.911 cm`
short). "LQR exceeds `x_max`" means LQR crossed MPC's chosen margin —
nothing more; read as "LQR nearly hits the rail," it would be false.

**Why 1.09 cm, and why that's expected, not weak evidence.** This x0 sits
in the both-stable, linear-basin regime: to recover the pendulum at all,
LQR's own position feedback is already braking the cart hard from
`x=1.24 m` (peak `|u|=11.626 N`, table above) — MPC's rail constraint only
has to add a HAIR of earlier braking on top of a trajectory that was
already being reeled in, which is why the gap is small. Pushing the IC
harder to widen that gap was tried empirically, not assumed: a 65-IC grid
(`x_start∈{1.24,1.28,1.32,1.36,1.40}`, `v_start∈{0.30,0.35,…,0.90}`, tip
held at `0.05` rad — measured) found exactly
**one** IC in the grid where both controllers stay upright: the documented
point itself. All other 64 lose at least one controller — and not always
the expected one. Just one step over (`x0=(1.24, 0.35)`, 5 cm/s faster), it
is MPC that fails FIRST: its 200 ms/`N=40` horizon can no longer certify a
feasible plan and falls into sustained fallback (final `‖x‖=11.8`, pendulum
lost), while LQR — despite a rail violation that has already grown to
`4.97 cm`, up from `1.09 cm` — still fully recovers (final
`‖x‖=2.59×10⁻⁴`). Push harder still and LQR fails too. The joint-stability
window here is a sliver thinner than this grid's own 4 cm/5 cm·s⁻¹
resolution: there is no IC in this search where LQR's violation grows
appreciably while BOTH controllers stay upright, so `1.09 cm` is close to
the ceiling this comparison can show, not a sign of under-searching.

**The stronger constraint demonstration is elsewhere.** This showdown is a
genuine differentiator (MPC provably keeps its promise; LQR provably
doesn't) but it's a braking-*timing* story — the rail barely has to do any
work. For the rail genuinely shaping a plan, see test (c),
`test_cart_mpc_tracks_with_rail_genuinely_active_near_target`: tracking to
`+0.8 m` with `x_max` tightened to `0.812 m`, MPC's trajectory rides the
wall for the whole approach (peak `0.8120007 m` — `0.7 µm` past `x_max`,
right at OSQP's own solver tolerance), not a fleeting graze. That's what a
binding constraint actually shaping a trajectory looks like, and it's the
better example of this controller's rail constraint doing real
work.

## Failure modes (current, documented, guarded)

- **Fallback under short-horizon infeasibility**: a hard constraint's
  promise only holds while the QP stays feasible within `N` steps; push the
  rail-showdown x0 harder and MPC also fails (see the search above) — this
  is a genuine limit of a 200 ms lookahead, not a bug.
- **The conditioning wall** (table above): `H` loses positive-
  semi-definiteness by `N~500` for this plant; the `1e-9·I` floor is
  insurance against roundoff, not a fix for a badly-chosen horizon.
- **Objective-scale mistuning** (numerics lesson 3): a poorly-scaled TEST
  oracle can silently report `"optimal"` on a wrong answer, or fail to
  converge on a right one — this is a test-infrastructure risk, not a
  production one (`dpend/` never uses `cvxpy`), but it is exactly the kind
  of validation-gate trap this repo's LQR note warned about generically
  ("a validation gate is only as honest as the numerics of the measurement
  behind it").
- **Saturation / fallback voids no *new* guarantee beyond what's
  documented**: like `LQRController`, the fallback law (`clip(−K_lqr x̃0,
  ±u_max)`) is not Lyapunov-certified — it exists so a control tick never
  returns NaN, not as a substitute stability proof.
- **No angle wrapping / small-angle regime**: same caveat as the LQR note;
  large excursions are swing-up territory.

## Final measured results

- **cond(H):** 1.626e2 (N=40) → 4.607e4 (N=100) → 1.338e20 (N=500, **H
  indefinite**) → 5.703e31 (N=800).
- **DARE exactness:** max|u_mpc−(−K_d x0)| = 1.371e-6 (20 random states).
- **K_d vs K_lqr** (cart): max per-entry relative gap 8.81%.
- **ADMM scale sweep** (near-rail): scale 1.0 stalls (`user_limit`, 500k
  iters); scale 1e-3 (shipped) converges cleanly (121475 iters, gap 1.09e-5);
  scale 3.826e-6 (over-normalized) falsely reports `"optimal"` (gap 4.67).
- **Oracle-stall / CLARABEL** (harder near-rail-tracking): OSQP-declarative
  oracle stalls (`user_limit`, rail violation 9.30e-5 m); CLARABEL converges,
  agrees with production to 1.313e-5 relative in objective while differing
  by 0.90 in raw `U`.
- **Warm-start benefit** (clean, same-QP-instance test): 1.60× fewer mean
  ADMM iterations (calm run), 1.34× (rail-showdown instance).
- **KKT dual, measured**: exactly 1/40 rail rows nonzero (the terminal
  step) at the tightest tracking tick, `y=56.98`; all box rows zero.
- **Rail showdown**: MPC peak|x_cart|=1.40005 m vs LQR 1.41089 m (1.09 cm
  past the same bound), from x0=(1.24, 0.05, 0, 0.30, 0, 0).
- **Solve time**: mean 0.0862 ms, max 0.2938 ms (ctrl_dt=5 ms budget).
