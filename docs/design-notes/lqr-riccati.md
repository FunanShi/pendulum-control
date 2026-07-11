# Design note — LQR via a hand-rolled Riccati (CARE) solver

*The decision, the alternatives, why, and the failure modes — including the numerical
wall hit on the way. Every number below was measured, not asserted.*

## The decision, in one paragraph

`dpend/controllers/riccati.py` solves the continuous algebraic Riccati
equation `AᵀP + PA − PBR⁻¹BᵀP + Q = 0` **by hand, in pure numpy**, using the
**Hamiltonian eigenvector method seeded into a Newton–Kleinman polish**, with
a hard validation gate (exactly n stable Hamiltonian eigenvalues; P symmetric
positive-definite; Frobenius-relative residual < 1e-8 — violations raise, never
return). `scipy.linalg.solve_continuous_are` appears only in tests, as the
independent oracle. `K = R⁻¹BᵀP`; `u = −K(x̂ − z_ref)`.

## Why hand-rolled at all

The Riccati solution is the piece of this project worth understanding from the
inside, so the library belongs in the oracle layer (tests), not the hot path.
That choice also forced the numerics decisions below, which turned out to be the
most instructive part.

## Why the Hamiltonian eigenvector method (and its known weakness)

The textbook-robust CARE route is an **ordered Schur decomposition** of the
Hamiltonian `H = [[A, −BR⁻¹Bᵀ], [−Q, −Aᵀ]]` (Laub's method). numpy has no
ordered Schur; scipy's is off-limits in the production path; hand-rolling QZ
reordering is numerical-software engineering beyond this project's scope. The
eigenvector method — take the n stable eigenvectors `[X₁; X₂]` of H, form
`P = Re(X₂X₁⁻¹)` — needs only `np.linalg.eig` and is exact in exact
arithmetic. Its documented weakness: accuracy degrades as `cond(X₁)` grows,
i.e. when closed-loop eigenvalues approach degeneracy (eigenvectors nearly
parallel).

## The wall (measured)

With the factory's default weights, the two plants behaved differently:

| | cart (n=6) | fixed acrobot (n=4), default Q=diag(10,10,1,1), R=0.1 |
|---|---|---|
| closed-loop eig gap | comfortable | **0.135 s⁻¹** (near-degenerate pair) |
| cond(X₁) | 1.3×10³ | **9.6×10³** |
| eigenvector-method residual | 1.7×10⁻¹¹ ✓ | **2.014×10⁻⁸ — GATE FIRED** (limit 1e-8) |
| P vs scipy oracle | 3.2×10⁻¹¹ | still ~7×10⁻¹² *relative* (answer fine!) |

Three independent checks confirmed **no bug**: the computed P matched the
oracle; the residual excess tracked cond(X₁)·ε; and perturbing Q moved the
eigenvalue gap and the residual together. The solver was correctly refusing to
bless a result computed at its numerical edge. **The gate did exactly its job**
— and the right response was to repair the numerics, not to loosen the gate,
which is the verify-don't-assert norm working as process rather than only as
tests.

**Second root cause (found while installing the fix):** the residual's
*evaluation form* matters as much as the solve. Computing
`AᵀP + PA − P(BR⁻¹Bᵀ)P + Q` with the explicit `P·S·P` product has a
floating-point noise floor near 1e-8 at this plant's scaling — **even scipy's
reference P measured 4.2×10⁻⁹ in that form** — which made the converging
polish look like it was bouncing. Re-expressing the identical quantity via
`W = PB` (then `AᵀP + PA − WR⁻¹Wᵀ + Q`) drops the floor by ~2 orders. Formula
and 1e-8 gate unchanged; only the arithmetic ordering. Lesson: a validation
gate is only as honest as the numerics of the *measurement* behind it.

## The fork (four options, one chosen)

1. **Loosen the gate to 1e-6.** Would still catch subspace-selection bugs
   (they err by orders of magnitude) — but papers over the conditioning story
   instead of engaging it. Weakest position. *Rejected.*
2. **Condition-aware gate** (tolerance ∝ cond(X₁)·ε). Principled — that *is*
   the method's attainable floor — but accepts degraded accuracy rather than
   repairing it. *Rejected in favor of repair.*
3. **Retune the fixed plant's default Q/R** away from the degeneracy. Chooses
   control weights to dodge a solver weakness, and silently changes the acrobot
   scenario's documented behavior. Backwards. *Rejected.*
4. **Newton–Kleinman polish** (Kleinman 1968): keep the eigenvector result as
   a *seed*, then iterate `K_k = R⁻¹BᵀP_k`;
   `(A−BK_k)ᵀP_{k+1} + P_{k+1}(A−BK_k) = −(Q + K_kᵀRK_k)` — each step one
   **Lyapunov solve**, done in pure numpy by Kronecker vectorization
   (`(I⊗Aᵀ + Aᵀ⊗I)vec(P) = −vec(S)`). Given a *stabilizing* seed, the
   iteration is monotone and quadratically convergent: it repairs
   conditioning-amplified error to ~machine precision, so the 1e-8 gate stays
   honest on every plant. This eigenvector-seed + Newton-polish pairing is the
   classic industrial arrangement. ***Chosen.***

## Failure modes (documented, guarded)

- **Non-stabilizing seed** (truly defective/uncontrollable H): Kleinman
  requires a stabilizing start; the solver asserts the seed's closed loop is
  Hurwitz and raises with diagnostics otherwise.
- **Polish non-convergence**: capped at 5 iterations; raises with the residual
  history rather than returning an unvalidated P.
- **Saturation** (`params["u_max"]`): clipping voids the Lyapunov guarantee
  `V̇ = −xᵀ(Q+KᵀRK)x < 0`; off by default, documented where enabled.
- **Tracking validity**: `u = −K(z − [r,0,…])` is *exact* only on plants with
  a translation-invariant DOF (the cart's x — gravity independent of x, tested
  as the equilibrium family). The fixed plant ignores the reference.
- **No angle wrapping**: linear-regime controller; large excursions are
  swing-up territory, where `wrap_to_pi` and region-of-attraction logic enter.

## Final measured results (Newton–Kleinman in place)

- **Fixed acrobot:** NK residual history `2.801e-08 → 9.439e-11` — converged
  in **one** iteration (quadratic convergence doing exactly what the theory
  says); independently-recomputed residual 1.093e-10; Hurwitz margin
  **2.688 s⁻¹**; P-vs-scipy max delta 4.426e-07 abs (19× inside the per-entry
  allclose margin — and this P's residual is ~6× *better* than scipy's own).
- **Cart:** residual 3.2e-11 class, Hurwitz margin **1.297 s⁻¹**, P-vs-scipy
  delta 3.229e-11.
- Random stabilizable systems (n=2..6, seeded): oracle deltas 7e-15 → 1.1e-9.
- **e2e settle times (closed-loop `simulate()`, standard rates):** cart
  regulation (x0=(0,0.15,−0.10,0,0,0), `scenarios/cart_lqr.py`) settles
  ‖z‖<0.01 at **3.635 s** (final ‖z‖=6.3e-5, |u|<1e-3 N by 7.06 s); cart
  tracking (+0.8 m rail target from the origin) settles within 2% of target
  at **3.055 s** (final angles ~1e-5 rad); fixed acrobot at the
  measured-convergent RoA edge (x0=(0.02,−0.01,0,0)) settles ‖x‖<0.01 at
  **2.66 s** (final ‖x‖=1.0e-8) — the canonical 0.10 rad tip, by contrast,
  crosses ‖x‖>1 by **5 ms** and never returns (blows past ‖x‖~1500 by
  0.12 s, non-finite by 0.75 s), pinning the RoA boundary this note's
  "Bonus finding" describes from both sides.
- **K excerpt** (factory-default Q/R): cart
  `K ≈ [10.0, −122.9, −384.7, 17.9, −81.6, −73.7]` (N, per
  [x, θ1, θ2, ẋ, θ̇1, θ̇2]); fixed acrobot
  `K ≈ [−683.5, −178.1, −251.5, −74.6]` (N·m, per [θ1, θ2, θ̇1, θ̇2]).

**Precision, precisely** (is this P more accurate than scipy's?): the polish
drives the **residual — the backward error — to ~1e-10** (extended-precision
truth: 6.6e-11); it does *not* make P "machine precision." The 4.4e-7 P-vs-scipy
gap is the **conditioning ball**: every solution with a sub-gate residual lives
within roughly `cond(P)·ε·‖P‖ ≈ 5×10⁵ · 2×10⁻¹⁶ · 10⁴ ≈ 10⁻⁶` of every other —
this P and scipy's are two points inside the same ball, and *neither* is
meaningfully "closer to the true P" than that radius. Backward error is the
honest claim; forward error is bounded by conditioning, not by the algorithm.

## Bonus finding — the acrobot's region of attraction (measured)

With canonical weights, the **linear** closed loop converges from a 0.10 rad
tip — but the **nonlinear** plant diverges from it, and converges only from
tips ≲ **0.02 rad**. Not a controller bug (the linear analysis is verified
above); it is the real, small RoA of an elbow-only acrobot near its unstable
equilibrium — the same plant whose controllability condition number (6.4×10³)
was the worst of the three configurations. This is the motivating fact for the
swing-up controller: energy swing-up exists precisely because linear catch
basins are this small, and the mode-switch design (`xᵀPx < c` trigger) needs
exactly this kind of measured boundary. The scenario's x0 is set to a
measured-convergent tip, with the RoA boundary documented at the call site.

## Can LQR lift the pendulum? (measured experiment)

Tip sweep on the cart plant, factory weights, 15 s each: **converges from
0.10/0.20/0.30 rad** (‖z‖→1e-8; the 0.30 rad catch uses 1.42 m of the ±1.5 m
rail); **diverges from 0.50 rad onward** — so the cart's RoA boundary lies in
**(0.30, 0.50) rad**, ~10× the acrobot's, still nowhere near π. From exact
hanging: final θ₁ = +57.9 rad (**nine revolutions of wind-up** — unwrapped
error grows once the pendulum goes over the top, a positive feedback of
wrongness), peak |u| = 59 kN, and the cart dragged **105 m** — the unsaturated
LQR out-muscles the 500 N/m end-stop and breaks the rail fiction entirely.
Lessons: (1) linear validity is the binding constraint, not authority — LQR
had 59 kN and no idea what to do with it; swing-up is an energy-phase problem
(the ~39 J hanging→upright gap) that the energy-shaping controller addresses;
(2) the linear MPC shares the linearization and won't lift it either — what it
adds is *constraint handling* (rail, u_max) inside the optimization;
(3) UI caveat: in LQR mode a >0.4 rad disturbance triggers the same blow-through
live — optional `u_max` mitigation documented below.

**Live confirmation + supervisor decision:** a real live drag released at
θ₁=0.355 rad diverged — the same wind-up / 27 m rail-blowout failure this
note's UI caveat predicted — refining the boundary from (0.30, 0.50) to
**(0.30, 0.355) rad**. Fix, two parts: `u_max=150 N` now bounds control effort
UI-wide by default (was the "unapplied" mitigation above), and a live RoA
supervisor disengages to MANUAL whenever `V(x)=xᵀPx` exceeds `V_lim`, V
evaluated once at the single measured-convergent tip (θ₁=0.30 rad, all else 0)
— the tightest point this note can defend. Caveat, stated once and left
un-softened: the true basin is not exactly a V-sublevel set, so this trigger is
a conservative UI safeguard, not a certified region of attraction.
