"""Linear MPC — receding-horizon regulator as a hand-condensed QP, solved
each control tick by OSQP (warm-started).

Hand-roll line: the formulation (condensing, QP matrices, constraints) is
numpy only. ``osqp`` is the solver; ``scipy.sparse`` is a data-format
adapter (OSQP's C core requires csc_matrix inputs). ``scipy.linalg.expm``
and ``cvxpy`` are test-only oracles (tests/test_mpc.py), never imported
here.

Prediction model: continuous (A, B) from ``plant.linearize(plant.upright)``,
exactly ZOH-discretized (``c2d_zoh``) at the control tick:

    x_{k+1} = A_d x_k + B_d u_k,    A_d ∈ ℝ^{n×n},  B_d ∈ ℝ^{n×m}

Tracking shift (railed plants only): the rail is a translation-invariant
DOF, so re-centering the predicted origin on a target z_ref along that one
coordinate is an EXACT equilibrium shift, not an approximation. Everything
below predicts the error state x̃ = x − z_ref (z_ref = 0 ⇒ x̃ ≡ x).

Condensing — eliminate the state sequence, leaving a QP in the inputs
alone. With X̃ = [x̃_1; …; x̃_N] and U = [u_0; …; u_{N-1}] (x̃_0 is given,
not a decision variable), unrolling the recursion gives X̃ = Sx x̃_0 + Su U:

    Sx ∈ ℝ^{N·n × n}:    block row i = A_d^{i+1}
    Su ∈ ℝ^{N·n × N·m}:  block (i, j) = A_d^{i−j} B_d if j ≤ i, else 0
                         (lower-block-triangular — causality)

Cost: per-step weights Q, R (from ``plant.lqr_weights``, the same tuple
``lqr_factory`` reads — LQR and the MPC terminal cost provably use
identical numbers) and terminal weight P_f:

    J(U) = ½ X̃ᵀ Q̄ X̃ + ½ Uᵀ R̄ U,  Q̄ = blkdiag(Q,…,Q, P_f),  R̄ = blkdiag(R,…,R)

(x̃_0's Q-cost is a constant — it cannot change the argmin and is omitted.)
Substituting X̃(U) gives OSQP's standard form ½ UᵀHU + fᵀU with

    H = SuᵀQ̄Su + R̄   (constant — built once at construction)
    f = SuᵀQ̄Sx x̃_0   (the only state-dependent term — recomputed every tick)

H is symmetrized (rounding only) then regularized with +1e-9·I. Not a rank
fix — R̄ ≻ 0 already makes H ≻ 0 in exact arithmetic; it is conditioning
insurance: A_d is open-loop unstable, so Su's late-horizon blocks grow
exponentially and cancellation forming SuᵀQ̄Su can erode R̄'s 0.1-scale PD
floor. 1e-9 is ~1e-8 relative to H — far below every tested tolerance.

Conditioning cliff (measured — do not lengthen the horizon blindly):
cond(H) grows explosively with N on this open-loop-unstable plant — 1.6e2
at the default N=40, 4.6e4 at N=100, >1e13 (unusable by OSQP or a dense
solve) by N≈300-400. Condensed MPC is a poor fit for long horizons on
unstable plants; a sparse KKT formulation never forms SuᵀQ̄Su and avoids
this. tests/test_mpc.py's conditioning regression pins the N=40 default.

Terminal cost P_f = ``solve_dare(A_d, B_d, Q, R)``: the DARE solution is
the fixed point of the finite-horizon backward Riccati recursion, so
seeding with it makes every stage's cost-to-go exactly P* — unconstrained
finite-N MPC ≡ infinite-horizon discrete LQR (gain K_d) for ANY N ≥ 1, a
theorem, tested to solver tolerance. It is also the canonical stability
choice (Rawlings, Mayne & Diehl: terminal cost as control-Lyapunov function
under the terminal LQR law). A continuous-CARE P_f was tried first and
measured 6-9% off per gain entry — dt·|λ_fast| ≈ 0.08 is not fast-sampling
here; see docs/design-notes/linear-mpc.md. Scale note: the DARE P
sums undiscounted per-tick costs, so ‖P_dare‖ ≈ ‖P_care‖/dt ≈ 200× here;
the UI's V-supervisor is unaffected (V and its trigger level both come from
the same controller-exposed P, so the comparison is scale-invariant).

Constraints, in OSQP's ``l ≤ A_c U ≤ u`` form:

  Box rows (every plant): I_{N·m} U, bounds ±u_max at every predicted step.

  Rail rows (railed plants only — omitted entirely otherwise): the
  cart-position rows of Su bound the ABSOLUTE position x_k = x̃_k + z_ref —
  the wall is fixed in the world and does not move with the tracking
  target, so with pos = the cart-position rows:

      −x_max − z_ref − (Sx x̃_0)_pos  ≤  (Su U)_pos  ≤  x_max − z_ref − (Sx x̃_0)_pos

  Bounding x̃_k directly would be correct only at z_ref = 0 — the easiest
  subtle bug in this formulation; pinned by a dedicated tracking test.

A_c and H never change after construction (``setup()`` once); only f, l, u
are recomputed per tick (``update()``), then a warm start — the previous
solution shifted one step, last block repeated — and ``solve()``. Fallback
on any non-solved status: the shifted previous plan's first entry if one
exists, else ``clip(−K_lqr x̃_0, ±u_max)``. Always defined, never NaN — but
not Lyapunov-certified (same caveat as LQRController's clip).
"""
from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse

from dpend.controllers.discretize import c2d_zoh
from dpend.controllers.riccati import dlqr_gain, solve_care, solve_dare
from dpend.model.linearize import ctrb, rank_and_cond
from dpend.reference import ReferenceSource

_OSQP_OK_STATUSES = ("solved", "solved inaccurate")


def _block_diag(mats: list[np.ndarray]) -> np.ndarray:
    """Hand-rolled block-diagonal stack (numpy only — no scipy.linalg.block_diag
    in dpend/). mats: list of 2-D arrays (any shapes); result is
    (sum(rows), sum(cols)) with each mat placed on the diagonal, zero elsewhere."""
    rows = sum(mat.shape[0] for mat in mats)
    cols = sum(mat.shape[1] for mat in mats)
    out = np.zeros((rows, cols))
    r = c = 0
    for mat in mats:
        rr, cc = mat.shape
        out[r:r + rr, c:c + cc] = mat
        r += rr
        c += cc
    return out


def _condense(A_d: np.ndarray, B_d: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the stacked-prediction matrices Sx (N·n, n), Su (N·n, N·m) from
    A_d powers (see module docstring, "Condensing"). powers[k] = A_d^k."""
    n, m = A_d.shape[0], B_d.shape[1]
    powers = [np.eye(n)]
    for _ in range(N):
        powers.append(powers[-1] @ A_d)

    Sx = np.vstack(powers[1:])  # block i = A_d^{i+1}, i=0..N-1
    Su = np.zeros((N * n, N * m))
    for i in range(N):
        for j in range(i + 1):
            Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = powers[i - j] @ B_d
    return Sx, Su


def _shift(U: np.ndarray, m: int) -> np.ndarray:
    """Receding-horizon warm-start/fallback shift: drop the first m-block,
    keep the rest, repeat the last m-block as the new final entry. Same
    length as U (N·m)."""
    shifted = np.empty_like(U)
    shifted[:-m] = U[m:]
    shifted[-m:] = U[-m:]
    return shifted


class MPCController:
    """Condensed linear MPC: u = first block of argmin_U ½UᵀHU + fᵀU s.t.
    l ≤ A_c U ≤ u, re-solved every tick, warm-started. Derivation: module
    docstring.

    Public frozen API (other components read these):
      K_d: (m,n) discrete LQR gain from the DARE P — the gain the
        exact-equivalence theorem holds for.
      K_lqr: (m,n) continuous CARE gain — cross-reference and last-resort
        fallback law (measured <10% per entry from K_d, the O(dt·|λ_fast|)
        discretization gap).
      P: (n,n) DARE terminal cost — read by the UI's V-supervisor, which
        computes both V(x) and its trigger from this same matrix (its ~1/dt
        scale vs the LQR controller's P is therefore self-consistent).
      status: "optimal" | "fallback" — the LAST solve's outcome; "optimal"
        before the first tick.

    A_d, B_d, Sx, Su, H, N, n, m, ctrl_dt are public for test/telemetry
    inspection only — not part of the frozen contract above.

    Principal-branch precondition: ``update()`` shifts x̂ by z_ref with a
    raw subtraction, no angle wrapping — angle components of x̂ must already
    be within ±π of upright (ModeSwitch re-wraps before every catch call;
    any other caller on a wound angle must wrap first).
    """

    def __init__(self, *, A_d: np.ndarray, B_d: np.ndarray, Q: np.ndarray,
                 R: np.ndarray, P_f: np.ndarray, K_lqr: np.ndarray,
                 K_d: np.ndarray, N: int,
                 u_max: np.ndarray, x_max: float | None, z_ref_fn, n: int,
                 m: int, ctrl_dt: float):
        self.n, self.m, self.N, self.ctrl_dt = n, m, N, ctrl_dt
        self.A_d = np.asarray(A_d, dtype=float)
        self.B_d = np.asarray(B_d, dtype=float)
        self.P = np.asarray(P_f, dtype=float)          # public: DARE terminal cost / Lyapunov V(x)=xᵀPx
        self.K_lqr = np.asarray(K_lqr, dtype=float)    # public: continuous LQR gain, fallback law
        self.K_d = np.asarray(K_d, dtype=float)        # public: discrete LQR gain (exact equivalence)
        self.status = "optimal"                        # public: meaningful after >=1 update()

        self._u_max = np.asarray(u_max, dtype=float)   # (m,)
        self._x_max = x_max                            # float or None (no rail rows)
        self._z_ref_fn = z_ref_fn                       # t -> scalar rail target, or None
        self._U_prev: np.ndarray | None = None          # warm-start/fallback memory

        self.Sx, self.Su = _condense(self.A_d, self.B_d, N)

        Qbar = _block_diag([Q] * (N - 1) + [self.P])   # blkdiag(Q,...,Q [N-1], P_f)
        Rbar = _block_diag([R] * N)                     # blkdiag(R,...,R  [N])
        self._Qbar = Qbar

        H = self.Su.T @ Qbar @ self.Su + Rbar
        H = 0.5 * (H + H.T) + 1e-9 * np.eye(N * m)      # symmetrize (roundoff only) + conditioning floor
        self.H = H  # public (inspection): the Hessian exactly as handed to OSQP

        A_box = np.eye(N * m)
        if self._x_max is not None:
            pos_rows = [i * n for i in range(N)]        # cart position = state index 0, every step
            self._Su_pos = self.Su[pos_rows, :]          # (N, N·m)
            self._Sx_pos = self.Sx[pos_rows, :]          # (N, n)
            A_c = np.vstack([A_box, self._Su_pos])
        else:
            self._Su_pos = None
            self._Sx_pos = None
            A_c = A_box

        # OSQP setup once: structure (H, A_c) is fixed for this controller's
        # life — only q/l/u change per tick.
        self._osqp = osqp.OSQP()
        l0, u0 = self._bounds(np.zeros(n), 0.0)
        self._osqp.setup(P=sparse.csc_matrix(H), q=np.zeros(N * m),
                          A=sparse.csc_matrix(A_c), l=l0, u=u0,
                          eps_abs=1e-6, eps_rel=1e-6, polishing=True, verbose=False)

    def _bounds(self, x_tilde0: np.ndarray, z_ref: float) -> tuple[np.ndarray, np.ndarray]:
        """(l, u) for l ≤ A_c U ≤ u at the current x̃_0/z_ref (module
        docstring, "Constraints"). Box bounds are constant; rail bounds (if
        any) are recomputed from x̃_0/z_ref every call."""
        l = np.tile(-self._u_max, self.N)
        u = np.tile(self._u_max, self.N)
        if self._Su_pos is None:
            return l, u
        pred_pos = self._Sx_pos @ x_tilde0  # (N,): (Sx x̃0)_pos-rows
        l_rail = -self._x_max - z_ref - pred_pos
        u_rail = self._x_max - z_ref - pred_pos
        return np.concatenate([l, l_rail]), np.concatenate([u, u_rail])

    def reset(self, t0: float, x0: np.ndarray) -> None:
        """Clear warm-start/fallback memory — a fresh episode's plan has no
        relationship to the old one's — and return status to "optimal" (it
        reports the last solve; a fresh episode has none)."""
        self._U_prev = None
        self.status = "optimal"

    def update(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        x0 = np.asarray(x_hat, dtype=float)
        z_ref = float(self._z_ref_fn(t)) if self._z_ref_fn is not None else 0.0
        x_tilde0 = x0.copy()
        x_tilde0[0] -= z_ref  # rail coordinate only; no-op (z_ref≡0) off the rail

        f = self.Su.T @ (self._Qbar @ (self.Sx @ x_tilde0))
        l, u = self._bounds(x_tilde0, z_ref)

        self._osqp.update(q=f, l=l, u=u)
        if self._U_prev is not None:
            self._osqp.warm_start(x=_shift(self._U_prev, self.m))
        # raise_error=False: a solver failure must surface as a status
        # string, not an exception (osqp 1.1.3 warns the default will flip).
        # The try/except is the second layer of the same guarantee: the tick
        # must reach the fallback branch, never crash.
        try:
            res = self._osqp.solve(raise_error=False)
            solved = res.info.status in _OSQP_OK_STATUSES
        except Exception:
            res = None
            solved = False

        if solved:
            self.status = "optimal"
            U_star = np.asarray(res.x, dtype=float)
            self._U_prev = U_star
            return U_star[:self.m].copy()

        self.status = "fallback"
        if self._U_prev is not None:
            shifted = _shift(self._U_prev, self.m)
            self._U_prev = shifted
            return shifted[:self.m].copy()

        # Last resort: the continuous gain, clipped (K_lqr vs K_d differ by
        # the ~8% discretization gap — immaterial for a clipped emergency action).
        u_cmd = np.clip(-self.K_lqr @ x_tilde0, -self._u_max, self._u_max)
        self._U_prev = np.tile(u_cmd, self.N)
        return u_cmd


def mpc_factory(plant, params: dict) -> MPCController:
    """Build an MPCController for `plant` from scenario `params`.

    1. Linearize at upright; gate on controllability (the Riccati solves
       need a stabilizable pair).
    2. Q/R: `plant.lqr_weights` defaults (same source as lqr_factory),
       overridable via params["Q"]/["R"].
    3. Discretize first (the DARE needs the discrete pair):
       ctrl_dt = params.get("ctrl_dt", 5e-3), N = params.get("N", 40).
       ctrl_dt and N are coupled — the lookahead is N·ctrl_dt (200 ms at
       the defaults) and this factory does not auto-adjust N; set both to
       keep a fixed horizon duration.
    4. P_f = solve_dare(A_d, B_d, Q, R); K_d = dlqr_gain(...). K_lqr =
       R⁻¹BᵀP_c from solve_care — continuous cross-reference/fallback gain
       (P_c itself is not retained; the public P is the DARE P).
    5. u_max = params.get("u_max", 150.0) [N or N·m per plant.input_labels].
    6. x_max: railed plants only, default plant.rail − 0.1 m margin;
       ignored on plants with no rail coordinate to constrain.
    7. Tracking: params.get("reference") or ReferenceSource(), railed
       plants only; z_ref_fn returns the scalar rail target (not
       lqr_factory's e1-vector).
    """
    A, B = plant.linearize(plant.upright)

    rank, cond = rank_and_cond(ctrb(A, B))
    if rank != plant.n:
        raise ValueError(
            f"mpc_factory: {plant.name} plant not controllable at upright — "
            f"rank(ctrb(A,B))={rank} != n={plant.n} (cond={cond:.3e}); the "
            "terminal-cost Riccati solves require a stabilizable (A,B) pair"
        )

    Q_diag = params.get("Q") or plant.lqr_weights[0]
    R_diag = params.get("R") or plant.lqr_weights[1]
    Q = np.diag(np.asarray(Q_diag, dtype=float))
    R = np.diag(np.asarray(R_diag, dtype=float))

    ctrl_dt = params.get("ctrl_dt", 5e-3)
    N = params.get("N", 40)
    A_d, B_d = c2d_zoh(A, B, ctrl_dt)

    P_f = solve_dare(A_d, B_d, Q, R)          # DARE terminal cost — the exact-equivalence choice
    K_d = dlqr_gain(A_d, B_d, Q, R, P_f)      # discrete LQR gain (public: the equivalence gain)
    P_c = solve_care(A, B, Q, R)              # continuous CARE — only to form K_lqr below
    K_lqr = np.linalg.solve(R, B.T @ P_c)     # continuous cross-reference + fallback gain

    u_max = np.broadcast_to(
        np.atleast_1d(np.asarray(params.get("u_max", 150.0), dtype=float)), (plant.m,)
    ).copy()

    x_max = params.get("x_max", (plant.rail - 0.1) if plant.rail is not None else None)
    z_ref_fn = None
    if plant.rail is not None:
        ref = params.get("reference") or ReferenceSource()  # self-default (mirrors lqr_factory)
        z_ref_fn = lambda t: ref.r(t)  # noqa: E731 - scalar rail target, not lqr_factory's e1-vector
    else:
        x_max = None  # no cart-position state on this plant; ignore any override

    return MPCController(
        A_d=A_d, B_d=B_d, Q=Q, R=R, P_f=P_f, K_lqr=K_lqr, K_d=K_d, N=N,
        u_max=u_max, x_max=x_max, z_ref_fn=z_ref_fn, n=plant.n, m=plant.m,
        ctrl_dt=ctrl_dt,
    )
