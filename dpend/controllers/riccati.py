"""Hand-rolled algebraic Riccati solvers: the continuous CARE (LQR) and the
discrete DARE (MPC's terminal cost), plus the discrete gain ``dlqr_gain``.

The CARE: for the stabilizing P ⪰ 0,

    AᵀP + PA − P B R⁻¹ Bᵀ P + Q = 0                                   (CARE)

solved by a Hamiltonian-eigenvector SEED then a Newton–Kleinman POLISH
(Kleinman, IEEE T-AC 1968).

Seed: stacking state and costate ζ = [x; λ] turns the infinite-horizon LQ
necessary conditions into ζ̇ = Hζ with the Hamiltonian

    H = [[ A, −B R⁻¹ Bᵀ ],
         [ −Q, −Aᵀ      ]]                              (2n × 2n)

whose spectrum is symmetric about the imaginary axis: for (A, B)
stabilizable and (A, Q) detectable, exactly n eigenvalues are stable. The
optimal costate is λ = Px on H's stable invariant subspace, so with that
subspace's basis stacked as [X1; X2], P₀ = Re(X2 X1⁻¹), symmetrized —
both Re(·) and the symmetrization strip rounding only (conjugate pairs
make a real basis possible; the exact P is symmetric).

Polish: the eigenvector construction is textbook, not numerically robust —
X1⁻¹ amplifies rounding when stable eigenvalues nearly repeat. Measured
here: the fixed plant's defaults put a closed-loop pair 0.135 rad/s apart,
inflating cond(X1) to ~1e4 and the seed's residual to ~2e-8, over this
module's 1e-8 gate, even though the subspace itself was right. Kleinman's
Newton iteration repairs it: given a stabilizing K_k = R⁻¹BᵀP_k, solve the
Lyapunov equation

    (A − B K_k)ᵀ P_{k+1} + P_{k+1} (A − B K_k) = −(Q + K_kᵀ R K_k)

— exactly Newton's method on the CARE residual, so quadratically
convergent, every iterate symmetric, stabilizing, monotone. One step
measured 2.8e-8 → 9.4e-11; if the seed already passes, zero polish steps
run. (Ordered real Schur is the seed-robustness upgrade if ever needed.)

Residuals are evaluated cancellation-safely as W = PB then WR⁻¹Wᵀ: the
naive P@(BR⁻¹Bᵀ)@P forms O(‖P‖²) intermediates whose cancellation carries
a noise floor at this module's own gate (~1e-8 at this plant's scaling);
the W-form is ~100× quieter (measured 7.3e-9 → 9.4e-11 on one identical P).

Validation (raises ValueError, never returns a silently-wrong P): exactly n
stable Hamiltonian eigenvalues; X1 invertible; a stabilizing seed (the
Kleinman precondition — the iteration preserves stabilizing iterates, it
cannot create one); polish under the 1e-8 relative-residual gate within 5
iterations (healthy solves need 0–2); P symmetric PD; and a final residual
re-check so no code path returns an unvalidated P.

Hand-roll line: numpy only — scipy is the test oracle, never a production
dependency.
"""
from __future__ import annotations

import numpy as np


def _solve_lyapunov(A_cl: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Solve the continuous Lyapunov equation  A_clᵀ P + P A_cl = −S  for P.

    Kronecker vectorization, column-stacking vec (numpy order='F'):
    vec(MXN) = (Nᵀ ⊗ M) vec(X) ⇒ (Iₙ ⊗ A_clᵀ + A_clᵀ ⊗ Iₙ) vec(P) = −vec(S).
    (The transpose convention — A_clᵀP + PA_cl, not A_clP + PA_clᵀ — is
    pinned by a hand-derived 2×2 case in tests/test_lqr.py.)

    Unique solution iff λ_i + λ_j ≠ 0 for all pairs — guaranteed for Hurwitz
    A_cl, which is how the Kleinman iteration always calls it. Dense O(n⁶):
    fine at n ≤ 6 (Bartels–Stewart is the large-n upgrade). S symmetric ⇒ P
    symmetric; symmetrized to strip rounding.
    """
    A_cl = np.asarray(A_cl, dtype=float)
    S = np.asarray(S, dtype=float)
    n = A_cl.shape[0]
    I_n = np.eye(n)
    M = np.kron(I_n, A_cl.T) + np.kron(A_cl.T, I_n)  # (n², n²)
    vec_P = np.linalg.solve(M, -S.flatten(order="F"))
    P = vec_P.reshape((n, n), order="F")
    return (P + P.T) / 2.0


def solve_care(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
               *, info: dict | None = None) -> np.ndarray:
    """Solve AᵀP + PA − P B R⁻¹ Bᵀ P + Q = 0 for the stabilizing P ⪰ 0
    (method and validation: module docstring).

    A: (n,n). B: (n,m). Q: (n,n) symmetric PSD (PD in every use here).
    R: (m,m) symmetric PD. Unit-generic — carries whatever units (A,B,Q,R)
    encode.

    info: optional dict, filled with diagnostics (no behavior change):
    "iterations" (0 = seed already under the gate), "residual_history"
    (seed first), "seed_closed_loop_max_re".

    Returns P (n,n), symmetric positive definite. Raises ValueError on any
    validation failure.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    n = A.shape[0]
    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError(f"A must be square (n,n), got shape {A.shape}")
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError(f"B must have {n} rows to match A, got shape {B.shape}")
    if Q.shape != (n, n):
        raise ValueError(f"Q must be ({n},{n}) to match A, got shape {Q.shape}")

    Rinv = np.linalg.inv(R)
    S = B @ Rinv @ B.T  # (n,n): the Hamiltonian's −BR⁻¹Bᵀ block, reused in the residual
    H = np.block([[A, -S],
                  [-Q, -A.T]])  # (2n,2n) Hamiltonian

    eigvals, eigvecs = np.linalg.eig(H)
    stable = eigvals.real < 0.0
    n_stable = int(np.count_nonzero(stable))
    if n_stable != n:
        n_unstable = int(np.count_nonzero(eigvals.real > 0.0))
        n_marginal = 2 * n - n_stable - n_unstable
        raise ValueError(
            f"Hamiltonian has {n_stable} stable / {n_unstable} unstable / "
            f"{n_marginal} marginal (Re≈0, neither <0 nor >0) eigenvalues out "
            f"of {2 * n}; expected exactly {n} stable. (A,B) is likely not "
            f"stabilizable, or (A,Q) not detectable. Eigenvalues (Re, sorted): "
            f"{np.sort(eigvals.real)}"
        )

    V_stable = eigvecs[:, stable]
    X1, X2 = V_stable[:n, :], V_stable[n:, :]
    try:
        X1_inv = np.linalg.inv(X1)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the stable invariant subspace's X1 block is singular — cannot "
            "form P = X2 X1⁻¹; H is (near-)defective for this (A,B,Q,R) "
            "(see the conditioning caveat in this module's docstring)"
        ) from exc
    P = np.real(X2 @ X1_inv)
    P = (P + P.T) / 2.0  # symmetrize away floating-point asymmetry only

    # Kleinman precondition: the iteration preserves stabilizing iterates —
    # it cannot repair a non-stabilizing seed, so gate here.
    K0 = np.linalg.solve(R, B.T @ P)
    seed_cl_eigs = np.linalg.eigvals(A - B @ K0)
    seed_max_re = float(np.max(seed_cl_eigs.real))
    if seed_max_re >= 0.0:
        raise ValueError(
            f"eigenvector seed P₀ is not stabilizing: max Re eig(A − B R⁻¹Bᵀ P₀) "
            f"= {seed_max_re:.3e} ≥ 0 (closed-loop eigenvalues: "
            f"{np.sort_complex(seed_cl_eigs)}). Newton–Kleinman requires a "
            "stabilizing seed — the stable subspace was mis-selected or H is "
            "too ill-conditioned (see this module's docstring)"
        )

    # Newton–Kleinman polish: quadratic convergence — a healthy solve exits
    # in 0–2 steps (0 = seed already under the gate, returned untouched).
    norm_Q = np.linalg.norm(Q, ord="fro")

    def _rel_residual(Pk: np.ndarray) -> float:
        # Quadratic term as (PB)R⁻¹(PB)ᵀ, not P@S@P — the cancellation-safe
        # evaluation order (module docstring; ~100× quieter, measured).
        # P symmetric ⇒ Wᵀ = BᵀP.
        W = Pk @ B
        return float(
            np.linalg.norm(A.T @ Pk + Pk @ A - W @ Rinv @ W.T + Q, ord="fro") / norm_Q
        )

    rel = _rel_residual(P)
    history = [rel]
    iterations = 0
    while rel >= 1e-8 and iterations < 5:
        K = np.linalg.solve(R, B.T @ P)  # stabilizing gain of the current iterate
        P = _solve_lyapunov(A - B @ K, Q + K.T @ R @ K)
        iterations += 1
        rel = _rel_residual(P)
        history.append(rel)

    if info is not None:
        info["iterations"] = iterations
        info["residual_history"] = list(history)
        info["seed_closed_loop_max_re"] = seed_max_re

    if rel >= 1e-8:
        raise ValueError(
            "Newton–Kleinman polish failed to reach the 1e-8 relative-residual "
            "gate within 5 iterations — residual history (seed first): "
            f"{[f'{r:.3e}' for r in history]}. Quadratic convergence makes this "
            "unreachable for a well-posed (A,B,Q,R); do not loosen the gate — "
            "investigate (see this module's docstring)"
        )

    eigP = np.linalg.eigvalsh(P)
    if np.any(eigP <= 0.0):
        raise ValueError(
            f"P is not positive definite (eigvalsh = {eigP}); (A,B,Q,R) may "
            "not admit a stabilizing solution, or the stable subspace was "
            "mis-selected — see the conditioning caveat in this module's "
            "docstring"
        )

    rel_residual = _rel_residual(P)  # same specified formula, same safe evaluation order
    if rel_residual >= 1e-8:
        raise ValueError(
            f"Riccati residual {rel_residual:.3e} ≥ 1e-8 relative tolerance — "
            "likely a subspace-selection bug, not honest numerical error, on "
            "well-conditioned inputs (see this module's docstring)"
        )

    return P


def solve_dare(A_d: np.ndarray, B_d: np.ndarray, Q: np.ndarray, R: np.ndarray,
               *, info: dict | None = None) -> np.ndarray:
    """Solve the discrete-time algebraic Riccati equation

        P = Q + A_dᵀP A_d − A_dᵀP B_d (R + B_dᵀP B_d)⁻¹ B_dᵀP A_d       (DARE)

    for the stabilizing P ≻ 0 — the MPC terminal cost (mpc.py).

    Method: value iteration, seeded P₀ = Q. Each sweep is one backward-DP
    step of the finite-horizon LQ recursion (iterate k = the k-step
    cost-to-go) — exactly the recursion the MPC equivalence theorem is
    stated in, so the solver and the theorem are the same object. Converges
    geometrically at rate ρ(A_d − B_dK*)² for stabilizable (A_d, B_d) and
    detectable (A_d, Q); measured ~1.2–1.9k sweeps on this repo's plants at
    dt = 5 ms — milliseconds of one-time cost. No analog of the eigenvector
    method's cond(X1) failure mode; Hewer's Newton iteration is the
    quadratic upgrade if construction time ever mattered.

    Stopping: ‖P_{k+1} − P_k‖_F / ‖P_{k+1}‖_F < 1e-12, capped at 20000
    sweeps (bug backstop, not an operating point); measured ~2-3e-10
    per-entry vs the scipy oracle at the gate.

    A_d: (n,n). B_d: (n,m). Q: (n,n) symmetric PSD (PD here ⇒ detectable).
    R: (m,m) symmetric PD. Unit-generic. info: optional dict, filled with
    {"iterations", "final_delta"}. Returns P (n,n), symmetric PD
    (symmetrized per sweep — rounding only). Raises ValueError on
    non-convergence or a non-PD result. Hand-roll line: numpy only;
    ``scipy.linalg.solve_discrete_are`` is the test oracle.
    """
    A_d = np.asarray(A_d, dtype=float)
    B_d = np.asarray(B_d, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    n = A_d.shape[0]
    if A_d.ndim != 2 or A_d.shape != (n, n):
        raise ValueError(f"A_d must be square (n,n), got shape {A_d.shape}")
    if B_d.ndim != 2 or B_d.shape[0] != n:
        raise ValueError(f"B_d must have {n} rows to match A_d, got shape {B_d.shape}")
    if Q.shape != (n, n):
        raise ValueError(f"Q must be ({n},{n}) to match A_d, got shape {Q.shape}")

    P = Q.copy()
    deltas: list[float] = []
    for k in range(1, 20001):
        # One DP backup: the m×m gain solve (R + BᵀPB is PD), then the Riccati map.
        K = np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)  # (m,n)
        P_next = Q + A_d.T @ P @ A_d - (A_d.T @ P @ B_d) @ K
        P_next = 0.5 * (P_next + P_next.T)  # strip rounding asymmetry per sweep
        delta = float(np.linalg.norm(P_next - P, ord="fro")
                      / np.linalg.norm(P_next, ord="fro"))
        deltas.append(delta)
        P = P_next
        if delta < 1e-12:
            break
    else:
        raise ValueError(
            "DARE value iteration did not reach the 1e-12 relative-change "
            f"gate within 20000 sweeps — delta history endpoints: first "
            f"{[f'{d:.3e}' for d in deltas[:3]]}, last "
            f"{[f'{d:.3e}' for d in deltas[-5:]]}. Geometric convergence "
            "makes this unreachable for a stabilizable/detectable "
            "(A_d,B_d,Q,R); do not loosen the gate — investigate (is the "
            "closed-loop spectral radius pathologically close to 1?)"
        )

    if info is not None:
        info["iterations"] = k
        info["final_delta"] = delta

    eigP = np.linalg.eigvalsh(P)
    if np.any(eigP <= 0.0):
        raise ValueError(
            f"DARE P is not positive definite (eigvalsh = {eigP}); "
            "(A_d,B_d,Q,R) may not admit a stabilizing solution"
        )

    return P


def dlqr_gain(A_d: np.ndarray, B_d: np.ndarray, Q: np.ndarray, R: np.ndarray,
              P: np.ndarray) -> np.ndarray:
    """K_d = (R + B_dᵀP B_d)⁻¹ B_dᵀP A_d — the discrete LQR gain from a DARE
    solution P (u = −K_d x); via np.linalg.solve, no explicit inverse.
    Q is accepted for call-signature symmetry with ``solve_dare`` but is
    mathematically unused — P already encodes it. Returns K_d (m,n).
    """
    A_d = np.asarray(A_d, dtype=float)
    B_d = np.asarray(B_d, dtype=float)
    R = np.asarray(R, dtype=float)
    P = np.asarray(P, dtype=float)
    return np.linalg.solve(R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
