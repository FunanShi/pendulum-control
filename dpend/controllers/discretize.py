"""Hand-rolled matrix exponential + exact zero-order-hold discretization —
the numerical foundation the condensed MPC (``mpc.py``) predicts over.

Hand-roll line: numpy only. ``scipy.linalg.expm`` and
``scipy.signal.cont2discrete`` are test oracles (tests/test_mpc.py), never
production dependencies.

``expm`` — scaling and squaring: a direct Taylor series needs many terms
and cancels badly when ‖M‖ is large, so shrink the argument first via
e^M = (e^{M/2^s})^{2^s} with

    s = max(0, ⌈log2(‖M‖₁)⌉ + 1)

which lands ‖M/2^s‖₁ ≤ 0.5. The series on the scaled argument
(term_k = term_{k-1}·A/k — one matmul per term) is truncated on measured
convergence, ‖term_k‖₁ < eps·‖result‖₁, not at a fixed count; ~10-15 terms
in practice, capped at 30 as a bug backstop. Then square s times.

``c2d_zoh`` — for ẋ = Ax + Bu with u held constant over one tick of length
dt (what a digital controller does), the exact discrete update has closed
forms

    A_d = e^{A·dt},    B_d = (∫₀^dt e^{Aτ} dτ) · B

computed in one shot via Van Loan's augmented matrix (IEEE T-AC 1978):

    M_aug = [[A, B],   (n+m, n+m),   e^{M_aug · dt} = [[A_d, B_d],
             [0, 0]]                                   [0,   I_m ]]

(the zero bottom rows integrate to exactly I_m — the trick is exact, no
approximation beyond ``expm`` itself). Both functions are cross-checked
against the scipy oracles at tight tolerances in tests/test_mpc.py;
measured deltas are ~1e-16.
"""
from __future__ import annotations

import numpy as np

_EPS = np.finfo(float).eps
_MAX_TERMS = 30


def expm(M: np.ndarray) -> np.ndarray:
    """Matrix exponential e^M via scaling-and-squaring (module docstring).
    M: (n,n), any real square matrix — a generic numerical primitive.
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    if M.ndim != 2 or M.shape != (n, n):
        raise ValueError(f"expm: M must be square (n,n), got shape {M.shape}")

    norm1 = float(np.linalg.norm(M, ord=1))
    if norm1 == 0.0:
        return np.eye(n)  # e^0 = I; log2(0) is undefined, so special-case first

    s = max(0, int(np.ceil(np.log2(norm1))) + 1)
    A = M / (2.0 ** s)

    result = np.eye(n)
    term = np.eye(n)
    for k in range(1, _MAX_TERMS + 1):
        term = term @ A / k
        result = result + term
        if np.linalg.norm(term, ord=1) < _EPS * np.linalg.norm(result, ord=1):
            break

    for _ in range(s):
        result = result @ result

    return result


def c2d_zoh(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact ZOH discretization (A,B,dt) -> (A_d, B_d) via the augmented-matrix
    trick (see module docstring). A: (n,n). B: (n,m). dt: control tick [s].
    Returns A_d: (n,n), B_d: (n,m) — the SAME shapes as the continuous pair,
    so callers don't need to special-case dimensions.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    m = B.shape[1]

    M_aug = np.zeros((n + m, n + m))
    M_aug[:n, :n] = A
    M_aug[:n, n:] = B
    # bottom (m,n+m) block stays zero: ẏ = 0 ⇒ bottom-right integrates to I_m.

    E = expm(M_aug * dt)
    A_d = E[:n, :n]
    B_d = E[:n, n:]
    return A_d, B_d
