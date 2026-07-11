"""Hand-rolled Riccati (CARE) solver vs the scipy oracle, closed-loop Hurwitz
checks on both plants, the stabilizability failure path, and LQR Q/R-override,
saturation, and tracking algebra. scipy is a test-only oracle — the production
Riccati solve is numpy-only."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_continuous_are

from dpend.controllers import base
from dpend.controllers.lqr import LQRController, lqr_factory
from dpend.controllers.riccati import solve_care
from dpend.model import cart_linearize as cl
from dpend.model import linearize as fl
from dpend.model.cart_params import CartParams
from dpend.model.linearize import ctrb, rank_and_cond
from dpend.model.params import Params, actuation_matrix
from dpend.model.plant import cart_plant
from dpend.reference import ReferenceSource
from dpend.registry import CONTROLLERS


def _fixed_AB():
    """Fixed-pivot (acrobot) A,B at upright — the n=4 plant."""
    p = Params()
    B = actuation_matrix("acrobot")
    return fl.linearize(p, B, fl.UPRIGHT)


def _cart_AB():
    """Cart-mounted A,B at upright — the n=6 plant."""
    return cl.linearize(CartParams(), cl.UPRIGHT)


def _default_QR(n, m):
    """Factory-default diagonal Q/R, duplicated deliberately: pins the spec's
    numbers, not what lqr.py computes internally."""
    if n == 6:
        Qd = [10.0, 50.0, 50.0, 1.0, 5.0, 5.0]
    elif n == 4:
        Qd = [10.0, 10.0, 1.0, 1.0]
    else:
        raise ValueError(f"no default Q/R fixture for n={n}")
    Rd = [0.1] * m
    return np.diag(Qd), np.diag(Rd)


_PLANTS_AB = {"fixed": _fixed_AB, "cart": _cart_AB}


# --- solve_care vs scipy oracle, both plants, factory-default Q/R ---

@pytest.mark.parametrize("plant_name", ["fixed", "cart"])
def test_solve_care_matches_scipy_oracle(plant_name):
    """solve_care matches scipy solve_continuous_are on both plants, factory Q/R.
    Tight rtol: both are ~machine-accurate on these small, well-conditioned systems."""
    A, B = _PLANTS_AB[plant_name]()
    n, m = A.shape[0], B.shape[1]
    Q, R = _default_QR(n, m)

    P = solve_care(A, B, Q, R)
    P_scipy = solve_continuous_are(A, B, Q, R)

    delta = float(np.max(np.abs(P - P_scipy)))
    print(f"\n[{plant_name}] n={n} m={m}: max|P - P_scipy| = {delta:.3e}")
    assert np.allclose(P, P_scipy, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("plant_name", ["fixed", "cart"])
def test_solve_care_returns_symmetric_pd_with_small_residual(plant_name):
    """P symmetric, positive definite, CARE residual small — recomputed here
    from the raw definition, not by trusting solve_care's internal checks;
    residual tolerance matches the solver's own validation gate."""
    A, B = _PLANTS_AB[plant_name]()
    n = A.shape[0]
    Q, R = _default_QR(n, B.shape[1])

    P = solve_care(A, B, Q, R)
    assert np.array_equal(P, P.T)  # exact: solver symmetrizes explicitly

    eigP = np.linalg.eigvalsh(P)
    print(f"\n[{plant_name}] eigvalsh(P) min={eigP.min():.3e} max={eigP.max():.3e}")
    assert np.all(eigP > 0)

    Rinv = np.linalg.inv(R)
    # Evaluation order is load-bearing: left-assoc P @ B is the cancellation-
    # safe W=PB form (~1e-10 here); grouping as P @ (B R⁻¹ Bᵀ) @ P has a ~1e-8
    # noise floor and would trip the bound on correct code.
    W = P @ B  # (n, m)
    residual = A.T @ P + P @ A - W @ Rinv @ W.T + Q
    rel = float(np.linalg.norm(residual, "fro") / np.linalg.norm(Q, "fro"))
    print(f"[{plant_name}] independently-recomputed relative residual = {rel:.3e}")
    assert rel < 1e-8


# --- generic solver: 5 seeded random stabilizable systems ---

def test_solve_care_matches_scipy_on_random_stabilizable_systems():
    """5 systems, n=2..6, m=1..2 — solve_care is generic, not plant-tuned.
    Construct stabilizable pairs (A,B random, retry the seed if ctrb-
    deficient — a measure-zero event for Gaussian entries, but guarded)."""
    n_m_pairs = [(2, 1), (3, 2), (4, 1), (5, 2), (6, 1)]
    for i, (n, m) in enumerate(n_m_pairs):
        seed = 1000 + i
        for attempt in range(50):
            rng = np.random.default_rng(seed)
            A = rng.normal(size=(n, n))
            B = rng.normal(size=(n, m))
            rank, _ = rank_and_cond(ctrb(A, B))
            if rank == n:
                break
            seed += 1  # retry: this draw was ctrb-deficient
        else:
            raise AssertionError(f"could not draw a controllable (A,B) for n={n},m={m}")

        Q = np.eye(n)
        R = np.eye(m)
        P = solve_care(A, B, Q, R)
        P_scipy = solve_continuous_are(A, B, Q, R)
        delta = float(np.max(np.abs(P - P_scipy)))
        print(f"\nrandom system {i} (n={n}, m={m}, seed={seed}, attempt={attempt}): "
              f"max|P-P_scipy| = {delta:.3e}")
        assert np.allclose(P, P_scipy, rtol=1e-9, atol=1e-12)


# --- closed-loop Hurwitz, both plants ---

@pytest.mark.parametrize("plant_name", ["fixed", "cart"])
def test_closed_loop_hurwitz_both_plants(plant_name):
    """(A - B K) is Hurwitz with K = R^-1 B^T P from the factory-default Q/R;
    prints the stability margin (-max Re eig)."""
    A, B = _PLANTS_AB[plant_name]()
    n = A.shape[0]
    Q, R = _default_QR(n, B.shape[1])

    P = solve_care(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    closed_loop_eigs = np.linalg.eigvals(A - B @ K)
    max_re = float(np.max(closed_loop_eigs.real))
    print(f"\n[{plant_name}] closed-loop eig(A-BK) real parts: "
          f"{np.sort(closed_loop_eigs.real)}")
    print(f"[{plant_name}] Hurwitz margin (-max Re eig) = {-max_re:.6f}  "
          f"(max Re eig = {max_re:.6e})")
    assert max_re < 0.0


# --- failure path: uncontrollable pair raises ValueError ---

def test_solve_care_raises_on_uncontrollable_pair():
    """Cart A with B zeroed: the upright instability is unactuated, so (A,B)
    is not stabilizable and solve_care must raise (the Hamiltonian's
    stable-eigenvalue count is 7, not the required n=6)."""
    A, _ = _cart_AB()
    B_zero = np.zeros((A.shape[0], 1))
    Q, R = _default_QR(A.shape[0], 1)
    with pytest.raises(ValueError):
        solve_care(A, B_zero, Q, R)


# --- Newton–Kleinman polish + Lyapunov helper ---

def test_solve_lyapunov_2x2_hand_check():
    """Pin _solve_lyapunov's conventions with a hand-derived 2×2 case.
    A_cl = [[-1, 1], [0, -2]] (non-symmetric on purpose), S = [[2, 1], [1, 2]]:
    A_clᵀP + PA_cl = -S gives P = [[1, 2/3], [2/3, 5/6]]; the wrong transpose
    convention gives [[3/2, 1/2], [1/2, 1/2]]. A non-symmetric A_cl separates
    the two, catching kron/vec-order/transpose bugs."""
    from dpend.controllers.riccati import _solve_lyapunov

    A_cl = np.array([[-1.0, 1.0], [0.0, -2.0]])
    S = np.array([[2.0, 1.0], [1.0, 2.0]])
    P = _solve_lyapunov(A_cl, S)

    P_hand = np.array([[1.0, 2.0 / 3.0], [2.0 / 3.0, 5.0 / 6.0]])
    P_wrong_convention = np.array([[1.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(P, P_hand, atol=1e-13)
    assert not np.allclose(P, P_wrong_convention, atol=1e-3)
    # and verify by direct substitution, independent of the hand algebra:
    np.testing.assert_allclose(A_cl.T @ P + P @ A_cl, -S, atol=1e-13)


def test_newton_kleinman_polish_converges_fast_on_fixed_plant():
    """The case that motivated the polish: the fixed plant's default Q/R leave
    the eigenvector seed's residual at ~2.8e-8 (a near-degenerate stable pair
    inflates cond(X1) to ~1e4). Newton–Kleinman is quadratically convergent
    from a stabilizing seed [Kleinman 1968], so it must reach the 1e-8 gate in
    ≤3 iterations (measured: 1)."""
    A, B = _fixed_AB()
    Q, R = _default_QR(4, 1)

    diag = {}
    P = solve_care(A, B, Q, R, info=diag)

    print(f"\n[fixed] NK residual history (seed first): "
          f"{[f'{r:.3e}' for r in diag['residual_history']]}")
    print(f"[fixed] NK iterations = {diag['iterations']}, "
          f"seed closed-loop max Re = {diag['seed_closed_loop_max_re']:.6f}")
    assert diag["iterations"] <= 3
    assert diag["residual_history"][-1] < 1e-8
    if diag["iterations"]:  # whenever the polish ran, it must have improved
        assert diag["residual_history"][-1] < diag["residual_history"][0]
    assert diag["seed_closed_loop_max_re"] < 0.0  # Kleinman precondition held

    # the polished P is still the right object (oracle re-check):
    P_scipy = solve_continuous_are(A, B, Q, R)
    assert np.allclose(P, P_scipy, rtol=1e-9, atol=1e-12)


# --- Q/R override changes K; u_max clips ---

def test_qr_override_changes_gain():
    """Distinct Q/R diagonals produce a different K than the defaults.
    Cart plant (the property is plant-agnostic)."""
    plant = cart_plant()
    ctrl_default = lqr_factory(plant, {})
    ctrl_override = lqr_factory(
        plant, {"Q": [100.0, 100.0, 100.0, 10.0, 10.0, 10.0], "R": [1.0]}
    )

    x_err = np.array([0.2, 0.1, -0.05, 0.0, 0.0, 0.0])
    u_default = ctrl_default.update(0.0, x_err)
    u_override = ctrl_override.update(0.0, x_err)
    print(f"\nu_default={u_default}, u_override={u_override}")
    assert not np.allclose(u_default, u_override)


def test_u_max_clips_saturation():
    """A huge state error must not produce an unbounded force when u_max is set."""
    plant = cart_plant()
    u_max = 0.5
    ctrl = lqr_factory(plant, {"u_max": u_max})

    huge_error = np.array([1e6, 1e6, -1e6, 1e6, -1e6, 1e6])
    u = ctrl.update(0.0, huge_error)
    print(f"\nu (huge error, u_max={u_max}) = {u}")
    assert np.all(np.abs(u) <= u_max + 1e-12)

    ctrl_unbounded = lqr_factory(plant, {})
    u_unbounded = ctrl_unbounded.update(0.0, huge_error)
    print(f"u (huge error, no u_max) = {u_unbounded}")
    assert np.any(np.abs(u_unbounded) > u_max)  # confirms the clip actually did something


# --- tracking algebra ---

def test_tracking_zero_error_at_shifted_equilibrium():
    """z_ref_fn = e1*0.8: at x_hat = e1*0.8 the error is exactly zero, so u
    must be exactly zeros(m) (IEEE754: -K @ 0 is exact, no tolerance needed)."""
    n, m = 6, 1
    K = np.arange(1, n * m + 1, dtype=float).reshape(m, n)  # arbitrary nonzero gain
    e1 = np.zeros(n); e1[0] = 1.0
    ctrl = LQRController(K, z_ref_fn=lambda t: e1 * 0.8)

    u = ctrl.update(0.0, e1 * 0.8)
    np.testing.assert_array_equal(u, np.zeros(m))


def test_tracking_pushes_toward_positive_target():
    """At x_hat=0 with target +0.8 m on the rail coordinate, u == +K[:,0]*0.8
    exactly, and the force is positive (stabilizing K[0,0] > 0): it pushes the
    cart toward +x."""
    plant = cart_plant()
    ctrl = lqr_factory(plant, {"reference": ReferenceSource(0.8)})
    # recompute the same K the factory used, independently:
    A, B = cl.linearize(CartParams(), cl.UPRIGHT)
    Q, R = _default_QR(6, 1)
    P = solve_care(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    u = ctrl.update(0.0, np.zeros(6))
    expected = K[:, 0] * 0.8
    np.testing.assert_allclose(u, expected, rtol=1e-12)
    print(f"\nK[:,0]={K[:, 0]}, u(track +0.8 from origin)={u}")
    assert u[0] > 0.0  # positive force: pushes the cart toward +x


# --- protocol conformance (through the registry) ---

def test_lqr_controller_conforms_to_protocol():
    assert "lqr" in CONTROLLERS
    plant = cart_plant()
    ctrl = CONTROLLERS["lqr"](plant, {})
    assert isinstance(ctrl, base.Controller)
    assert isinstance(ctrl, LQRController)
