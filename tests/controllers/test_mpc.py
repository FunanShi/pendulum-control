"""MPC unit battery: expm/c2d_zoh/solve_dare vs scipy oracles, the exact
unconstrained-MPC ≡ discrete-LQR equivalence, a cvxpy+OSQP formulation oracle,
constraint respect, and the infeasibility fallback. scipy and cvxpy are
test-only oracles — dpend/ never imports cvxpy, and mpc.py uses scipy.sparse
only as an OSQP input-format adapter."""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest
from scipy.linalg import expm as scipy_expm
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from dpend.controllers import base
from dpend.controllers.discretize import c2d_zoh, expm
from dpend.controllers.mpc import MPCController, mpc_factory
from dpend.controllers.riccati import solve_dare
from dpend.model import cart_linearize as cl
from dpend.model import linearize as fl
from dpend.model.cart_params import CartParams
from dpend.model.params import Params, actuation_matrix
from dpend.model.plant import cart_plant, fixed_pivot_plant
from dpend.reference import ReferenceSource
from dpend.registry import CONTROLLERS


def _default_QR(n, m):
    """Factory-default diagonal Q/R, duplicated deliberately: pins the spec's
    numbers as an independent oracle input, not what mpc.py/lqr.py compute."""
    if n == 6:
        Qd = [10.0, 50.0, 50.0, 1.0, 5.0, 5.0]
    elif n == 4:
        Qd = [10.0, 10.0, 1.0, 1.0]
    else:
        raise ValueError(f"no default Q/R fixture for n={n}")
    Rd = [0.1] * m
    return np.diag(Qd), np.diag(Rd)


# --- expm vs scipy.linalg.expm; c2d_zoh vs scipy.signal.cont2discrete ---

@pytest.mark.parametrize("n", range(2, 9))
def test_expm_matches_scipy_oracle_random_matrices(n):
    """Hand-rolled expm matches scipy on seeded random matrices, n=2..8, ‖M‖_1 ≤ ~5.
    Tight rtol: both are machine-accurate here, so looseness would hide a squaring/term bug."""
    rng = np.random.default_rng(2000 + n)
    M_raw = rng.normal(size=(n, n))
    target_norm = float(rng.uniform(0.1, 5.0))
    M = M_raw / np.linalg.norm(M_raw, ord=1) * target_norm

    E = expm(M)
    E_scipy = scipy_expm(M)
    delta = float(np.max(np.abs(E - E_scipy)))
    print(f"\n[expm oracle] n={n}, ||M||_1={target_norm:.3f}: max|expm-scipy| = {delta:.3e}")
    assert np.allclose(E, E_scipy, rtol=1e-12, atol=1e-10)


def test_c2d_zoh_matches_scipy_cont2discrete_both_plants():
    """c2d_zoh matches scipy cont2discrete("zoh") on both plants' (A,B) at ctrl_dt=5 ms.
    An independent routine, so this validates the augmented-matrix wiring, not just expm."""
    A_f, B_f = fl.linearize(Params(), actuation_matrix("acrobot"), fl.UPRIGHT)
    A_c, B_c = cl.linearize(CartParams(), cl.UPRIGHT)
    dt = 5e-3
    for name, (A, B) in {"fixed": (A_f, B_f), "cart": (A_c, B_c)}.items():
        A_d, B_d = c2d_zoh(A, B, dt)
        n = A.shape[0]
        C = np.eye(n)
        D = np.zeros((n, B.shape[1]))
        A_d_ref, B_d_ref, *_ = cont2discrete((A, B, C, D), dt, method="zoh")
        delta_A = float(np.max(np.abs(A_d - A_d_ref)))
        delta_B = float(np.max(np.abs(B_d - B_d_ref)))
        print(f"\n[c2d_zoh oracle, {name}] max|Ad-Ad_ref|={delta_A:.3e}, "
              f"max|Bd-Bd_ref|={delta_B:.3e}")
        assert np.allclose(A_d, A_d_ref, rtol=1e-9, atol=1e-10)
        assert np.allclose(B_d, B_d_ref, rtol=1e-9, atol=1e-10)


# --- solve_dare oracle + exact unconstrained-MPC ≡ discrete-LQR equivalence ---

@pytest.mark.parametrize("plant_name", ["fixed", "cart"])
def test_solve_dare_matches_scipy_oracle(plant_name):
    """Hand-rolled solve_dare matches scipy solve_discrete_are on both plants' ZOH (A_d, B_d), dt=5 ms.
    rtol 1e-9: the stopping criterion leaves ≈ delta·rho²/(1−rho²) ≈ 1e-10 relative error, ~3-6x margin."""
    if plant_name == "fixed":
        A, B = fl.linearize(Params(), actuation_matrix("acrobot"), fl.UPRIGHT)
    else:
        A, B = cl.linearize(CartParams(), cl.UPRIGHT)
    n, m = A.shape[0], B.shape[1]
    Q, R = _default_QR(n, m)
    A_d, B_d = c2d_zoh(A, B, 5e-3)

    info = {}
    P = solve_dare(A_d, B_d, Q, R, info=info)
    P_scipy = solve_discrete_are(A_d, B_d, Q, R)

    delta = float(np.max(np.abs(P - P_scipy)))
    rel = float(np.max(np.abs(P - P_scipy) / np.abs(P_scipy)))
    print(f"\n[dare oracle, {plant_name}] iterations={info['iterations']}, "
          f"max|P - P_scipy| = {delta:.3e} (per-entry rel max = {rel:.3e})")
    assert np.allclose(P, P_scipy, rtol=1e-9, atol=1e-12)

    eigP = np.linalg.eigvalsh(P)
    assert np.all(eigP > 0)  # stabilizing solution is PD (Q PD here)


def test_unconstrained_mpc_first_input_equals_discrete_lqr_exactly():
    """With P_f = the DARE solution (a fixed point of the backward recursion),
    unconstrained MPC's first input equals −K_d·x0 exactly for any N; 20 random cart states.
    atol 1e-5, rtol 0: the equivalence is exact and OSQP's eps 1e-6 sets the only gap.
    reset() each draw so warm-starting cannot couple the solves."""
    plant = cart_plant()
    ctrl = mpc_factory(plant, {"u_max": 1e9, "x_max": None, "N": 40})

    rng = np.random.default_rng(4242)
    max_dev = 0.0
    for _ in range(20):
        x0 = rng.normal(scale=0.01, size=plant.n)
        ctrl.reset(0.0, x0)
        u_mpc = ctrl.update(0.0, x0)
        assert ctrl.status == "optimal"
        u_dlqr = -ctrl.K_d @ x0
        dev = float(np.max(np.abs(u_mpc - u_dlqr)))
        max_dev = max(max_dev, dev)
        np.testing.assert_allclose(u_mpc, u_dlqr, rtol=0.0, atol=1e-5)
    print(f"\n[exact equivalence] max |u_mpc − (−K_d x0)| over 20 random states = {max_dev:.3e}")


def test_discrete_gain_within_ten_percent_of_continuous_gain_on_cart():
    """K_d (discrete) and K_lqr (continuous) agree to <10% per entry on the cart.
    Bound rationale: discretization gap ~ O(dt·|λ_fast|) = 0.005·15.95 ≈ 8%. Cart only —
    the fixed plant's dt·|λ| ≈ 0.23 is outside the small-gap regime (~12.5% measured)."""
    plant = cart_plant()
    ctrl = mpc_factory(plant, {})
    rel = np.abs(ctrl.K_d - ctrl.K_lqr) / np.abs(ctrl.K_lqr)
    max_rel = float(np.max(rel))
    print(f"\n[K_d vs K_lqr, cart] per-entry relative gaps: {np.round(rel, 4)}; "
          f"max = {max_rel:.4f}")
    assert max_rel < 0.10


def test_condensed_hessian_well_conditioned_at_factory_default_horizon():
    """cond(H) at the factory default N=40 stays < 1e4 — guards the conditioning
    cliff: cond(H) grows exponentially with N for this unstable plant (1.6e2 at N=40,
    4.6e4 at N=100, >1e13 by N~300-400), so a casual N bump fails here. Measures the
    controller's own public H — the matrix actually shipped to OSQP."""
    plant = cart_plant()
    ctrl = mpc_factory(plant, {})  # factory defaults: N=40
    assert ctrl.H.shape == (ctrl.N * ctrl.m, ctrl.N * ctrl.m)

    cond_H = float(np.linalg.cond(ctrl.H))
    print(f"\n[conditioning] cond(H) at factory default N={ctrl.N}: {cond_H:.3e}")
    assert cond_H < 1e4  # measured 1.626e2; 1e4 = 61x margin


# --- cvxpy declarative formulation oracle ---

# Oracle objective scale — argmin-invariant (a positive scalar on the whole
# objective moves the optimum's value, never its location). The DARE P_f puts
# cost entries at ~2.6e5, where the oracle's ADMM stalls unconverged at the
# 200k-iteration cap; 1e-3 brings the objective to O(1e2) and it converges
# (~50k iterations, gap ~5e-7). Do not normalize further to O(1): eps_abs then
# dominates termination and OSQP returns garbage (gap 0.199) under an
# "optimal" status. The objective must stay large relative to eps_abs.
_ORACLE_OBJ_SCALE = 1e-3


def _cvxpy_oracle_u_star(A_d, B_d, Q, R, P_f, N, x_tilde0, z_ref, u_max, x_max, n, m):
    """Independent declarative re-derivation of the same finite-horizon QP —
    explicit x_k/u_k variables, dynamics equality constraints, no condensing —
    at eps 1e-9, three orders under the 1e-5 comparison atol. The dropped 1/2
    and _ORACLE_OBJ_SCALE are argmin-invariant, so the QP is identical."""
    U = cp.Variable((N, m))
    X = cp.Variable((N, n))
    cost = 0
    constraints = []
    x_prev = x_tilde0
    for k in range(N):
        constraints.append(X[k] == A_d @ x_prev + B_d @ U[k])
        Qk = P_f if k == N - 1 else Q
        cost += (cp.quad_form(X[k], cp.psd_wrap(_ORACLE_OBJ_SCALE * Qk))
                 + cp.quad_form(U[k], cp.psd_wrap(_ORACLE_OBJ_SCALE * R)))
        constraints += [U[k] >= -u_max, U[k] <= u_max]
        if x_max is not None:
            constraints += [X[k, 0] + z_ref >= -x_max, X[k, 0] + z_ref <= x_max]
        x_prev = X[k]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=200000,
               polishing=True, verbose=False)
    assert prob.status == cp.OPTIMAL, f"cvxpy oracle status: {prob.status}"
    return U.value.reshape(-1)


@pytest.mark.parametrize("case", ["nominal", "near_rail"])
def test_condensed_solve_matches_cvxpy_formulation_oracle(case):
    """Full U* (not just the first input) from the condensed osqp solve matches
    the cvxpy oracle at the factory config (N=40, u_max=150, x_max=rail-0.1) —
    one interior-optimum state, one with the rail constraint genuinely active
    at the optimum (min slack printed as evidence)."""
    plant = cart_plant()
    ctrl = mpc_factory(plant, {})
    N, n, m = ctrl.N, ctrl.n, ctrl.m
    A, B = plant.linearize(plant.upright)
    Q, R = _default_QR(n, m)
    A_d, B_d = c2d_zoh(A, B, ctrl.ctrl_dt)
    # scipy's DARE (not our solve_dare) keeps the oracle independent of
    # dpend's Riccati code; the P_f's agree to ~3e-10 — invisible at this atol.
    P_f = solve_discrete_are(A_d, B_d, Q, R)
    u_max = np.full(m, 150.0)
    x_max = plant.rail - 0.1

    if case == "nominal":
        x0 = np.array([0.1, 0.05, -0.02, 0.0, 0.0, 0.0])
    else:
        # 0.4 m/s into the wall from 0.02 m away: the optimal braking
        # trajectory rides the wall (measured min slack ~ -1e-15), so the rail
        # rows are genuinely active. Harder instances (1.5 m/s from 0.05 m)
        # are solved correctly by the production controller but stall the
        # cvxpy oracle's ADMM — this is the strongest instance the oracle
        # itself can certify at this atol.
        x0 = np.array([x_max - 0.02, 0.02, -0.01, 0.4, 0.0, 0.0])

    ctrl.reset(0.0, x0)
    ctrl.update(0.0, x0)
    assert ctrl.status == "optimal"
    U_ours = ctrl._U_prev

    U_oracle = _cvxpy_oracle_u_star(A_d, B_d, Q, R, P_f, N, x0.copy(), 0.0, u_max, x_max, n, m)

    gap = float(np.max(np.abs(U_ours - U_oracle)))
    if case == "near_rail":
        # activity evidence: roll our U* forward, report the tightest slack
        U_2d = U_ours.reshape(N, m)
        x = x0.copy()
        min_slack = np.inf
        for k in range(N):
            x = A_d @ x + B_d @ U_2d[k]
            min_slack = min(min_slack, x_max - abs(float(x[0])))
        print(f"\n[cvxpy oracle, {case}] max|U_ours - U_cvxpy| = {gap:.3e}, "
              f"min rail slack = {min_slack:.3e} m")
        assert min_slack < 1e-3  # self-certification: active set non-empty here
    else:
        print(f"\n[cvxpy oracle, {case}] max|U_ours - U_cvxpy| = {gap:.3e}")
    assert np.allclose(U_ours, U_oracle, atol=1e-5)


# --- open-loop constraint respect (incl. the tracking-shift/absolute-rail case) ---

def test_predicted_trajectory_respects_rail_bounds_open_loop():
    """From a near-rail drifting x0, the predicted open-loop trajectory under
    U* keeps cart position within ±x_max (+1e-6 solver slack) at all N steps."""
    plant = cart_plant()
    ctrl = mpc_factory(plant, {})
    x_max = plant.rail - 0.1
    x0 = np.array([x_max - 0.05, 0.02, -0.01, 0.3, 0.0, 0.0])

    ctrl.reset(0.0, x0)
    ctrl.update(0.0, x0)
    assert ctrl.status == "optimal"
    U = ctrl._U_prev.reshape(ctrl.N, ctrl.m)

    x = x0.copy()
    max_abs_pos = 0.0
    for k in range(ctrl.N):
        x = ctrl.A_d @ x + ctrl.B_d @ U[k]
        max_abs_pos = max(max_abs_pos, abs(float(x[0])))
    print(f"\n[open-loop constraint respect] x0[0]={x0[0]:.3f}, "
          f"max|predicted cart position| = {max_abs_pos:.6f} (x_max={x_max})")
    assert max_abs_pos <= x_max + 1e-6


def test_tracking_shift_uses_absolute_rail_bounds_not_shifted_coordinates():
    """Rail bounds apply to the absolute (physical) cart position, not the
    tracking-shifted coordinate — the wall does not move with the reference.
    z_ref=0.8, cart moving 1.5 m/s into the wall: dropping the z_ref term
    would overshoot 0.2 m past the physical rail. Self-certification: min
    slack must be < 1e-3 m, else the wall never binds and the test cannot
    distinguish correct bookkeeping from the bug."""
    plant = cart_plant()
    z_ref = 0.8
    ctrl = mpc_factory(plant, {"reference": ReferenceSource(z_ref)})
    x_max = plant.rail - 0.1
    # near the physical wall, moving 1.5 m/s into it (constraint-activating)
    x0 = np.array([x_max - 0.05, 0.02, -0.01, 1.5, 0.0, 0.0])

    ctrl.reset(0.0, x0)
    ctrl.update(0.0, x0)
    assert ctrl.status == "optimal"
    U = ctrl._U_prev.reshape(ctrl.N, ctrl.m)

    z_ref_vec = np.zeros(plant.n)
    z_ref_vec[0] = z_ref
    x_tilde = x0 - z_ref_vec
    max_abs_pos = 0.0
    min_slack = np.inf
    for k in range(ctrl.N):
        x_tilde = ctrl.A_d @ x_tilde + ctrl.B_d @ U[k]
        abs_pos = float(x_tilde[0]) + z_ref  # absolute cart position [m]
        max_abs_pos = max(max_abs_pos, abs(abs_pos))
        min_slack = min(min_slack, x_max - abs(abs_pos))
    print(f"\n[tracking-shift absolute-rail] z_ref={z_ref}, x0[0]={x0[0]:.3f}, "
          f"xdot0={x0[3]:.1f} m/s: max|predicted ABSOLUTE cart position| = "
          f"{max_abs_pos:.6f} (x_max={x_max}), min slack = {min_slack:.3e} m")
    assert max_abs_pos <= x_max + 1e-6
    assert min_slack < 1e-3  # self-certification: the wall actually binds here


# --- infeasibility fallback ---

def test_infeasible_state_triggers_fallback():
    """Cart 1 m past the wall makes the QP unconditionally infeasible (position
    over one 5 ms tick is u-independent): status "fallback", u finite and within
    the box bound; reset() then clears the sticky status back to "optimal"."""
    plant = cart_plant()
    ctrl = mpc_factory(plant, {})
    x_max = plant.rail - 0.1
    x0 = np.zeros(plant.n)
    x0[0] = x_max + 1.0

    ctrl.reset(0.0, x0)
    u = ctrl.update(0.0, x0)
    print(f"\n[infeasibility fallback] status={ctrl.status}, u={u}")
    assert ctrl.status == "fallback"
    assert np.all(np.isfinite(u))
    assert np.all(np.abs(u) <= 150.0 + 1e-9)

    ctrl.reset(0.0, np.zeros(plant.n))
    assert ctrl.status == "optimal"  # reset clears last episode's outcome


def test_solver_exception_maps_to_fallback_not_crash(monkeypatch):
    """A raising solver must never crash a control tick: any exception routes
    to the fallback branch — clipped continuous-LQR, finite, |u| ≤ u_max,
    status "fallback"."""
    plant = cart_plant()
    ctrl = mpc_factory(plant, {})

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated osqp API drift / internal solver error")

    monkeypatch.setattr(ctrl._osqp, "solve", _boom)

    x0 = np.zeros(plant.n)
    x0[1] = 0.1  # ordinary near-upright state; nothing special about it
    ctrl.reset(0.0, x0)
    u = ctrl.update(0.0, x0)
    print(f"\n[solver-exception fallback] status={ctrl.status}, u={u}")
    assert ctrl.status == "fallback"
    assert np.all(np.isfinite(u))
    assert np.all(np.abs(u) <= 150.0 + 1e-9)


# --- protocol conformance + exposed gains + factory on fixed plant (no rail) ---

def test_mpc_conforms_to_protocol_and_exposes_gains_and_p():
    assert "mpc" in CONTROLLERS
    plant = cart_plant()
    ctrl = CONTROLLERS["mpc"](plant, {})
    assert isinstance(ctrl, base.Controller)
    assert isinstance(ctrl, MPCController)
    assert ctrl.K_lqr.shape == (plant.m, plant.n)
    assert ctrl.K_d.shape == (plant.m, plant.n)
    assert ctrl.P.shape == (plant.n, plant.n)
    assert isinstance(ctrl.status, str)


def test_mpc_factory_on_fixed_plant_no_rail_still_solves():
    """No rail → no state rows in the constraint matrix (x_max and _Su_pos
    stay None) and the controller still solves normally."""
    plant = fixed_pivot_plant(actuation="acrobot")
    ctrl = CONTROLLERS["mpc"](plant, {})
    assert plant.rail is None
    assert ctrl._x_max is None
    assert ctrl._Su_pos is None

    x0 = np.array([0.02, -0.01, 0.0, 0.0])
    ctrl.reset(0.0, x0)
    u = ctrl.update(0.0, x0)
    print(f"\n[fixed plant, no rail] status={ctrl.status}, u={u}")
    assert ctrl.status == "optimal"
    assert u.shape == (plant.m,)
    assert np.all(np.isfinite(u))
