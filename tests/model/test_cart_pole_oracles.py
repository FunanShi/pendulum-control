"""Independent oracles for the single-pole cart-pole: scipy reference-
integrator energy check, sympy 2-DOF Lagrangian re-derivation, end-stop
dissipation. No code shared with dpend.model.cart_pole_dynamics beyond parameters."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dpend.model.cart_pole_params import CartPoleParams


def test_cart_pole_energy_conserved_under_reference_integrator():
    """u=0, friction=0 => E constant. DOP853 at rtol=1e-11 makes integrator
    error negligible; tolerance 1e-8 relative is ~1000x that error but far
    below any sign/term error in M, C, or g."""
    from dpend.model.cart_pole_dynamics import energy, f

    cp = CartPoleParams()
    u = np.array([0.0])  # N -- no cart force
    z0 = np.array([0.0, 0.4, 0.0, 0.0])  # [x, theta, xdot, thetadot] -- well off equilibrium, swings hard

    sol = solve_ivp(
        lambda t, z: f(z, u, cp),
        (0.0, 10.0),  # s
        z0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-12,
        dense_output=False,
        t_eval=np.linspace(0.0, 10.0, 501),
    )
    assert sol.success
    # In-rail sanity: with p_x conserved (=0, rest IC) x(t) is bounded by
    # 2*b/mt =~ 0.33 m no matter how hard the pole swings — comfortably inside
    # L_rail=1.5, so end-stop damping (which breaks conservation) never engages.
    x_max = float(np.max(np.abs(sol.y[0, :])))
    print(f"\nmax|x| during free swing: {x_max:.4f} m (L_rail={cp.L_rail})")
    assert x_max < cp.L_rail

    E = np.array([energy(cp, z) for z in sol.y.T])
    drift = np.max(np.abs(E - E[0])) / max(abs(E[0]), 1.0)
    print(f"reference-integrator relative energy drift over 10 s: {drift:.3e}")
    assert drift < 1e-8


@pytest.fixture(scope="module")
def sympy_mcg_cart_pole():
    """Independently derive M, C*qdot, g for the 2-DOF cart-pole via sympy
    Euler-Lagrange; returns lambdified callables (x, theta, xdot, thetadot) -> arrays.
    Frame: cart at (x, 0); pole COM at (x - l*sin(theta), l*cos(theta));
    q = [x, theta]. I is the pole's own COM inertia (parallel-axis
    decomposition: T_pole = 1/2 mp v_com^2 + 1/2 I thetadot^2)."""
    import sympy as sp

    t = sp.symbols("t")
    mc, mp, l, I, g0 = sp.symbols("mc mp l I g0", positive=True)

    x_cart = sp.Function("x_cart")(t)
    th = sp.Function("th")(t)

    # World frame: x right, y up; theta from +y (upward vertical), CCW+.
    # Cart at (x, 0). Pole COM: (x - l*sin(theta), l*cos(theta)).
    x1 = x_cart - l * sp.sin(th)
    y1 = l * sp.cos(th)

    xdot_cart, thdot = sp.diff(x_cart, t), sp.diff(th, t)

    T_cart = sp.Rational(1, 2) * mc * xdot_cart**2
    v1sq = sp.diff(x1, t) ** 2 + sp.diff(y1, t) ** 2
    T_pole = sp.Rational(1, 2) * mp * v1sq + sp.Rational(1, 2) * I * thdot**2
    T = T_cart + T_pole

    # Potential energy (pole only; cart has no PE): V = g0*mp*y1 = mp*g0*l*cos(theta)
    V = g0 * mp * y1

    L = T - V

    q = [x_cart, th]
    qd = [xdot_cart, thdot]
    qdd = [sp.diff(v, t) for v in qd]

    # Euler-Lagrange: EL = M q_ddot + C*qdot + g (tau with u=0)
    EL = sp.Matrix(
        [sp.simplify(sp.diff(sp.diff(L, qd_i), t) - sp.diff(L, q_i)) for q_i, qd_i in zip(q, qd)]
    )

    Msym = EL.jacobian(sp.Matrix(qdd))
    g_sym = EL.subs([(a, 0) for a in qdd + qd])              # q_ddot=qdot=0 => gravity only
    Cqd_sym = sp.simplify(EL - Msym * sp.Matrix(qdd) - g_sym)  # velocity terms

    # Substitute numerical values (CartPoleParams defaults)
    subs = {
        mc: 1.0, mp: 0.5, l: 0.5, I: 0.5 * 0.5**2 / 12, g0: 9.81,
    }

    x_cart_, th_, xdot_cart_, w_ = sp.symbols("x_cart_ th_ xdot_cart_ w_")
    repl = [
        (sp.diff(x_cart, t), xdot_cart_),
        (sp.diff(th, t), w_),
        (x_cart, x_cart_),
        (th, th_),
    ]

    def lam(expr):
        return sp.lambdify(
            (x_cart_, th_, xdot_cart_, w_),
            expr.subs(subs).subs(repl),
            "numpy",
        )

    return lam(Msym), lam(Cqd_sym), lam(g_sym)


def test_cart_pole_hand_coded_mcg_match_sympy(sympy_mcg_cart_pole):
    """Hand-coded M, C*qdot, g match the sympy fixture at 10 random configs.
    atol=1e-9: C*qdot scale is a few N, so this catches any wrong factor while
    sitting well above roundoff."""
    from dpend.model.cart_pole_dynamics import coriolis_matrix, gravity, mass_matrix

    M_s, Cqd_s, g_s = sympy_mcg_cart_pole
    cp = CartPoleParams()
    rng = np.random.default_rng(42)

    max_M_residual = 0.0
    max_Cqd_residual = 0.0
    max_g_residual = 0.0

    for _ in range(10):
        x_cart = rng.uniform(-1.0, 1.0)  # m
        theta = rng.uniform(-np.pi, np.pi)
        xdot, w = rng.uniform(-4.0, 4.0, size=2)

        M_hand = mass_matrix(cp, theta)
        M_sympy = np.array(M_s(x_cart, theta, xdot, w), dtype=float)
        M_diff = np.linalg.norm(M_hand - M_sympy)
        max_M_residual = max(max_M_residual, M_diff)
        np.testing.assert_allclose(M_hand, M_sympy, atol=1e-9, err_msg="M mismatch vs sympy")

        Cqd_hand = coriolis_matrix(cp, theta, w) @ np.array([xdot, w])
        Cqd_sympy = np.array(Cqd_s(x_cart, theta, xdot, w), dtype=float).ravel()
        Cqd_diff = np.linalg.norm(Cqd_hand - Cqd_sympy)
        max_Cqd_residual = max(max_Cqd_residual, Cqd_diff)
        np.testing.assert_allclose(Cqd_hand, Cqd_sympy, atol=1e-9, err_msg="C*qdot mismatch vs sympy")

        g_hand = gravity(cp, theta)
        g_sympy = np.array(g_s(x_cart, theta, xdot, w), dtype=float).ravel()
        g_diff = np.linalg.norm(g_hand - g_sympy)
        max_g_residual = max(max_g_residual, g_diff)
        np.testing.assert_allclose(g_hand, g_sympy, atol=1e-9, err_msg="g mismatch vs sympy")

    print("\nmax sympy residuals (10 random configs):")
    print(f"  M:     {max_M_residual:.3e}")
    print(f"  C*qd:  {max_Cqd_residual:.3e}")
    print(f"  g:     {max_g_residual:.3e}")


def test_cart_pole_end_stop_dissipation():
    """End-stop dynamics from z0 = [1.2, 0, 2.0, 0] (coasting at +stop, pole
    upright). At theta=0 gravity and Coriolis vanish, so the pole holds until
    first contact; three stop impacts then destabilize it via mass-matrix
    coupling, measured max|x| = 1.552862 m. The L_rail+0.1 bound is empirical
    for this z0 (stable to 6 figures under IC/tolerance perturbation) —
    changes to z0/k_stop/masses must re-measure it. Also: energy monotone
    non-increasing within each contact episode (c_stop dissipation), then
    flat (<1e-8 rel) on the final in-rail second (no damping in-rail)."""
    from dpend.model.cart_pole_dynamics import energy, f

    cp = CartPoleParams()
    u = np.array([0.0])
    z0 = np.array([1.2, 0.0, 2.0, 0.0])  # [x, theta, xdot, thetadot]

    sol = solve_ivp(
        lambda t, z: f(z, u, cp),
        (0.0, 5.0),  # s
        z0,
        method="DOP853",
        rtol=1e-9,
        atol=1e-10,  # stiff-ish contact; tight tolerance needed
        dense_output=True,
        t_eval=np.linspace(0.0, 5.0, 1001),
    )
    assert sol.success, "end-stop integration failed"

    x_traj = sol.y[0, :]
    E_traj = np.array([energy(cp, sol.y[:, i]) for i in range(sol.y.shape[1])])

    # Assertion 1: max|x| < L_rail + 0.1 (empirical bound for this z0 — see
    # docstring; measured 1.552862 m, so ~1.9x the actual overshoot)
    x_max = np.max(np.abs(x_traj))
    print(f"\nend-stop test:")
    print(f"  max(|x|): {x_max:.6f} m (limit: {cp.L_rail + 0.1:.4f} m)")
    assert x_max < cp.L_rail + 0.1, f"overshoot exceeded: {x_max} > {cp.L_rail + 0.1}"

    # Assertion 2: energy monotone non-increasing within each contact episode.
    # Split into contiguous runs — diffing across in-rail gaps would pass only
    # by sign luck.
    idx = np.where(np.abs(x_traj) > cp.L_rail)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    worst_intra_dE = -np.inf
    for run in runs:
        if len(run) > 1:
            dE = np.diff(E_traj[run])
            worst_intra_dE = max(worst_intra_dE, float(np.max(dE)))
            assert np.all(dE < 1e-10), \
                "energy not monotone non-increasing within a contact episode"
    print(f"  contact episodes: {len(runs)}; worst intra-episode energy diff: "
          f"{worst_intra_dE:.3e} J (non-increase check)")

    # Assertion 3: flat energy on final in-rail second (t in [4, 5])
    final_indices = np.where(sol.t >= 4.0)[0]
    if len(final_indices) > 0:
        E_final = E_traj[final_indices]
        drift_final = np.max(np.abs(E_final - E_final[0])) / max(abs(E_final[0]), 1.0)
        print(f"  final-second energy drift (t in [4, 5]): {drift_final:.3e} (limit: 1e-8)")
        assert drift_final < 1e-8, f"energy not conserved in-rail: drift = {drift_final}"
