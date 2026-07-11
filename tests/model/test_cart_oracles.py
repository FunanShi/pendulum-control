"""Independent oracles for the cart-mounted double pendulum: scipy reference-
integrator energy check, sympy 3-DOF Lagrangian re-derivation, end-stop
dissipation. No code shared with dpend.model.cart_dynamics beyond parameters."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dpend.model.cart_params import CartParams


def test_cart_energy_conserved_under_reference_integrator():
    """u=0, friction=0 ⇒ E constant. DOP853 at rtol=1e-11 makes integrator
    error negligible; tolerance 1e-8 relative is ~1000× that error but ~10⁶×
    below any sign/term error in M, C, or g."""
    from dpend.model.cart_dynamics import energy, f

    cp = CartParams()
    u = np.array([0.0])  # N — no cart force
    z0 = np.array([0.0, 0.4, -0.3, 0.0, 0.0, 0.0])  # [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂] — well off equilibrium, swings hard

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
    E = np.array([energy(cp, z) for z in sol.y.T])
    drift = np.max(np.abs(E - E[0])) / max(abs(E[0]), 1.0)
    print(f"\nreference-integrator relative energy drift over 10 s: {drift:.3e}")
    assert drift < 1e-8


@pytest.fixture(scope="module")
def sympy_mcg_cart():
    """Independently derive M, C·q̇, g for 3-DOF cart system via sympy Euler–Lagrange
    and return lambdified callables (x, theta1, theta2, xdot, w1, w2) -> arrays.

    Frame convention: cart at (x, 0) in world frame. COM positions per standard:
    a point at distance r along a link at absolute angle θ from +y is at
    (x_base − r·sinθ, r·cosθ). Generalized coords q = [x, θ₁, θ₂].
    """
    import sympy as sp

    t = sp.symbols("t")
    # Pendulum parameters (from pend defaults)
    m1, m2, lc1, lc2, l1, I1, I2, g0 = sp.symbols(
        "m1 m2 lc1 lc2 l1 I1 I2 g0", positive=True
    )
    # Cart parameter
    mc = sp.symbols("mc", positive=True)

    # Generalized coordinates
    x_cart = sp.Function("x_cart")(t)
    th1 = sp.Function("th1")(t)
    th2 = sp.Function("th2")(t)

    # World frame: x right, y up; angles from +y (upward vertical), CCW+.
    # Cart at (x, 0).
    # Link 1 COM: (x − lc1·sinθ₁, lc1·cosθ₁)
    x1 = x_cart - lc1 * sp.sin(th1)
    y1 = lc1 * sp.cos(th1)
    # Link 2 COM: (x − l1·sinθ₁ − lc2·sin(θ₁+θ₂), l1·cosθ₁ + lc2·cos(θ₁+θ₂))
    x2 = x_cart - l1 * sp.sin(th1) - lc2 * sp.sin(th1 + th2)
    y2 = l1 * sp.cos(th1) + lc2 * sp.cos(th1 + th2)

    # Velocities
    xdot_cart, w1s, w2s = sp.diff(x_cart, t), sp.diff(th1, t), sp.diff(th2, t)

    # Kinetic energy: cart + link 1 + link 2
    # Cart: ½·mc·ẋ²
    T_cart = sp.Rational(1, 2) * mc * xdot_cart**2

    # Link 1: ½·m1·v1² + ½·I1·θ̇₁²
    v1sq = sp.diff(x1, t)**2 + sp.diff(y1, t)**2
    T1 = sp.Rational(1, 2) * m1 * v1sq + sp.Rational(1, 2) * I1 * w1s**2

    # Link 2: ½·m2·v2² + ½·I2·(θ̇₁+θ̇₂)²
    v2sq = sp.diff(x2, t)**2 + sp.diff(y2, t)**2
    T2 = sp.Rational(1, 2) * m2 * v2sq + sp.Rational(1, 2) * I2 * (w1s + w2s)**2

    T = T_cart + T1 + T2

    # Potential energy (pendulum only; cart has no PE)
    # V = g₀(m₁y₁ + m₂y₂)
    V = g0 * (m1 * y1 + m2 * y2)

    L = T - V

    q = [x_cart, th1, th2]
    qd = [xdot_cart, w1s, w2s]
    qdd = [sp.diff(v, t) for v in qd]

    # Euler–Lagrange: EL = M q̈ + C·q̇ + g  (τ with u=0)
    EL = sp.Matrix(
        [sp.simplify(sp.diff(sp.diff(L, qd_i), t) - sp.diff(L, q_i)) for q_i, qd_i in zip(q, qd)]
    )

    # Extract M, C, g
    Msym = EL.jacobian(sp.Matrix(qdd))
    g_sym = EL.subs([(a, 0) for a in qdd + qd])          # q̈=q̇=0 ⇒ gravity only
    Cqd_sym = sp.simplify(EL - Msym * sp.Matrix(qdd) - g_sym)  # velocity terms

    # Substitute numerical values (pend defaults + mc=1.0)
    subs = {
        m1: 1.0, m2: 1.0, l1: 1.0, lc1: 0.5, lc2: 0.5,
        I1: 1.0 / 12.0, I2: 1.0 / 12.0, g0: 9.81,
        mc: 1.0
    }

    # Create substitution rules for symbolic functions
    x_cart_, th1_, th2_, xdot_cart_, w1_, w2_ = sp.symbols(
        "x_cart_ th1_ th2_ xdot_cart_ w1_ w2_"
    )
    repl = [
        (sp.diff(x_cart, t), xdot_cart_),
        (sp.diff(th1, t), w1_),
        (sp.diff(th2, t), w2_),
        (x_cart, x_cart_),
        (th1, th1_),
        (th2, th2_),
    ]

    def lam(expr):
        return sp.lambdify(
            (x_cart_, th1_, th2_, xdot_cart_, w1_, w2_),
            expr.subs(subs).subs(repl),
            "numpy"
        )

    return lam(Msym), lam(Cqd_sym), lam(g_sym)


def test_cart_hand_coded_mcg_match_sympy(sympy_mcg_cart):
    """Hand-coded M, C·q̇, g match the sympy fixture at 10 random configs.
    atol=1e-9: C·q̇ scale is tens of N, so this is ~1e-10 relative — catches
    any wrong factor while sitting well above roundoff."""
    from dpend.model.cart_dynamics import coriolis_matrix, gravity, mass_matrix

    M_s, Cqd_s, g_s = sympy_mcg_cart
    cp = CartParams()
    rng = np.random.default_rng(42)

    max_M_residual = 0.0
    max_Cqd_residual = 0.0
    max_g_residual = 0.0

    for _ in range(10):
        x_cart = rng.uniform(-1.0, 1.0)  # m
        th1, th2 = rng.uniform(-np.pi, np.pi, size=2)
        xdot, w1, w2 = rng.uniform(-4.0, 4.0, size=3)

        # Hand-coded M
        M_hand = mass_matrix(cp, th1, th2)
        M_sympy = np.array(M_s(x_cart, th1, th2, xdot, w1, w2), dtype=float)
        M_diff = np.linalg.norm(M_hand - M_sympy)
        max_M_residual = max(max_M_residual, M_diff)
        np.testing.assert_allclose(
            M_hand, M_sympy,
            atol=1e-9, err_msg="M mismatch vs sympy")

        # Hand-coded C·q̇
        Cqd_hand = coriolis_matrix(cp, th1, th2, w1, w2) @ np.array([xdot, w1, w2])
        Cqd_sympy = np.array(Cqd_s(x_cart, th1, th2, xdot, w1, w2), dtype=float).ravel()
        Cqd_diff = np.linalg.norm(Cqd_hand - Cqd_sympy)
        max_Cqd_residual = max(max_Cqd_residual, Cqd_diff)
        np.testing.assert_allclose(
            Cqd_hand, Cqd_sympy,
            atol=1e-9, err_msg="C·q̇ mismatch vs sympy")

        # Hand-coded g
        g_hand = gravity(cp, th1, th2)
        g_sympy = np.array(g_s(x_cart, th1, th2, xdot, w1, w2), dtype=float).ravel()
        g_diff = np.linalg.norm(g_hand - g_sympy)
        max_g_residual = max(max_g_residual, g_diff)
        np.testing.assert_allclose(
            g_hand, g_sympy,
            atol=1e-9, err_msg="g mismatch vs sympy")

    print(f"\nmax sympy residuals (10 random configs):")
    print(f"  M:   {max_M_residual:.3e}")
    print(f"  C·q̇: {max_Cqd_residual:.3e}")
    print(f"  g:   {max_g_residual:.3e}")


def test_cart_end_stop_dissipation():
    """End-stop dynamics from z0 = [1.2, 0, 0, 2.0, 0, 0] (coasting at +stop).
    At θ=0 gravity and Coriolis vanish, so the pendulum holds until first
    contact; three stop impacts then destabilize it via mass-matrix coupling,
    releasing ~38 J of PE and a deepest excursion max|x| = 1.7327 m on the
    opposite wall. The 1.75 m bound is empirical for this z0 (stable to ~1e-6
    under tolerance and IC perturbation) — changes to z0/k_stop/masses must
    re-derive it. Also: energy monotone non-increasing within each contact
    episode (c_stop dissipation), then flat (<1e-8 rel) on the final in-rail
    second (no damping in-rail)."""
    from dpend.model.cart_dynamics import energy, f

    cp = CartParams()
    u = np.array([0.0])
    z0 = np.array([1.2, 0.0, 0.0, 2.0, 0.0, 0.0])  # [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]

    # Integrate 5 seconds with tight tolerances (contact is stiff-ish)
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

    # Assertion 1: max|x| < L_rail + 0.25 (empirical bound for this z0 — see docstring)
    x_max = np.max(np.abs(x_traj))
    print(f"\nend-stop test:")
    print(f"  max(|x|): {x_max:.4f} m (limit: {cp.L_rail + 0.25:.4f} m)")
    assert x_max < cp.L_rail + 0.25, f"overshoot exceeded: {x_max} > {cp.L_rail + 0.25}"

    # Assertion 2: energy monotone non-increasing within each contact episode.
    # Split the contact indices into contiguous runs — diffing across in-rail
    # gaps would pass only by sign luck.
    idx = np.where(np.abs(x_traj) > cp.L_rail)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    worst_intra_dE = -np.inf
    for run in runs:
        if len(run) > 1:
            dE = np.diff(E_traj[run])
            worst_intra_dE = max(worst_intra_dE, float(np.max(dE)))
            # monotone non-increase, allowing small numerical noise < 1e-10 absolute
            assert np.all(dE < 1e-10), \
                "energy not monotone non-increasing within a contact episode"
    print(f"  contact episodes: {len(runs)}; worst intra-episode energy diff: "
          f"{worst_intra_dE:.3e} J (non-increase check)")

    # Assertion 3: flat energy on final in-rail second (t ∈ [4, 5])
    final_indices = np.where(sol.t >= 4.0)[0]
    if len(final_indices) > 0:
        E_final = E_traj[final_indices]
        # In-rail: no stop contact, only internal friction (zero by default)
        drift_final = np.max(np.abs(E_final - E_final[0])) / max(abs(E_final[0]), 1.0)
        print(f"  final-second energy drift (t ∈ [4, 5]): {drift_final:.3e} (limit: 1e-8)")
        assert drift_final < 1e-8, f"energy not conserved in-rail: drift = {drift_final}"
