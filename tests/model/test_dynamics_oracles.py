"""Independent oracles for the fixed-pivot dynamics: scipy reference-
integrator energy check and sympy Lagrangian re-derivation. No code shared
with dpend.model.dynamics beyond the parameter values."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dpend.model.params import Params, actuation_matrix


def test_energy_conserved_under_reference_integrator():
    """u=0, friction=0 ⇒ E constant. DOP853 at rtol=1e-11 makes integrator
    error negligible; tolerance 1e-8 relative is ~1000× that error but ~10⁶×
    below any sign/term error in M, C, or g."""
    from dpend.model.dynamics import energy, f

    p = Params()
    B = actuation_matrix("acrobot")
    u0 = np.zeros(1)
    x0 = np.array([0.4, -0.3, 0.0, 0.0])  # rad — well off equilibrium, swings hard

    sol = solve_ivp(
        lambda t, x: f(x, u0, p, B),
        (0.0, 10.0),  # s
        x0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-12,
        dense_output=False,
        t_eval=np.linspace(0.0, 10.0, 501),
    )
    assert sol.success
    E = np.array([energy(p, x) for x in sol.y.T])
    drift = np.max(np.abs(E - E[0])) / max(abs(E[0]), 1.0)
    print(f"\nreference-integrator relative energy drift over 10 s: {drift:.3e}")
    assert drift < 1e-8


@pytest.fixture(scope="module")
def sympy_mcg():
    """Independently derive M, C·q̇, g via sympy Euler–Lagrange and return
    lambdified callables (theta1, theta2, w1, w2) -> arrays."""
    import sympy as sp

    t = sp.symbols("t")
    m1, m2, l1, lc1, lc2, I1, I2, g0 = sp.symbols(
        "m1 m2 l1 lc1 lc2 I1 I2 g0", positive=True
    )
    th1 = sp.Function("th1")(t)
    th2 = sp.Function("th2")(t)

    # World frame: x right, y up; angles from +y (upward vertical), CCW+.
    # A point at distance r along a link at absolute angle θ: (−r·sinθ, r·cosθ).
    x1, y1 = -lc1 * sp.sin(th1), lc1 * sp.cos(th1)
    x2 = -l1 * sp.sin(th1) - lc2 * sp.sin(th1 + th2)
    y2 = l1 * sp.cos(th1) + lc2 * sp.cos(th1 + th2)

    w1s, w2s = sp.diff(th1, t), sp.diff(th2, t)
    v1sq = sp.diff(x1, t) ** 2 + sp.diff(y1, t) ** 2
    v2sq = sp.diff(x2, t) ** 2 + sp.diff(y2, t) ** 2
    T = (
        sp.Rational(1, 2) * m1 * v1sq
        + sp.Rational(1, 2) * I1 * w1s**2
        + sp.Rational(1, 2) * m2 * v2sq
        + sp.Rational(1, 2) * I2 * (w1s + w2s) ** 2  # link 2 spins at θ̇₁+θ̇₂ (θ₂ relative)
    )
    V = g0 * (m1 * y1 + m2 * y2)
    L = T - V

    q = [th1, th2]
    qd = [w1s, w2s]
    qdd = [sp.diff(v, t) for v in qd]
    EL = sp.Matrix(
        [sp.simplify(sp.diff(sp.diff(L, qd_i), t) - sp.diff(L, q_i)) for q_i, qd_i in zip(q, qd)]
    )  # EL = M q̈ + Cq̇ + g  (τ with u=0)

    Msym = EL.jacobian(sp.Matrix(qdd))
    g_sym = EL.subs([(a, 0) for a in qdd + qd])          # q̈=q̇=0 ⇒ gravity only
    Cqd_sym = sp.simplify(EL - Msym * sp.Matrix(qdd) - g_sym)  # velocity terms

    subs = {m1: 1.0, m2: 1.0, l1: 1.0, lc1: 0.5, lc2: 0.5,
            I1: 1.0 / 12.0, I2: 1.0 / 12.0, g0: 9.81}
    th1_, th2_, w1_, w2_ = sp.symbols("th1_ th2_ w1_ w2_")
    repl = [(sp.diff(th1, t), w1_), (sp.diff(th2, t), w2_), (th1, th1_), (th2, th2_)]

    def lam(expr):
        return sp.lambdify((th1_, th2_, w1_, w2_), expr.subs(subs).subs(repl), "numpy")

    return lam(Msym), lam(Cqd_sym), lam(g_sym)


def test_hand_coded_mcg_match_sympy(sympy_mcg):
    from dpend.model.dynamics import coriolis_matrix, gravity, mass_matrix

    M_s, Cqd_s, g_s = sympy_mcg
    p = Params()
    rng = np.random.default_rng(42)
    for _ in range(10):
        th1, th2 = rng.uniform(-np.pi, np.pi, size=2)
        w1, w2 = rng.uniform(-4.0, 4.0, size=2)
        np.testing.assert_allclose(
            mass_matrix(p, th2), np.array(M_s(th1, th2, w1, w2), dtype=float),
            atol=1e-9, err_msg="M mismatch vs sympy")
        np.testing.assert_allclose(
            coriolis_matrix(p, th2, w1, w2) @ np.array([w1, w2]),
            np.array(Cqd_s(th1, th2, w1, w2), dtype=float).ravel(),
            atol=1e-9, err_msg="C·q̇ mismatch vs sympy")
        np.testing.assert_allclose(
            gravity(p, th1, th2), np.array(g_s(th1, th2, w1, w2), dtype=float).ravel(),
            atol=1e-9, err_msg="g mismatch vs sympy")
