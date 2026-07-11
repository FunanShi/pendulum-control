"""RK4 correctness: 4th-order convergence on a known system, trajectory match
vs a scipy reference on the real plant, and the measured energy drift (RK4 is
not symplectic; the drift is bounded, not pretended zero)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from dpend.model.params import Params, actuation_matrix


def test_rk4_fourth_order_convergence():
    """On ẋ = Ax (harmonic oscillator) halving dt must cut global error ~16×.
    Accept 12–20: exact 16 only in the asymptotic limit."""
    from dpend.sim.integrators import rk4_step

    A = np.array([[0.0, 1.0], [-1.0, 0.0]])  # period 2π, exact solution known
    dyn = lambda x, u: A @ x
    x0 = np.array([1.0, 0.0])

    def err(dt):
        n = round(2 * np.pi / dt)
        x = x0.copy()
        for _ in range(n):
            x = rk4_step(dyn, x, np.zeros(0), dt)
        return np.linalg.norm(x - x0)  # one full period returns to start

    e1, e2 = err(2 * np.pi / 200), err(2 * np.pi / 400)
    ratio = e1 / e2
    print(f"\nRK4 error ratio on dt halving: {ratio:.2f} (theory: 16)")
    assert 12.0 < ratio < 20.0


def test_rk4_matches_scipy_reference_on_pendulum():
    """2 s free swing: RK4 @ 1 kHz vs DOP853 @ rtol 1e-11, atol 1e-6.
    Horizon deliberately short: the double pendulum is chaotic, so per-step
    float differences grow ~e^(λt) — at 2 s the amplification is benign and a
    failure means a real integrator bug, not chaos."""
    from dpend.model.dynamics import f
    from dpend.sim.integrators import rk4_rollout_zoh

    p = Params()
    B = actuation_matrix("acrobot")
    u0 = np.zeros(1)
    dyn = lambda x, u: f(x, u, p, B)
    x0 = np.array([0.4, -0.3, 0.0, 0.0])

    x_rk4 = rk4_rollout_zoh(dyn, x0, u0, 1e-3, 2000)  # 2 s at 1 kHz
    sol = solve_ivp(lambda t, x: f(x, u0, p, B), (0.0, 2.0), x0,
                    method="DOP853", rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(x_rk4, sol.y[:, -1], atol=1e-6)


def test_rk4_energy_drift_measured():
    """|ΔE|/|E₀| over 10 s at 1 kHz stays under 1e-6 — deliberately loose
    (measured drift is typically far smaller): catches term/sign errors while
    tolerating RK4's non-symplectic creep. The measured value is printed."""
    from dpend.model.dynamics import energy, f
    from dpend.sim.integrators import rk4_step

    p = Params()
    B = actuation_matrix("acrobot")
    u0 = np.zeros(1)
    dyn = lambda x, u: f(x, u, p, B)
    x = np.array([0.4, -0.3, 0.0, 0.0])
    E0 = energy(p, x)
    worst = 0.0
    for _ in range(10_000):  # 10 s @ 1 kHz
        x = rk4_step(dyn, x, u0, 1e-3)
        worst = max(worst, abs(energy(p, x) - E0))
    drift = worst / max(abs(E0), 1.0)
    print(f"\nRK4 relative energy drift over 10 s @ 1 kHz: {drift:.3e}")
    assert drift < 1e-6
