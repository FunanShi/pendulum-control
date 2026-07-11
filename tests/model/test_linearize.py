"""Fixed-pivot linearization: analytic A,B vs central finite differences,
stability character at both equilibria, controllability per actuation config."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.model.params import Params, actuation_matrix


def _fd_jacobians(p, B, x_eq, eps=1e-6):
    """Central finite differences of f around (x_eq, 0). eps=1e-6 balances
    truncation (∝eps²) against roundoff (∝1e-16/eps): error ~1e-10·|f'''|."""
    from dpend.model.dynamics import f

    m = B.shape[1]
    u0 = np.zeros(m)
    A = np.zeros((4, 4))
    for j in range(4):
        dx = np.zeros(4); dx[j] = eps
        A[:, j] = (f(x_eq + dx, u0, p, B) - f(x_eq - dx, u0, p, B)) / (2 * eps)
    Bl = np.zeros((4, m))
    for j in range(m):
        du = np.zeros(m); du[j] = eps
        Bl[:, j] = (f(x_eq, u0 + du, p, B) - f(x_eq, u0 - du, p, B)) / (2 * eps)
    return A, Bl


@pytest.mark.parametrize("config", ["full", "acrobot", "pendubot"])
@pytest.mark.parametrize("eq_name", ["UPRIGHT", "HANGING"])
def test_linearize_matches_finite_differences(config, eq_name):
    import dpend.model.linearize as lin

    p = Params()
    B = actuation_matrix(config)
    x_eq = getattr(lin, eq_name)
    A, Bl = lin.linearize(p, B, x_eq)
    A_fd, Bl_fd = _fd_jacobians(p, B, x_eq)
    np.testing.assert_allclose(A, A_fd, atol=1e-7)
    np.testing.assert_allclose(Bl, Bl_fd, atol=1e-7)


def test_upright_unstable_hanging_marginal():
    """Physics pin: upright linearization has a right-half-plane eigenvalue;
    hanging (frictionless) is marginally stable — max real part ≈ 0."""
    import dpend.model.linearize as lin

    p = Params()
    B = actuation_matrix("acrobot")
    A_up, _ = lin.linearize(p, B, lin.UPRIGHT)
    A_dn, _ = lin.linearize(p, B, lin.HANGING)
    assert np.max(np.linalg.eigvals(A_up).real) > 1.0   # strongly unstable (1/s)
    assert np.max(np.linalg.eigvals(A_dn).real) < 1e-9  # marginal (pure oscillation)


@pytest.mark.parametrize("config", ["full", "acrobot", "pendubot"])
def test_controllable_at_upright(config):
    """All three configs are controllable at the upright for the default
    parameters — rank 4. Condition numbers are printed for comparison, not
    asserted (measure before claiming orderings)."""
    import dpend.model.linearize as lin

    p = Params()
    B = actuation_matrix(config)
    A, Bl = lin.linearize(p, B, lin.UPRIGHT)
    rank, cond = lin.rank_and_cond(lin.ctrb(A, Bl))
    print(f"\nctrb {config:9s}: rank={rank}, cond={cond:.3e}")
    assert rank == 4


def test_ctrb_obsv_dimension_generic():
    """ctrb/obsv take n from their arguments, not a hardcoded 4 — checked
    with a 6-state random system."""
    import dpend.model.linearize as lin

    rng = np.random.default_rng(1)
    A6 = rng.normal(size=(6, 6))
    B6 = rng.normal(size=(6, 1))
    C6 = rng.normal(size=(2, 6))
    assert lin.ctrb(A6, B6).shape == (6, 6)
    assert lin.obsv(A6, C6).shape == (12, 6)


def test_angle_only_observability():
    """Measuring only (θ₁,θ₂) — C = [I₂ 0₂] — is observable at the upright
    (velocities reconstructible), rank 4."""
    import dpend.model.linearize as lin

    p = Params()
    B = actuation_matrix("acrobot")
    A, _ = lin.linearize(p, B, lin.UPRIGHT)
    C = np.hstack([np.eye(2), np.zeros((2, 2))])
    rank, _ = lin.rank_and_cond(lin.obsv(A, C))
    assert rank == 4
