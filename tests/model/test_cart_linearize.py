"""Cart linearization: analytic vs central FD, stability, controllability."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.model.cart_params import CartParams


def _fd_jacobians(cp, z_eq, eps=1e-6):
    """Central finite differences of f around (z_eq, 0). eps=1e-6 balances
    truncation (∝eps²) against roundoff (∝1e-16/eps): error ~1e-10·|f'''|."""
    from dpend.model.cart_dynamics import f

    u0 = np.zeros(1)
    A = np.zeros((6, 6))
    for j in range(6):
        dz = np.zeros(6)
        dz[j] = eps
        A[:, j] = (f(z_eq + dz, u0, cp) - f(z_eq - dz, u0, cp)) / (2 * eps)
    B_lin = np.zeros((6, 1))
    du = np.zeros(1)
    du[0] = eps
    B_lin[:, 0] = (f(z_eq, u0 + du, cp) - f(z_eq, u0 - du, cp)) / (2 * eps)
    return A, B_lin


@pytest.mark.parametrize("eq_name", ["UPRIGHT", "HANGING"])
def test_linearize_matches_finite_differences(eq_name):
    import dpend.model.cart_linearize as lin

    cp = CartParams()
    z_eq = getattr(lin, eq_name)
    A, B_lin = lin.linearize(cp, z_eq)
    A_fd, B_lin_fd = _fd_jacobians(cp, z_eq)
    np.testing.assert_allclose(A, A_fd, atol=1e-7)
    np.testing.assert_allclose(B_lin, B_lin_fd, atol=1e-7)


def test_upright_unstable_hanging_marginal():
    """Physics pin: upright linearization has a right-half-plane eigenvalue;
    hanging (frictionless) is marginally stable — max real part ≈ 0."""
    import dpend.model.cart_linearize as lin

    cp = CartParams()
    A_up, _ = lin.linearize(cp, lin.UPRIGHT)
    A_dn, _ = lin.linearize(cp, lin.HANGING)
    eig_up = np.linalg.eigvals(A_up)
    eig_dn = np.linalg.eigvals(A_dn)
    max_real_up = np.max(eig_up.real)
    max_real_dn = np.max(eig_dn.real)
    print(f"\nUpright max re(λ) = {max_real_up:.6e}")
    print(f"Hanging max re(λ) = {max_real_dn:.6e}")
    assert max_real_up > 1.0   # strongly unstable
    assert max_real_dn < 1e-9  # marginal


def test_controllable_at_upright():
    """Controllability at the upright — rank 6. Condition number is printed
    for comparison, not asserted."""
    import dpend.model.cart_linearize as lin
    import dpend.model.linearize as lin_fixed

    cp = CartParams()
    A, B_lin = lin.linearize(cp, lin.UPRIGHT)
    rank, cond = lin_fixed.rank_and_cond(lin_fixed.ctrb(A, B_lin))
    print(f"\nctrb cart: rank={rank}, cond={cond:.3e}")
    assert rank == 6
