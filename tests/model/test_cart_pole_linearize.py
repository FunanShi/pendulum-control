"""Cart-pole linearization: analytic vs central FD, stability, controllability.
Mirrors test_cart_linearize.py for the single-pole plant."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.model.cart_pole_params import CartPoleParams


def _fd_jacobians(cp, z_eq, eps=1e-6):
    """Central finite differences of f around (z_eq, 0). eps=1e-6 balances
    truncation (~eps^2) against roundoff (~1e-16/eps): error ~1e-10*|f'''|
    (same rationale as test_cart_linearize.py)."""
    from dpend.model.cart_pole_dynamics import f

    u0 = np.zeros(1)
    A = np.zeros((4, 4))
    for j in range(4):
        dz = np.zeros(4)
        dz[j] = eps
        A[:, j] = (f(z_eq + dz, u0, cp) - f(z_eq - dz, u0, cp)) / (2 * eps)
    B_lin = np.zeros((4, 1))
    du = np.zeros(1)
    du[0] = eps
    B_lin[:, 0] = (f(z_eq, u0 + du, cp) - f(z_eq, u0 - du, cp)) / (2 * eps)
    return A, B_lin


@pytest.mark.parametrize("eq_name", ["UPRIGHT", "HANGING"])
def test_linearize_matches_finite_differences(eq_name):
    import dpend.model.cart_pole_linearize as lin

    cp = CartPoleParams()
    z_eq = getattr(lin, eq_name)
    A, B_lin = lin.linearize(cp, z_eq)
    A_fd, B_lin_fd = _fd_jacobians(cp, z_eq)
    np.testing.assert_allclose(A, A_fd, atol=1e-7)
    np.testing.assert_allclose(B_lin, B_lin_fd, atol=1e-7)


def test_upright_unstable_hanging_marginal():
    """Physics pin: upright linearization has a right-half-plane eigenvalue;
    hanging (frictionless) is marginally stable -- max real part ~= 0."""
    import dpend.model.cart_pole_linearize as lin

    cp = CartPoleParams()
    A_up, _ = lin.linearize(cp, lin.UPRIGHT)
    A_dn, _ = lin.linearize(cp, lin.HANGING)
    eig_up = np.linalg.eigvals(A_up)
    eig_dn = np.linalg.eigvals(A_dn)
    max_real_up = np.max(eig_up.real)
    max_real_dn = np.max(eig_dn.real)
    print(f"\nUpright max re(lambda) = {max_real_up:.6e}")
    print(f"Hanging max re(lambda) = {max_real_dn:.6e}")
    assert max_real_up > 1.0   # strongly unstable
    assert max_real_dn < 1e-9  # marginal


def test_controllable_at_upright():
    """Controllability at the upright — rank 4. Condition number is printed
    for comparison, not asserted."""
    import dpend.model.cart_pole_linearize as lin
    import dpend.model.linearize as lin_fixed

    cp = CartPoleParams()
    A, B_lin = lin.linearize(cp, lin.UPRIGHT)
    rank, cond = lin_fixed.rank_and_cond(lin_fixed.ctrb(A, B_lin))
    print(f"\nctrb cart-pole: rank={rank}, cond={cond:.3e}")
    assert rank == 4
