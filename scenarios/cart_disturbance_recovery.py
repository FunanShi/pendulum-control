"""Disturbance-rejection demo: the cart-mounted double pendulum, balanced
upright by LQR, is shoved by a brief impulse force on the cart — the cart
lurches ~0.6 m and both links swing ~20 deg, then everything returns to
upright and centre.

The shove (180 N x 50 ms) sits just inside the naked LQR's region of
attraction: 190 N still recovers, 200 N tips over the top (measured force
sweep). The ~0.6 m cart peak stays well inside the +/-1.5 m rail, so no
end-stop is involved.

Run: python batch.py scenarios/cart_disturbance_recovery.py --save-anim
"""
import numpy as np

from dpend.config import Scenario

_KICK_T_S, _KICK_DUR_S, _KICK_F_N = 0.8, 0.05, 180.0  # impulse "shove": force on the cart [N]


def _shove(t, x):
    """Brief +x impulse on the cart at t=_KICK_T_S.

    Returns the generalized external force τ_ext = [Fx, τ1, τ2] ∈ ℝ³
    [N, N·m, N·m] (only the cart is shoved); zeros otherwise. `x` is unused
    (the disturbance is open-loop).
    """
    if _KICK_T_S <= t < _KICK_T_S + _KICK_DUR_S:
        return np.array([_KICK_F_N, 0.0, 0.0])
    return np.zeros(3)


scenario = Scenario(
    plant="cart",
    controller="lqr",
    # balanced upright: [x, θ1, θ2, ẋ, θ̇1, θ̇2]  [m, rad, rad, m/s, rad/s, rad/s]
    x0=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    duration_s=4.5,
    disturbance=_shove,
)
