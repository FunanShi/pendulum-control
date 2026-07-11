"""LQR — infinite-horizon linear-quadratic regulator.

Linearize about the upright origin, solve the CARE for P (hand-rolled
``riccati.solve_care``, cross-checked against scipy in tests), apply
u = −Kx̂ with K = R⁻¹BᵀP. P doubles as the Lyapunov/cost-to-go matrix.
"""
from __future__ import annotations

import numpy as np

from dpend.controllers.riccati import solve_care
from dpend.model.linearize import ctrb, rank_and_cond
from dpend.reference import ReferenceSource


class LQRController:
    """u = −K (x̂ − z_ref(t)), full-state feedback with optional tracking and
    saturation. Input units per ``plant.input_labels`` (N or N·m).

    K: (m,n) gain — input per unit state error. Built by ``lqr_factory`` as
    R⁻¹BᵀP; this class is pure application, no Riccati solve. Public: the
    UI's RoA supervisor reads it off the instance.
    P: (n,n) SPD CARE solution, or None when built ad hoc with only K.
    Defines V(x) = xᵀPx, the closed loop's cost-to-go/Lyapunov function.
    Public so the UI's V-supervisor triggers on the SAME matrix the
    stability argument uses, not an independent re-derivation.
    z_ref_fn: t [s] → ℝⁿ setpoint, or None to regulate to the origin.
    u_max: None (pure LQR) or an elementwise clip bound [input units].
    Saturation VOIDS the Lyapunov guarantee (V̇ < 0 was derived for the
    unsaturated linear law) — an actuator-limit escape hatch, not a
    substitute for constrained MPC.

    Principal-branch precondition: ``update()`` subtracts z_ref raw, with no
    angle wrapping — angle components of x̂ must already be within ±π of the
    corresponding z_ref/upright component (ModeSwitch re-wraps before every
    catch call; any other caller on a wound angle must wrap first).
    """

    def __init__(self, K: np.ndarray, P: np.ndarray | None = None, z_ref_fn=None, u_max=None):
        self.K = np.asarray(K, dtype=float)
        self.P = None if P is None else np.asarray(P, dtype=float)
        self._z_ref_fn = z_ref_fn
        self._u_max = None if u_max is None else np.asarray(u_max, dtype=float)

    def reset(self, t0: float, x0: np.ndarray) -> None:
        pass  # stateless beyond config: LQR has no memory across ticks

    def update(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        n = self.K.shape[1]
        z_ref = self._z_ref_fn(t) if self._z_ref_fn is not None else np.zeros(n)
        u = -self.K @ (np.asarray(x_hat, dtype=float) - np.asarray(z_ref, dtype=float))
        if self._u_max is not None:
            u = np.clip(u, -self._u_max, self._u_max)
        return u


def lqr_factory(plant, params: dict) -> LQRController:
    """Build an LQRController for `plant` from scenario `params`.

    1. Linearize at the upright equilibrium (plant-generic surface).
    2. Gate on controllability: rank(ctrb(A,B)) == plant.n, else ValueError.
    3. Q/R: `plant.lqr_weights` defaults (per plant, not per state
       dimension), overridable via params["Q"]/["R"] (diagonal lists).
    4. Solve the CARE; K = R⁻¹BᵀP.
    5. Tracking: railed plants only. The rail is a translation-invariant
       DOF — a continuous family of equilibria — so shifting the setpoint
       along it is EXACT, not an approximation: z_ref_fn = e1·ref.r(t).
       Plants without a rail have an isolated equilibrium and ignore any
       reference.
    6. u_max: params.get("u_max"), None by default (see the saturation
       caveat above).
    """
    A, B = plant.linearize(plant.upright)

    rank, cond = rank_and_cond(ctrb(A, B))
    if rank != plant.n:
        raise ValueError(
            f"lqr_factory: {plant.name} plant not controllable at upright — "
            f"rank(ctrb(A,B))={rank} != n={plant.n} (cond={cond:.3e}); LQR "
            "requires a stabilizable (A,B) pair"
        )

    Q_diag = params.get("Q") or plant.lqr_weights[0]
    R_diag = params.get("R") or plant.lqr_weights[1]
    Q = np.diag(np.asarray(Q_diag, dtype=float))
    R = np.diag(np.asarray(R_diag, dtype=float))

    P = solve_care(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)  # R⁻¹ Bᵀ P (m,n), via solve — no explicit inverse

    z_ref_fn = None
    if plant.rail is not None:
        ref = params.get("reference") or ReferenceSource()  # self-default
        e1 = np.zeros(plant.n)
        e1[0] = 1.0
        z_ref_fn = lambda t: e1 * ref.r(t)

    u_max = params.get("u_max")
    if u_max is not None:
        u_max = np.asarray(u_max, dtype=float)

    return LQRController(K, P, z_ref_fn=z_ref_fn, u_max=u_max)
