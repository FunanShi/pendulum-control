"""Cart-mounted double pendulum dynamics.

Closed forms: mass matrix M(θ₁,θ₂) (3×3), Coriolis C(q,q̇) (3×3), gravity
g(q) (3,), end-stop force τ_stop(x,ẋ), forward dynamics ż = f(z, u, cp,
tau_ext), and total energy E [J].

State: z = [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂] ∈ ℝ⁶
  x [m] — cart position, world +x, rail centered at 0
  θ₁, θ₂ [rad] — angles from upright vertical
  ẋ [m/s], θ̇₁, θ̇₂ [rad/s] — velocities
  Upright equilibrium = origin.
"""
from __future__ import annotations

import numpy as np

from dpend.model.cart_params import CartParams
from dpend.model.dynamics import _abd, gravity as fixed_gravity


def mass_matrix(cp: CartParams, theta1: float, theta2: float) -> np.ndarray:
    """Inertia M(θ₁,θ₂) [kg, kg·m²], symmetric PD.

    Block structure:
    M = [ mt              −k₁c₁−k₂c₁₂      −k₂c₁₂    ]
        [ −k₁c₁−k₂c₁₂     a+2b·c₂          d+b·c₂    ]  (lower-right 2×2 ≡ fixed-pivot M)
        [ −k₂c₁₂          d+b·c₂           d         ]

    where mt = mc+m₁+m₂, k₁ = m₁l_c1+m₂l₁, k₂ = m₂l_c2, a/b/d from _abd.
    """
    p = cp.pend
    a, b, d = _abd(p)
    mt = cp.mc + p.m1 + p.m2
    k1 = p.m1 * p.lc1 + p.m2 * p.l1
    k2 = p.m2 * p.lc2

    c1 = np.cos(theta1)
    c2 = np.cos(theta2)
    c12 = np.cos(theta1 + theta2)

    c1_coeff = -(k1 * c1 + k2 * c12)
    c12_coeff = -k2 * c12

    return np.array([
        [mt,           c1_coeff,      c12_coeff        ],
        [c1_coeff,     a + 2.0*b*c2,  d + b*c2         ],
        [c12_coeff,    d + b*c2,      d                ]
    ])


def coriolis_matrix(cp: CartParams, theta1: float, theta2: float,
                     w1: float, w2: float) -> np.ndarray:
    """Coriolis/centrifugal C(q,q̇) [N·s/m, N·m·s·rad⁻¹], Christoffel form.

    C @ [ẋ, θ̇₁, θ̇₂] gives the velocity-dependent torques.

    C = [ 0    k₁s₁θ̇₁+k₂s₁₂(θ̇₁+θ̇₂)    k₂s₁₂(θ̇₁+θ̇₂) ]
        [ 0    −h·θ̇₂                     −h·(θ̇₁+θ̇₂)  ]
        [ 0    h·θ̇₁                      0             ]

    where h = b·sinθ₂, and no ẋ-coupling Coriolis terms (they cancel in EL).
    """
    p = cp.pend
    _, b, _ = _abd(p)
    k1 = p.m1 * p.lc1 + p.m2 * p.l1
    k2 = p.m2 * p.lc2

    s1 = np.sin(theta1)
    s12 = np.sin(theta1 + theta2)
    h = b * np.sin(theta2)

    c_01 = k1 * s1 * w1 + k2 * s12 * (w1 + w2)
    c_02 = k2 * s12 * (w1 + w2)

    return np.array([
        [0.0,        c_01,           c_02          ],
        [0.0,        -h * w2,        -h * (w1+w2)  ],
        [0.0,        h * w1,         0.0           ]
    ])


def gravity(cp: CartParams, theta1: float, theta2: float) -> np.ndarray:
    """Gravity force/torque g(q) = ∂V/∂q [N, N·m].

    Cart gravity is 0 (motion along rail); pendulum gravity from fixed model.
    g = [0, fixed_g(θ₁,θ₂)]ᵀ
    """
    fixed_g = fixed_gravity(cp.pend, theta1, theta2)
    return np.array([0.0, fixed_g[0], fixed_g[1]])


def stop_force(cp: CartParams, x: float, xdot: float) -> float:
    """End-stop restoring force τ_stop_x [N], one per end of rail.

    τ_stop_x(x, ẋ) = −k_stop·(|x|−L) − c_stop·ẋ  if |x| > L_rail, else 0.

    Negative (pushes back) when x > L_rail; positive when x < −L_rail.
    """
    L = cp.L_rail
    if abs(x) > L:
        return -cp.k_stop * (abs(x) - L) * np.sign(x) - cp.c_stop * xdot
    else:
        return 0.0


def f(z: np.ndarray, u: np.ndarray, cp: CartParams, tau_ext: np.ndarray | None = None) -> np.ndarray:
    """Forward dynamics ż = f(z, u, cp, tau_ext).

    z = [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂] [m, rad, rad, m/s, rad/s, rad/s]
    u ∈ ℝ [N] (cart force)
    tau_ext ∈ ℝ³ | None — external generalized forces [N, N·m, N·m]

    q̈ = M⁻¹(B·u + τ_ext + τ_stop − C·q̇ − g − D·q̇),   B = [1,0,0]ᵀ

    where D = diag(bc, b₁, b₂) viscous damping.
    """
    x, theta1, theta2 = float(z[0]), float(z[1]), float(z[2])
    xdot, w1, w2 = float(z[3]), float(z[4]), float(z[5])

    M = mass_matrix(cp, theta1, theta2)
    C = coriolis_matrix(cp, theta1, theta2, w1, w2)

    qdot = np.array([xdot, w1, w2])

    tau = np.array([float(u[0]), 0.0, 0.0])

    if tau_ext is None:
        tau_ext = np.zeros(3)
    else:
        tau_ext = np.asarray(tau_ext, dtype=float)

    tau_stop = np.array([stop_force(cp, x, xdot), 0.0, 0.0])

    g = gravity(cp, theta1, theta2)

    D_qdot = np.array([cp.bc * xdot, cp.pend.b1 * w1, cp.pend.b2 * w2])

    qddot = np.linalg.solve(M, tau + tau_ext + tau_stop - C @ qdot - g - D_qdot)

    return np.concatenate([qdot, qddot])


def energy(cp: CartParams, z: np.ndarray) -> float:
    """Total energy E = ½q̇ᵀM(θ₂)q̇ + V(θ₁,θ₂) + ½k_stop·(|x|−L)²·𝟙[|x|>L] [J].

    Kinetic energy in all three DOF; potential V is pendulum only (cart has no PE);
    stop springs contribute when in contact (|x| > L_rail).
    """
    x, theta1, theta2 = float(z[0]), float(z[1]), float(z[2])
    xdot, w1, w2 = float(z[3]), float(z[4]), float(z[5])

    qdot = np.array([xdot, w1, w2])
    M = mass_matrix(cp, theta1, theta2)
    T = 0.5 * float(qdot @ (M @ qdot))

    # V = g₀((m₁l_c1+m₂l₁)cosθ₁ + m₂l_c2·cos(θ₁+θ₂))  (pendulum only)
    p = cp.pend
    k1 = p.m1 * p.lc1 + p.m2 * p.l1
    k2 = p.m2 * p.lc2
    V = p.g0 * (k1 * np.cos(theta1) + k2 * np.cos(theta1 + theta2))

    L = cp.L_rail
    if abs(x) > L:
        V_stop = 0.5 * cp.k_stop * (abs(x) - L)**2
    else:
        V_stop = 0.0

    return T + V + V_stop
