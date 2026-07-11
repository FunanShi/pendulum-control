"""Single-pole cart-pole dynamics.

Closed forms: mass matrix M(theta) (2x2), Coriolis C(theta,thetadot) (2x2),
gravity g(theta) (2,), end-stop force tau_stop(x,xdot), forward dynamics
zdot = f(z,u,cp,tau_ext), total energy E [J], and the energy-shaping
primitives (pendulum_energy, energy_upright, accel_to_force) the swing-up
controller consumes via the EnergyShapingCapable protocol (model/plant.py).

State: z = [x, theta, xdot, thetadot] in R^4
  x [m]            -- cart position, world +x, rail centered at 0
  theta [rad]       -- pole angle from the upward vertical (upright=0, hanging=pi)
  xdot [m/s], thetadot [rad/s] -- velocities
  Upright equilibrium = origin.

Lumped constants (used throughout): mt = mc+mp (total translating mass),
b = mp*l (pole's first moment about the pivot -- its gravity/coupling
moment arm), J = mp*l**2 + I (pole's inertia about the pivot, via the
parallel-axis theorem from I, its inertia about its own COM).
"""
from __future__ import annotations

import numpy as np

from dpend.model.cart_pole_params import CartPoleParams


def _mt_b_J(cp: CartPoleParams) -> tuple[float, float, float]:
    """Lumped constants (mt, b, J) -- defined in the module docstring."""
    mt = cp.mc + cp.mp
    b = cp.mp * cp.l
    J = cp.mp * cp.l**2 + cp.I
    return mt, b, J


def mass_matrix(cp: CartPoleParams, theta: float) -> np.ndarray:
    """Inertia M(theta) [kg, kg*m^2], symmetric PD.

    M = [[ mt,            -b*cos(theta) ],
         [ -b*cos(theta),  J            ]]
    """
    mt, b, J = _mt_b_J(cp)
    c = np.cos(theta)
    return np.array([[mt,      -b * c],
                      [-b * c,   J]])


def coriolis_matrix(cp: CartPoleParams, theta: float, thetadot: float) -> np.ndarray:
    """Coriolis/centrifugal C(theta,thetadot) [N*s/m, N*m*s/rad], Christoffel form.

    C @ [xdot, thetadot] gives the velocity-dependent torques:
        C = [[0, b*sin(theta)*thetadot],
             [0, 0                    ]]
    so C @ qdot = [b*sin(theta)*thetadot**2, 0] -- the centrifugal kick the
    swinging pole exerts back on the cart. Mdot - 2C is skew (passivity;
    checked in tests).
    """
    b = cp.mp * cp.l
    s = np.sin(theta)
    return np.array([[0.0, b * s * thetadot],
                      [0.0, 0.0]])


def gravity(cp: CartPoleParams, theta: float) -> np.ndarray:
    """Gravity force/torque g(q) = dV/dq [N, N*m], V = mp*g0*l*cos(theta).

    g = [0, -mp*g0*l*sin(theta)]^T. V is maximal at upright (theta=0) --
    the destabilizing potential that makes upright unstable and hanging
    marginally stable (same sign convention as the fixed-pivot/cart plants:
    angles measured from the upward vertical).
    """
    return np.array([0.0, -cp.mp * cp.g0 * cp.l * np.sin(theta)])


def stop_force(cp: CartPoleParams, x: float, xdot: float) -> float:
    """End-stop restoring force tau_stop_x [N], one per end of rail.

    tau_stop_x(x,xdot) = -k_stop*(x - sgn(x)*L) - c_stop*xdot  if |x| > L_rail,
    else 0. Negative (pushes back) when x > L_rail; positive when x < -L_rail.
    """
    L = cp.L_rail
    if abs(x) > L:
        return -cp.k_stop * (abs(x) - L) * np.sign(x) - cp.c_stop * xdot
    return 0.0


def f(z: np.ndarray, u: np.ndarray, cp: CartPoleParams,
      tau_ext: np.ndarray | None = None) -> np.ndarray:
    """Forward dynamics zdot = f(z, u, cp, tau_ext).

    z = [x, theta, xdot, thetadot] [m, rad, m/s, rad/s]
    u in R [N] (cart force)
    tau_ext in R^2 | None -- external generalized forces [N, N*m]

    q_ddot = M^-1(B*u + tau_ext + tau_stop - C*qdot - g - D*qdot), B=[1,0]^T,
    where D = diag(bc, bp) viscous damping.
    """
    x, theta = float(z[0]), float(z[1])
    xdot, thetadot = float(z[2]), float(z[3])

    M = mass_matrix(cp, theta)
    C = coriolis_matrix(cp, theta, thetadot)
    qdot = np.array([xdot, thetadot])

    tau = np.array([float(u[0]), 0.0])

    if tau_ext is None:
        tau_ext = np.zeros(2)
    else:
        tau_ext = np.asarray(tau_ext, dtype=float)

    tau_stop = np.array([stop_force(cp, x, xdot), 0.0])
    g = gravity(cp, theta)
    D_qdot = np.array([cp.bc * xdot, cp.bp * thetadot])

    qddot = np.linalg.solve(M, tau + tau_ext + tau_stop - C @ qdot - g - D_qdot)
    return np.concatenate([qdot, qddot])


def energy(cp: CartPoleParams, z: np.ndarray) -> float:
    """Total mechanical energy [J]:
    E = 1/2 qdot^T M(theta) qdot + V + 1/2 k_stop(|x|-L)^2 * 1[|x|>L].

    Kinetic energy in both DOF; potential V is pole-only (cart has no PE);
    the stop spring contributes only while in contact (|x| > L_rail).
    """
    x, theta = float(z[0]), float(z[1])
    xdot, thetadot = float(z[2]), float(z[3])

    qdot = np.array([xdot, thetadot])
    M = mass_matrix(cp, theta)
    T = 0.5 * float(qdot @ (M @ qdot))

    V = cp.mp * cp.g0 * cp.l * np.cos(theta)

    L = cp.L_rail
    if abs(x) > L:
        V_stop = 0.5 * cp.k_stop * (abs(x) - L) ** 2
    else:
        V_stop = 0.0

    return T + V + V_stop


def pendulum_energy(cp: CartPoleParams, z: np.ndarray) -> float:
    """Pole-subsystem energy about the fixed pivot (Astrom-Furuta energy
    shaping): E_pend(z) = 1/2*J*thetadot**2 + mp*g0*l*cos(theta) [J].

    Deliberately not the plant's total mechanical energy (`energy` above,
    which also counts the cart's kinetic energy and any stop-spring PE) --
    this is the pole-only quantity the swing-up controller pumps toward
    `energy_upright(cp)` via `accel_to_force`.
    """
    J = cp.mp * cp.l**2 + cp.I
    theta, thetadot = float(z[1]), float(z[3])
    return 0.5 * J * thetadot**2 + cp.mp * cp.g0 * cp.l * np.cos(theta)


def energy_upright(cp: CartPoleParams) -> float:
    """Pendulum-subsystem energy at the upright equilibrium: E_up = mp*g0*l [J].

    pendulum_energy at hanging rest = -mp*g0*l, so swing-up must inject
    2*mp*g0*l to go from hanging rest to upright rest.
    """
    return cp.mp * cp.g0 * cp.l


def accel_to_force(cp: CartPoleParams, z: np.ndarray, a_cmd: float) -> float:
    """Collocated partial-feedback-linearization (PFL) inverse: the cart
    force u [N] that produces a desired cart acceleration a_cmd [m/s^2]
    exactly (frictionless, in-rail, no tau_ext/tau_stop).

    From the theta-row of the EOM with q_ddot[0] = a_cmd, then the x-row:
        theta_ddot = (b*cos(theta)*a_cmd + mp*g0*l*sin(theta)) / J
        u = mt*a_cmd - b*cos(theta)*theta_ddot + b*sin(theta)*thetadot**2

    "Collocated": inverts through the actuated coordinate (x), unlike the
    Acrobot's non-collocated PFL. Round-tripping through `f` reproduces
    a_cmd to ~1e-9 (tests). See docs/design-notes/energy-swingup.md.
    """
    mt, b, J = _mt_b_J(cp)
    theta, thetadot = float(z[1]), float(z[3])
    c, s = np.cos(theta), np.sin(theta)

    thetaddot = (b * c * a_cmd + cp.mp * cp.g0 * cp.l * s) / J
    u = mt * a_cmd - b * c * thetaddot + b * s * thetadot**2
    return float(u)
