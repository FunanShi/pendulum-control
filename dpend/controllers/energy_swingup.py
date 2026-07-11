"""Energy-shaping swing-up (phase 2).

Pumps the pole from hanging toward upright by driving its subsystem energy
(``plant.pendulum_energy``) to the upright value with Åström–Furuta energy
feedback, realized as a cart acceleration through the plant's collocated
PFL inverse (``plant.accel_to_force``). Works on any ``EnergyShapingCapable``
plant; hands off to the LQR/MPC catch via ``mode_switch``. The plant is
injected — this module imports the protocol under TYPE_CHECKING only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dpend.model.plant import EnergyShapingCapable


class EnergySwingUp:
    """Åström–Furuta energy shaping + collocated PFL + cart centering, for
    any ``EnergyShapingCapable`` plant.

    z = [x, θ, ẋ, θ̇]: x [m], θ [rad] from the upward vertical (upright = 0,
    hanging = π), ẋ [m/s], θ̇ [rad/s].

    Per tick, the desired cart acceleration is

        a_cmd = k_E·(E_up − E)·θ̇·cosθ  −  k_x·x  −  k_d·ẋ      [m/s²]

    Energy pump: E = plant.pendulum_energy(z), E_up = plant.energy_upright.
    Along the collocated frictionless dynamics Ė = b·θ̇·cosθ·a_cmd
    (b = m_p·l), so this term gives Ė = k_E·b·(E_up−E)·θ̇²·cos²θ — the same
    sign as (E_up − E) for any k_E > 0, i.e. E approaches E_up monotonically
    from either side. Built from cosθ, the law is smooth on all of S¹ — no
    branch cut.

    Cart centering: −k_x·x − k_d·ẋ is an additive, cart-only PD pull toward
    rail center so the pump doesn't walk into the end stops; it does not
    perturb the energy argument above.

    a_cmd is realized exactly (frictionless, in-rail) by
    ``plant.accel_to_force``; the optional ±u_max [N] clip breaks that
    exactness near saturation (same escape-hatch caveat as
    LQRController.u_max).

    Gains (measured, grid-searched; chosen for rail margin — max|x| =
    0.854 m of the 1.5 m rail vs a candidate with marginally better min-V
    but 96% rail usage — and a monotonic approach into the catch basin):
      k_E = 1.5 [(m/s²)/(J·rad/s)], k_x = 4.0 [1/s²], k_d = 3.0 [1/s],
      u_max = None (measured peak |u| ≈ 21 N from hanging).

    Deterministic dead time from exact hanging [0, π, 0, 0]: the pump term
    is ∝ θ̇ = 0, so the only symmetry-breaking seed is gravity's residual
    torque from sin(π) ≈ 1.2e-16 — measured ~5 s before the swing is
    visible, independent of k_E, fully reproducible. Any noise or friction
    breaks the symmetry immediately on real hardware.

    Stateless: reset() is a no-op — update() is a pure function of the
    current x_hat — so ModeSwitch can reset it on every SWINGING re-entry
    freely.
    """

    def __init__(self, plant: "EnergyShapingCapable", k_E: float = 1.5,
                 k_x: float = 4.0, k_d: float = 3.0, u_max: float | None = None):
        self.plant = plant
        self.k_E = float(k_E)
        self.k_x = float(k_x)
        self.k_d = float(k_d)
        self.u_max = None if u_max is None else float(u_max)

    def reset(self, t0: float, x0: np.ndarray) -> None:
        pass  # stateless: every update() output is a pure function of x_hat

    def update(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        z = np.asarray(x_hat, dtype=float)
        x, theta, xdot, thetadot = float(z[0]), float(z[1]), float(z[2]), float(z[3])

        E = self.plant.pendulum_energy(z)
        E_up = self.plant.energy_upright

        a_cmd = (self.k_E * (E_up - E) * thetadot * np.cos(theta)
                 - self.k_x * x - self.k_d * xdot)

        u = self.plant.accel_to_force(z, a_cmd)
        if self.u_max is not None:
            u = float(np.clip(u, -self.u_max, self.u_max))
        return np.array([u])
