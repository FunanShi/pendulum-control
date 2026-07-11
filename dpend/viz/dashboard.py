"""Time-series dashboard from a Telemetry log.

Plots generalized coordinates, their rates, control input(s), and total
energy vs time — the telemetry-first view for tuning and debugging. Generic
over any Plant: panel legends carry caller-supplied state/input labels
(name + units), so this module never knows which plant produced the log.

Depends on: telemetry, matplotlib. Never imports sim/controllers/model.
"""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from dpend.telemetry.recorder import Telemetry


def _maybe_unwrap(series: np.ndarray, label: str) -> np.ndarray:
    """`np.unwrap(series)` when `label`'s unit suffix is exactly `"[rad]"`
    (an angle — `"[rad/s]"` does not match), else `series` unchanged. Same
    "angle iff endswith '[rad]'" convention `registry.swingup_factory` uses —
    kept independent (viz never imports registry) but deliberately the same
    string test.

    Angles are recorded wrapped to (-pi, pi], so a swing-up's continuous
    rotation sawtooths at every ±pi crossing if plotted raw; `np.unwrap`
    removes those synthetic 2π jumps. A plotting transform only — telemetry
    and non-angle series are untouched.
    """
    return np.unwrap(series) if label.endswith("[rad]") else series


def dashboard(tel: Telemetry, state_labels: tuple, input_labels: tuple,
              save_path=None, show: bool = True) -> None:
    """Four stacked time-series panels from telemetry: the tuning/debug view.

    state_labels: length-n tuple of "name [units]" strings for the x_true
    columns, in the plant's own [q, q̇] order (e.g. ("θ1 [rad]", "θ2 [rad]",
    "θ̇1 [rad/s]", "θ̇2 [rad/s]")). Split at n//2: the first half routes to
    the position panel, the rest to the rate panel; angle components go
    through `_maybe_unwrap`. input_labels: length-m tuple for the u columns
    (e.g. ("u0 [N·m]",) or ("F [N]",)). Both are supplied by the caller
    (batch.py, from the resolved Plant) — this module stays plant-agnostic.
    """
    n_q = len(state_labels) // 2
    t = tel.t_ns * 1e-9  # ns → s for plotting
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 9))

    for i in range(n_q):
        series = _maybe_unwrap(tel.x_true[:, i], state_labels[i])
        axes[0].plot(t, series, label=state_labels[i])
    axes[0].set_ylabel("position")

    for i in range(n_q, len(state_labels)):
        axes[1].plot(t, tel.x_true[:, i], label=state_labels[i])  # rates: never unwrapped (see _maybe_unwrap)
    axes[1].set_ylabel("rate")

    for i in range(tel.u.shape[1]):
        axes[2].step(t, tel.u[:, i], where="post", label=input_labels[i])  # ZOH → step plot
    axes[2].set_ylabel("input")

    axes[3].plot(t, tel.energy_J, label="E")
    axes[3].set_ylabel("energy [J]")
    axes[3].set_xlabel("t [s]")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
