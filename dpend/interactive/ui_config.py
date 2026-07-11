"""InteractiveConfig — UI-only tuning knobs for the live loop (mouse/key
feel, render rate, hiccup budget): human-interface constants, not properties
of the system being simulated.

Depends on nothing internal — a leaf dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InteractiveConfig:
    """Force/feel constants are hand-tuned at the demo and recorded here —
    not measured or derived quantities."""

    fps: int = 60                            # target render rate [Hz]
    window_px: tuple[int, int] = (960, 540)   # window size [px, px]
    k_drag: float = 60.0   # mouse-drag spring constant [N/m]
    c_drag: float = 12.0   # mouse-drag damping [N·s/m]
    f_key: float = 8.0     # constant force while an arrow key is held [N]
    f_max: float = 60.0    # clamp on the summed hand force (drag + key) [N];
                           # the scenario disturbance is scripted physics,
                           # not a UI input, and is not subject to this clamp
    max_substeps_per_frame: int = 100  # hiccup cap in sim_dt-rate integrator
                           # substeps per advance() call — ≈100 ms of physics
                           # per rendered frame at the standard 1 kHz sim_dt;
                           # the rest of a hiccup is dropped, not carried
                           # (see loop.py)
