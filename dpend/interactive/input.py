"""Pure hand-force law: mouse-drag spring+damper, arrow-key push, one clamp.

No pygame here — takes already-extracted plain numbers and returns a scalar
force; `app.py` is the only place a pygame event becomes these arguments.
Depends on `interactive.ui_config` only.

Units: positions/velocity in the plant's own frame (cart: x [m], ẋ [m/s]);
force in N — the cart's actuation channel.
"""
from __future__ import annotations

import numpy as np

from dpend.interactive.ui_config import InteractiveConfig


def hand_force(*, dragging: bool, x_mouse: float, x_cart: float, xdot_cart: float,
               key_dir: int, cfg: InteractiveConfig) -> float:
    """Spring+damper toward the mouse while `dragging`, plus a constant push
    while a key is held — summed, then clamped once to `±cfg.f_max`
    (clamping each term separately would let a key press add force after the
    drag term alone had already saturated).

        F = k_drag·(x_mouse − x_cart) − c_drag·ẋ_cart   [only while dragging]
          + key_dir·f_key                                [only while a key held]
        return clip(F, −f_max, +f_max)

    Idle (`dragging=False` and `key_dir==0`) returns exactly 0.0 — asserted
    explicitly so the invariant survives future edits.
    """
    if not dragging and key_dir == 0:
        return 0.0

    # Spring and damper are both gated on `dragging` (not just the damper): a
    # stale x_mouse from a previous drag must not exert a phantom pull once
    # the mouse button is released.
    spring = cfg.k_drag * (x_mouse - x_cart) if dragging else 0.0
    damper = -cfg.c_drag * xdot_cart if dragging else 0.0
    key = key_dir * cfg.f_key  # independent of dragging: keys work in either mode

    total = spring + damper + key
    return float(np.clip(total, -cfg.f_max, cfg.f_max))
