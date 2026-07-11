"""Interactive UI entry point — live cart-pendulum controller.

Launch: ./run

Controls:
  • Drag the cart (left-click) → spring-damper hand force, clamped to ±f_max [N]
  • ←/→ arrow keys → constant force (±f_key [N]) while held; both held cancels to 0
  • R → reset to x₀ (the chosen --start, not always upright); state, tip trace,
    sim time t, and drift counter all reset
  • Right-click → set the reference target x_ref [m]
  • ESC / window close → exit, save session to artifacts/live_<YYYYMMDD-HHMMSS>/telemetry.npz

All pygame work happens inside `Shell.run()`, not on import (module loads
safely in headless/pytest contexts).
"""
from __future__ import annotations


def main(argv=None) -> None:
    """Parse CLI args and launch the interactive UI."""
    from dpend.interactive.cli import run_app
    run_app(argv)


if __name__ == "__main__":
    main()
