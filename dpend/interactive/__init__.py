"""interactive — the live driver: a human at mouse/keyboard over the exact
same physics/control core as a scripted batch run. Live and batch share one
control-tick implementation (`sim.ticker.ControlTicker`), so they cannot
silently drift apart.

DAG role: a driver/leaf consumer like `sim` — nothing in `dpend` imports it.
No pygame import in this file, `ui_config.py`, `input.py`, or `loop.py`
(headless-testable); pygame is confined to `render.py`/`app.py`.
"""
