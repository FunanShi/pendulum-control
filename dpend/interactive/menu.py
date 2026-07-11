"""Launcher menu: pick a Plant, a Controller, and a start-state, then
"Start ▶" — the pygame screen `Shell` shows before a session begins.

`compatible_controllers` (shared with `app.py`) discovers which controller
keys are buildable for a plant by trying each factory with empty params and
watching whether it raises — a factory's own gating (controllability,
`EnergyShapingCapable`), never hard-coded per plant name, so a future
plant/controller combination needs no edit here.

DAG: pygame + `dpend.registry` + `dpend.model.plant` — nothing else internal.
"""
from __future__ import annotations

import pygame

from dpend.interactive.widgets import Button, ButtonGroup
from dpend.model.plant import PLANTS
from dpend.registry import CONTROLLERS

# Own palette, independent per file (see widgets.py's palette note).
_BG = (18, 18, 24)
_TITLE_COLOR = (235, 235, 235)
_SUBTITLE_COLOR = (170, 170, 180)

# Probe order for `compatible_controllers` — fixed so the Controller column
# lists options in a stable order. Excludes "pole_placement" (still a stub)
# and "swingup" (on a swing-up-capable plant, app._ui_controller wraps
# lqr/mpc themselves into a swing-up ModeSwitch, so it is not a separate UI
# option).
_CONTROLLER_PROBE_ORDER = ["zero", "lqr", "mpc"]

# Human-readable UI labels; button values stay the registry keys the
# CLI/config/Shell use — only the on-screen text changes.
_PLANT_LABELS = {
    "cart": "Cart 2 Pendulum",      # double-pendulum-on-cart (ℝ⁶)
    "cartpole": "Cart 1 Pendulum",  # single-pole cart-pole (ℝ⁴)
    "fixed": "2 Pendulum",          # fixed-pivot double pendulum (ℝ⁴)
}


def plant_label(key: str) -> str:
    return _PLANT_LABELS.get(key, key)


def controller_label(key: str) -> str:
    """'zero' → 'None' (no controller = manual drag); others use the key
    as-is. Shared by the menu and the in-sim strip (app.py)."""
    return "None" if key == "zero" else key


# Column layout constants, all in px — positions derive from `cfg.window_px`
# at construction, never an assumed window size.
_MARGIN_PX = 30
_BANNER_H_PX = 60
_GROUP_TITLE_H_PX = 26
_BUTTON_H_PX = 34
_BUTTON_GAP_PX = 8
_START_GAP_PX = 26
_START_BUTTON_H_PX = 64
_N_COLUMNS = 3
# Rows the fixed-size columns reserve: Controller holds at most
# len(_CONTROLLER_PROBE_ORDER) buttons, Start-state holds 2. The Plant column
# holds len(plants), unknown until construction, so the Start▶ button's y
# takes max(_MAX_ROWS, len(plants)) at __init__ — the go-button sits below
# every column regardless of how many plants are registered.
_MAX_ROWS = max(len(_CONTROLLER_PROBE_ORDER), 2)


def compatible_controllers(plant) -> list[str]:
    """Controller keys buildable for `plant` with empty params (i.e.
    `CONTROLLERS[key](plant, {})` doesn't raise), in `_CONTROLLER_PROBE_ORDER`.
    Cheap (a CARE/DARE solve on <=6x6 at worst) but not free — hot-path
    callers cache the result per plant name, not per frame/click.
    """
    compatible = []
    for key in _CONTROLLER_PROBE_ORDER:
        try:
            CONTROLLERS[key](plant, {})
        except Exception:
            continue
        compatible.append(key)
    return compatible


class MenuScreen:
    """Three `ButtonGroup` radios — Plant, Controller, Start (the
    start-state, `["upright", "hanging"]`) — plus one separate action button,
    `_start_button` (the "Start ▶" go-button). Naming note: the Start group's
    `selected_value` is the start-state string; `_start_button` is unrelated
    and its click makes `handle_click` return the literal `"start"`.

    `selection` is read live off the three groups every call, never
    snapshotted.
    """

    def __init__(self, cfg, plants: list[str], start_plant: str = "cart"):
        if start_plant not in plants:
            raise ValueError(
                f"MenuScreen: start_plant={start_plant!r} not in plants={list(plants)!r}"
            )

        self._cfg = cfg
        self._plants = list(plants)
        self._compat_cache: dict[str, list[str]] = {}

        win_w, win_h = cfg.window_px
        col_w = (win_w - 2 * _MARGIN_PX) // _N_COLUMNS
        self._button_w = col_w - _MARGIN_PX
        self._col_top_y = _BANNER_H_PX + _MARGIN_PX + _GROUP_TITLE_H_PX

        plant_x = _MARGIN_PX
        controller_x = _MARGIN_PX + col_w
        start_state_x = _MARGIN_PX + 2 * col_w
        self._controller_x = controller_x

        # --- Plant group ---
        plant_buttons = [
            Button(plant_label(name), self._row_rect(plant_x, i), value=name)
            for i, name in enumerate(self._plants)
        ]
        self._plant_group = ButtonGroup("Plant", plant_buttons, selected_value=start_plant)
        self._plant_origin = (plant_x, _BANNER_H_PX + _MARGIN_PX)

        # --- Controller group (depends on the selected plant) ---
        compatible0 = self._compatible_for(start_plant)
        controller_buttons = self._build_controller_buttons(compatible0)
        self._controller_group = ButtonGroup(
            "Controller", controller_buttons, selected_value=compatible0[0]
        )
        self._controller_origin = (controller_x, _BANNER_H_PX + _MARGIN_PX)

        # --- Start-state group ---
        start_labels = ["upright", "hanging"]
        start_buttons = [
            Button(label, self._row_rect(start_state_x, i), value=label)
            for i, label in enumerate(start_labels)
        ]
        self._start_group = ButtonGroup("Start", start_buttons, selected_value="upright")
        self._start_origin = (start_state_x, _BANNER_H_PX + _MARGIN_PX)

        # --- Start▶ go-button: full-width, below every column's reserved rows ---
        go_rows = max(_MAX_ROWS, len(self._plants))
        go_y = self._col_top_y + go_rows * (_BUTTON_H_PX + _BUTTON_GAP_PX) + _START_GAP_PX
        go_w = win_w - 2 * _MARGIN_PX
        self._start_button = Button(
            "Start ▶", (_MARGIN_PX, go_y, go_w, _START_BUTTON_H_PX), value="start"
        )

    # -- layout helper --

    def _row_rect(self, x: int, row: int) -> tuple[int, int, int, int]:
        y = self._col_top_y + row * (_BUTTON_H_PX + _BUTTON_GAP_PX)
        return (x, y, self._button_w, _BUTTON_H_PX)

    # -- compatibility cache --

    def _compatible_for(self, plant_name: str) -> list[str]:
        """`compatible_controllers`, memoized per plant name — the factory
        probe (a CARE/DARE solve) runs once per distinct plant selection,
        not per click."""
        if plant_name not in self._compat_cache:
            self._compat_cache[plant_name] = compatible_controllers(PLANTS[plant_name]())
        return self._compat_cache[plant_name]

    def _build_controller_buttons(self, compatible: list[str]) -> list[Button]:
        return [
            Button(controller_label(key), self._row_rect(self._controller_x, i), value=key)
            for i, key in enumerate(compatible)
        ]

    def _rebuild_controller_group(self) -> None:
        """After a Plant-group click: rebuild the Controller column from the
        new plant's compatible list. If the previous selection isn't in it,
        fall back to the first entry (always "zero" — `ZeroController` never
        gates on plant structure)."""
        plant_name = self._plant_group.selected_value
        compatible = self._compatible_for(plant_name)
        previous = self._controller_group.selected_value
        selected = previous if previous in compatible else compatible[0]
        self._controller_group = ButtonGroup(
            "Controller", self._build_controller_buttons(compatible), selected_value=selected
        )

    # -- public interface --

    @property
    def selection(self) -> dict:
        return {
            "plant": self._plant_group.selected_value,
            "controller": self._controller_group.selected_value,
            "start": self._start_group.selected_value,
        }

    def handle_click(self, pos) -> str | None:
        """Forward `pos` to all three radio groups (a miss is a no-op),
        rebuilding the Controller column if the Plant selection changed, then
        check the separate `_start_button`. Returns `"start"` only for that
        go-button; `None` otherwise (radio clicks are silent selection
        changes, not a mode transition).
        """
        if self._plant_group.click(pos):
            self._rebuild_controller_group()
        self._controller_group.click(pos)
        self._start_group.click(pos)

        if self._start_button.hit(pos):
            return "start"
        return None

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        """Background, title banner, the three radio groups, and the Start▶
        button (drawn `selected=True` — the primary action always renders in
        the accent color)."""
        surface.fill(_BG)

        title_surf = font.render("dpend — choose plant, controller, start", True, _TITLE_COLOR)
        surface.blit(title_surf, (_MARGIN_PX, 12))
        subtitle_surf = font.render(
            "click a plant / a controller / a start-state, then Start ▶",
            True, _SUBTITLE_COLOR,
        )
        surface.blit(subtitle_surf, (_MARGIN_PX, 12 + title_surf.get_height() + 4))

        self._plant_group.draw(surface, font, self._plant_origin)
        self._controller_group.draw(surface, font, self._controller_origin)
        self._start_group.draw(surface, font, self._start_origin)
        self._start_button.draw(surface, font, selected=True, enabled=True)
