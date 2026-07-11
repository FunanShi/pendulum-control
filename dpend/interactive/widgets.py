"""Reusable pygame button primitives: `Button` (one clickable rect) and
`ButtonGroup` (a radio: 0 or 1 of its buttons selected at a time).

Pure geometry + draw, no application state — `menu.py` decides where buttons
go and what their values mean. Button geometry is authoritative and
absolute: `rect` is set once at construction and never recomputed by
`draw()`, so `hit()`/`click()` work even if `draw()` is never called.

DAG: imports pygame only — no other dpend module.
"""
from __future__ import annotations

import pygame

# Own palette, matching render.py's aesthetic without importing its
# underscore-privates (menu.py duplicates likewise, same reason).
_BG = (18, 18, 24)
_BUTTON_COLOR = (45, 45, 55)
_BUTTON_SELECTED_COLOR = (95, 165, 230)
_BUTTON_DISABLED_COLOR = (30, 30, 34)
_BORDER_COLOR = (90, 90, 100)
_TEXT_COLOR = (235, 235, 235)
_TEXT_DISABLED_COLOR = (110, 110, 116)
_TITLE_COLOR = (235, 235, 235)

_BORDER_PX = 1


class Button:
    """One clickable rect: a `label`, a hit-box (`rect`), and an opaque
    `value` the caller reads back out (a controller key, a plant name, the
    literal string `"start"`, …). Standalone-vs-radio grouping is the
    caller's concern, not this class's.
    """

    def __init__(self, label: str, rect, value=None):
        self.label = label
        self.rect = pygame.Rect(rect)  # (x, y, w, h) px — absolute, screen space
        self.value = value

    def hit(self, pos) -> bool:
        """True iff screen pixel `pos` (x, y) falls inside this button's rect."""
        return self.rect.collidepoint(pos)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, *,
              selected: bool = False, enabled: bool = True) -> None:
        """Fill the rect (selected/enabled palette), border it, and blit
        `label` centered. `font` is caller-supplied — a pygame Font does not
        survive a quit() -> init() cycle, so it is never cached/created here
        (see render._font)."""
        if not enabled:
            fill, text_color = _BUTTON_DISABLED_COLOR, _TEXT_DISABLED_COLOR
        elif selected:
            fill, text_color = _BUTTON_SELECTED_COLOR, _BG  # dark-on-bright reads best
        else:
            fill, text_color = _BUTTON_COLOR, _TEXT_COLOR

        pygame.draw.rect(surface, fill, self.rect)
        pygame.draw.rect(surface, _BORDER_COLOR, self.rect, _BORDER_PX)

        label_surf = font.render(self.label, True, text_color)
        surface.blit(label_surf, label_surf.get_rect(center=self.rect.center))


class ButtonGroup:
    """Radio group over a fixed list of `Button`s: at most one selected at a
    time, tracked by `selected_value` — a `Button.value`, not a `Button`
    instance, so a caller can rebuild `buttons` wholesale while the selection
    carries over as plain data (re-validated by the caller, never a stale
    object reference).
    """

    def __init__(self, title: str, buttons: list[Button], selected_value=None):
        self.title = title
        self.buttons = buttons
        self.selected_value = selected_value

    def click(self, pos) -> bool:
        """First button hit (in list order) becomes selected; returns whether
        anything was hit. A miss changes nothing and returns False, so a
        caller can probe every group with the same click."""
        for b in self.buttons:
            if b.hit(pos):
                self.selected_value = b.value
                return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, origin) -> None:
        """Render the group's title at `origin`, then each button at its own
        absolute `rect` with `selected=(value == selected_value)`.
        Deliberately never repositions buttons from `origin` (it anchors the
        title text only) — geometry is set once by whoever constructs the
        buttons, so exactly one place computes positions."""
        x, y = origin
        title_surf = font.render(self.title, True, _TITLE_COLOR)
        surface.blit(title_surf, (x, y))
        for b in self.buttons:
            b.draw(surface, font, selected=(b.value == self.selected_value), enabled=True)
