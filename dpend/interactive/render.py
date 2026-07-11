"""Generic pygame renderer: world -> screen transform + one-frame scene draw.

Draws any `Plant`: geometry comes only from `plant.fk(z)` (ordered
joint-chain points, plus an optional cart pose) and the plant's own static
bounds (`plant.reach`, `plant.rail`) — never a hard-coded viewport or a
branch on `plant.name`. The live analog of `viz/animation.py`'s generic
replay renderer.

`draw_scene` is a stateless one-frame call taking only the current `z`; the
tip trace's history lives in `app.py` and rides in as `hud["tip_trace"]`.

DAG: model (`Plant` protocol fields only) + pygame; this and `app.py` are
the only pygame importers in `dpend/`.
"""
from __future__ import annotations

import pygame

# Visual constants (drawing-only, not physical parameters: the on-screen cart
# footprint is chosen for legibility, independent of cart_params).
_CART_W_M = 0.30   # visual cart footprint width [m]
_CART_H_M = 0.15   # visual cart footprint height [m]
_END_ZONE_FRAC = 0.08  # visual-only: outer fraction of each rail half flagged
                       # as the end-stop zone (the physics stop engages exactly
                       # at |x| > L_rail; this is a cue, not a second boundary)

_BG = (18, 18, 24)
_RAIL_COLOR = (95, 95, 105)
_END_ZONE_COLOR = (120, 65, 65)
_CART_COLOR = (210, 165, 60)
_LINK_COLOR = (230, 230, 235)
_JOINT_COLOR = (95, 165, 230)
_TRACE_COLOR = (105, 105, 135)  # tip-trace ribbon: dimmer than links, so the
                                 # current chain always reads on top of history
_REF_COLOR = (60, 200, 120)
_HUD_COLOR = (235, 235, 235)
_FORCE_BAR_BG = (60, 60, 70)
_FORCE_BAR_FG = (220, 95, 95)
_NOTICE_COLOR = (255, 90, 60)  # alert red, distinct from the HUD's neutral
                               # grey — a supervisor disengage reads as urgent


class WorldToScreen:
    """World frame (x right, y up, meters; origin = the plant's own origin —
    rail center for the cart, pivot for the fixed plant) -> screen pixels
    (origin top-left, y down — the y-flip every 2-D renderer needs).

    Scale derives from the plant's static geometry bounds, never a
    hard-coded constant. Span per axis = 2.2 × bound (2 × a 1.1 margin —
    the same 10% margin viz/animation.py uses): horizontal bound =
    max(plant.reach, plant.rail or 0), vertical bound = reach (the chain can
    extend its full reach above or below; no rail-like bound applies
    vertically). One shared `scale` serves both axes (uniform aspect:
    circles stay circles), taken from whichever axis is tighter, so nothing
    clips at any window aspect ratio or plant.
    """

    def __init__(self, plant, window_px: tuple[int, int]):
        self.window_px = (int(window_px[0]), int(window_px[1]))
        reach = float(plant.reach)
        rail_or_0 = float(plant.rail or 0.0)
        horiz_m = 2.2 * max(reach, rail_or_0)
        vert_m = 2.2 * reach
        if horiz_m <= 0.0:
            horiz_m = 2.2  # pathological guard (zero-reach, no-rail plant):
        if vert_m <= 0.0:  # an arbitrary finite scale beats zero/inf
            vert_m = 2.2
        scale_from_width = self.window_px[0] / horiz_m
        scale_from_height = self.window_px[1] / vert_m
        self.scale = min(scale_from_width, scale_from_height)  # tighter axis wins: never clips
        self.cx = self.window_px[0] / 2.0
        self.cy = self.window_px[1] / 2.0

    def to_px(self, xy) -> tuple[float, float]:
        """World (x, y) [m] -> screen (px_x, px_y), y flipped, origin at the
        window's center pixel."""
        x, y = float(xy[0]), float(xy[1])
        return (self.cx + x * self.scale, self.cy - y * self.scale)

    def to_world(self, px) -> tuple[float, float]:
        """Screen (px_x, px_y) -> world (x, y) [m] — the exact inverse of
        `to_px`. Turns a mouse pixel into a world position (drag target,
        right-click reference target)."""
        px_x, px_y = float(px[0]), float(px[1])
        return ((px_x - self.cx) / self.scale, (self.cy - px_y) / self.scale)


def _cart_rect_from_xy(cart_xy, w2s: WorldToScreen) -> pygame.Rect:
    cx, cy = w2s.to_px(cart_xy)
    w_px = max(1.0, _CART_W_M * w2s.scale)
    h_px = max(1.0, _CART_H_M * w2s.scale)
    return pygame.Rect(cx - w_px / 2.0, cy - h_px / 2.0, w_px, h_px)


def cart_rect_px(plant, z, w2s: WorldToScreen):
    """pygame.Rect (screen px) around the cart's projected position, or
    `None` if this plant has no cart pose (`plant.fk(z)[0] is None`). The
    same geometry `draw_scene` draws the cart at, so the drag-grab hit test
    and the on-screen cart can never disagree."""
    cart_xy, _ = plant.fk(z)
    if cart_xy is None:
        return None
    return _cart_rect_from_xy(cart_xy, w2s)


def _draw_rail(screen, rail_m: float, w2s: WorldToScreen) -> None:
    """Rail line + end zones: a visual cue for the soft end-stop (physics:
    exactly `|x| > rail_m`) — a highlighted outer fraction of each rail half
    plus an end-cap tick at the true limit.
    """
    y0 = w2s.to_px((0.0, 0.0))[1]
    x_left = w2s.to_px((-rail_m, 0.0))[0]
    x_right = w2s.to_px((rail_m, 0.0))[0]
    pygame.draw.line(screen, _RAIL_COLOR, (x_left, y0), (x_right, y0), 4)

    zone_m = _END_ZONE_FRAC * rail_m
    for sign in (-1.0, 1.0):
        outer = w2s.to_px((sign * rail_m, 0.0))[0]
        inner = w2s.to_px((sign * (rail_m - zone_m), 0.0))[0]
        lo, hi = sorted((outer, inner))
        pygame.draw.line(screen, _END_ZONE_COLOR, (lo, y0), (hi, y0), 8)
        pygame.draw.line(screen, _RAIL_COLOR, (outer, y0 - 10), (outer, y0 + 10), 3)


def _draw_reference_marker(screen, target_m: float, w2s: WorldToScreen) -> None:
    """Downward-pointing triangle above the target world x on the rail line —
    the cart-position setpoint a tracking controller glides toward. When
    nothing reads the reference it is just this marker."""
    x, y = w2s.to_px((target_m, 0.0))
    size = 8
    pygame.draw.polygon(screen, _REF_COLOR,
                        [(x, y - size), (x - size, y + size), (x + size, y + size)])


def _font(size: int = 20) -> pygame.font.Font:
    """Pygame's bundled default TTF (no system font file needed; works under
    `SDL_VIDEODRIVER=dummy`). Deliberately not cached at module level: a
    `pygame.font.Font` does not survive a `pygame.quit()` -> `init()` cycle
    (stale SDL_ttf handle, wrapper never invalidated) — a cached instance
    reused after such a cycle segfaults; one cheap Font construction per
    frame is the price. `size` defaults to the HUD's 20 px; the notice
    banner asks for larger."""
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame.font.Font(None, size)


def _hud_lines(hud: dict) -> list[str]:
    """Pure text-line builder for the HUD block — no pygame Surface/Font
    needed, so the content is unit-testable without a display.

    Optional lines append only when the data is present: `V = .. V_lim = ..`
    (both `v`/`v_limit` not None — the RoA supervisor diagnostic) and
    `swing: ..` (`swing_mode` not None). `swing_mode` is deliberately a
    separate line from `mode`: `mode` is the UI's own MANUAL/CONTROLLER
    interaction state, `swing_mode` is the hybrid controller's
    energy-shaping-vs-catch state — two orthogonal state machines, so they
    get distinct labels rather than colliding on one line.
    """
    lines = [
        f"mode: {hud.get('mode', '?')}   controller: {hud.get('controller', '?')}",
        f"t = {float(hud.get('t', 0.0)):7.3f} s    E = {float(hud.get('energy', 0.0)):8.3f} J",
        f"fps = {float(hud.get('fps', 0.0)):5.1f}   dropped = {float(hud.get('dropped_s', 0.0)):6.3f} s",
    ]
    v, v_limit = hud.get("v"), hud.get("v_limit")
    if v is not None and v_limit is not None:
        lines.append(f"V = {float(v):8.3e}   V_lim = {float(v_limit):8.3e}")

    swing_mode = hud.get("swing_mode")
    if swing_mode is not None:
        lines.append(f"swing: {swing_mode}")

    return lines


def _draw_hud(screen, hud: dict) -> int:
    """HUD text lines (see `_hud_lines`) + a force bar (hand force / f_max,
    signed, clamped to the bar's width). Returns the y pixel immediately
    below the drawn block, so `draw_scene` can place the notice banner
    beneath it without duplicating this layout math."""
    lines = _hud_lines(hud)

    font = _font()
    y = 8
    for line in lines:
        surf = font.render(line, True, _HUD_COLOR)
        screen.blit(surf, (8, y))
        y += surf.get_height() + 2

    force = float(hud.get("force", 0.0))
    f_max = float(hud.get("f_max", 1.0)) or 1.0  # guard f_max=0 (would divide by zero)
    bar_x, bar_y, bar_w, bar_h = 8, y + 4, 200, 12
    pygame.draw.rect(screen, _FORCE_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
    frac = max(-1.0, min(1.0, force / f_max))
    half = bar_w / 2.0
    fill_w = abs(frac) * half
    fill_x = bar_x + half if frac >= 0 else bar_x + half - fill_w
    pygame.draw.rect(screen, _FORCE_BAR_FG, (fill_x, bar_y, fill_w, bar_h))
    pygame.draw.line(screen, _HUD_COLOR, (bar_x + half, bar_y), (bar_x + half, bar_y + bar_h), 1)
    return bar_y + bar_h


def _draw_notice(screen, notice: str, y: int) -> None:
    """Prominent one-line banner (e.g. the RoA supervisor's disengage
    message) below the HUD block — bigger font, alert color. Only called
    when `notice` is non-empty (`draw_scene`'s guard)."""
    surf = _font(26).render(notice, True, _NOTICE_COLOR)
    screen.blit(surf, (8, y + 6))


def draw_scene(screen, plant, z, hud: dict, w2s: WorldToScreen) -> None:
    """Draw one frame: background, rail + end zones (if `plant.rail` is not
    None), cart rect (if `plant.fk(z)` returns a cart pose), joint polyline
    + markers, reference marker (if `hud["target"]` is not None), HUD text.
    Generic over any Plant: every branch tests a protocol field, never
    `plant.name`.

    `hud` keys consumed (all optional; each defaults sanely if absent):
    `mode`, `controller` (str), `t` [s], `energy` [J], `fps` (measured),
    `dropped_s` [s], `force` [N] (signed hand force, for the bar), `f_max`
    [N] (bar's full-scale value), `target` (world x [m] or None), `tip_trace`
    (sequence of world (x, y) [m] tip positions, oldest first — a thin
    history polyline under the chain; absent/short = no trace), `notice`
    (str — banner below the HUD when non-empty), `v`/`v_limit` (RoA
    diagnostic line, only when both not None), `swing_mode` (hybrid
    controller's "SWINGING"/"CATCHING" — a `swing: ..` line when not None).
    """
    screen.fill(_BG)

    if plant.rail is not None:
        _draw_rail(screen, float(plant.rail), w2s)

    trace = hud.get("tip_trace")
    if trace is not None and len(trace) >= 2:
        pygame.draw.lines(screen, _TRACE_COLOR, False, [w2s.to_px(p) for p in trace], 1)

    cart_xy, pts = plant.fk(z)

    if cart_xy is not None:
        pygame.draw.rect(screen, _CART_COLOR, _cart_rect_from_xy(cart_xy, w2s))

    pts_px = [w2s.to_px(p) for p in pts]
    if len(pts_px) >= 2:
        pygame.draw.lines(screen, _LINK_COLOR, False, pts_px, 3)
    for p in pts_px:
        pygame.draw.circle(screen, _JOINT_COLOR, (round(p[0]), round(p[1])), 5)

    target = hud.get("target")
    if target is not None:
        _draw_reference_marker(screen, float(target), w2s)

    hud_bottom_px = _draw_hud(screen, hud)
    notice = hud.get("notice")
    if notice:
        _draw_notice(screen, notice, hud_bottom_px)
