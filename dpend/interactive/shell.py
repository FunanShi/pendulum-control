"""`Shell` — owns the pygame window + main loop, switching between the
launcher `MenuScreen` and a running `App` (the sim). The only place pygame is
init()'d / quit() for a live session (App and MenuScreen both draw into a
Surface Shell hands them).

States: "MENU" (launcher; a Start▶ click builds an App → SIM) and "SIM" (a
running App; its in-sim Menu button sets app.want_menu, returning here to
MENU after finalizing that sim's telemetry). `start_selection` (a dict
{"plant","controller","start"[,"params"]}) skips the menu and boots SIM
directly.
"""
from __future__ import annotations

import time

import pygame

from dpend.interactive.app import App
from dpend.interactive.menu import MenuScreen
from dpend.interactive.ui_config import InteractiveConfig
from dpend.model.plant import PLANTS

_MENU, _SIM = "MENU", "SIM"


class Shell:
    def __init__(self, cfg: InteractiveConfig, plants, *, start_selection=None,
                 out_dir=None, now_fn=time.perf_counter):
        self._cfg = cfg
        self._now_fn = now_fn
        self._out_dir = out_dir   # telemetry sink for app.close() on menu-return / quit;
                                  # None → App's default artifacts/ path (a test seam)
        pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        self.screen = pygame.display.set_mode(cfg.window_px)
        pygame.display.set_caption("dpend")
        self._font = pygame.font.Font(None, 24)   # not cached across init/quit (see render._font's note)
        self._clock = pygame.time.Clock()
        self._menu = MenuScreen(cfg, list(plants))
        self.app = None
        self._done = False
        if start_selection is not None:
            self._enter_sim(start_selection)
        else:
            self.state = _MENU

    def _enter_sim(self, sel: dict) -> None:
        plant = PLANTS[sel["plant"]]()
        self.app = App(plant, self._cfg, self.screen,
                       controller_name=sel["controller"], start=sel["start"],
                       scenario_params=sel.get("params"), now_fn=self._now_fn)
        self.state = _SIM

    def step_once(self, synthetic_events=None) -> None:
        events = pygame.event.get() if synthetic_events is None else synthetic_events
        if self.state == _MENU:
            for ev in events:
                if ev.type == pygame.QUIT or (
                        ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    self._done = True
                    return
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self._menu.handle_click(ev.pos) == "start":
                        self._enter_sim(self._menu.selection)
            if self.state == _MENU:   # didn't just enter SIM this frame
                self._menu.draw(self.screen, self._font)
                pygame.display.flip()
                self._clock.tick(self._cfg.fps)
        else:  # _SIM
            self.app.step_once(events)
            if self.app.want_menu:
                self.app.close(self._out_dir)
                self.app = None
                self.state = _MENU
            elif not self.app.running:
                self.app.close(self._out_dir)
                self.app = None
                self._done = True

    def run(self) -> None:
        try:
            while not self._done:
                self.step_once()
        finally:
            pygame.quit()
