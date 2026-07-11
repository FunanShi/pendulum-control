import os; os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
from dpend.interactive.ui_config import InteractiveConfig
from dpend.interactive.shell import Shell
from dpend.model.plant import PLANTS

def _shell(**kw):
    return Shell(InteractiveConfig(fps=2000), sorted(PLANTS), **kw)

def test_starts_on_menu_then_start_builds_sim():
    sh = _shell()
    try:
        assert sh.state == "MENU" and sh.app is None
        sh.step_once(synthetic_events=[pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=sh._menu._start_button.rect.center, button=1)])
        assert sh.state == "SIM" and sh.app is not None
    finally:
        pygame.quit()

def test_flags_skip_menu_and_boot_sim():
    sh = _shell(start_selection={"plant": "cartpole", "controller": "swingup", "start": "hanging"})
    try:
        assert sh.state == "SIM" and sh.app.controller_name == "swingup"
    finally:
        pygame.quit()

def test_menu_button_returns_to_menu(tmp_path):
    sh = _shell(start_selection={"plant": "cart", "controller": "lqr", "start": "upright"},
                out_dir=tmp_path / "live_menu_return")   # out_dir seam: keep the finalize out of real artifacts/
    try:
        sh.app.want_menu = True
        sh.step_once(synthetic_events=[])
        assert sh.state == "MENU" and sh.app is None
    finally:
        pygame.quit()
