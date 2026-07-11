"""Shared pytest session setup. SDL_VIDEODRIVER=dummy must be set before
`import pygame` anywhere in this process (SDL reads it at init); conftest.py
is imported before any test module, so this runs first unconditionally.
setdefault (not assignment) so an explicit override — e.g. a real-display
manual run — still wins. SDL_AUDIODRIVER=dummy is set defensively too:
compose.yaml already sets it, but the suite should be correct standalone."""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless pygame — see module docstring
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")   # no sound device; belt-and-suspenders
