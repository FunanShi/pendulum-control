"""pygame-in-container spike — not pytest-collected. Self-driving (no human).

Real display:  docker compose run --rm dev python tests/manual/manual_pygame_check.py
Headless logic: docker compose run --rm -e SDL_VIDEODRIVER=dummy dev python tests/manual/manual_pygame_check.py
Prints measured fps over 300 frames and echoes synthetic mouse/key events
(pygame.event.post) to prove the event-queue path. Exit 0 = pass. SDL shm
errors on a real display → uncomment `ipc: host` in compose.yaml.
"""
import os, sys, time

import pygame

pygame.init()
screen = pygame.display.set_mode((640, 360))
pygame.display.set_caption("dpend pygame spike")
clock = pygame.time.Clock()
seen = set()
t0 = time.perf_counter()
for frame in range(300):
    if frame == 50:
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(320, 180), button=1))
    if frame == 60:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
    for ev in pygame.event.get():
        if ev.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN):
            seen.add(ev.type)
            print("event ok:", ev)
    x = 40 + (frame * 3) % 560
    screen.fill((20, 20, 30))
    pygame.draw.rect(screen, (200, 120, 40), (x, 160, 60, 40))
    pygame.display.flip()
    clock.tick(60)  # target 60; measured value printed below
fps = 300 / (time.perf_counter() - t0)
print(f"measured fps over 300 frames: {fps:.1f}")
print(f"driver: {pygame.display.get_driver()}")
pygame.quit()
sys.exit(0 if seen == {pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN} else 1)
