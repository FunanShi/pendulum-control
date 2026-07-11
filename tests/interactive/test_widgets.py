import os; os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
from dpend.interactive.widgets import Button, ButtonGroup

def test_button_hit_inside_and_outside():
    b = Button("LQR", (10,20,100,30), value="lqr")
    assert b.hit((15,25)) and not b.hit((200,25)) and b.value == "lqr"

def test_group_click_selects_radio():
    g = ButtonGroup("Controller", [Button("zero",(0,0,80,30),value="zero"),
                                   Button("lqr",(0,40,80,30),value="lqr")])
    assert g.click((5,45)) and g.selected_value == "lqr"
    assert g.click((5,5))  and g.selected_value == "zero"
    assert not g.click((500,500)) and g.selected_value == "zero"
