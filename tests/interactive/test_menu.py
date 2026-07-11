import os; os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
from dpend.interactive.ui_config import InteractiveConfig
from dpend.interactive.menu import MenuScreen, compatible_controllers
from dpend.model.plant import PLANTS, cart_pole_plant, cart_plant

def test_compatible_controllers_is_zero_lqr_mpc_for_controllable_plants():
    # swingup is no longer a standalone UI option — it's subsumed into lqr/mpc on
    # swing-up-capable plants (app._ui_controller). The UI list is just the balance
    # laws, in probe order, for any controllable plant.
    for p in (cart_pole_plant(), cart_plant()):
        assert compatible_controllers(p) == ["zero", "lqr", "mpc"]
        assert "swingup" not in compatible_controllers(p)

def test_menu_defaults_valid_and_start_signals():
    pygame.init()
    m = MenuScreen(InteractiveConfig(), plants=sorted(PLANTS), start_plant="cart")
    assert m.selection["controller"] in compatible_controllers(PLANTS[m.selection["plant"]]())
    assert m.handle_click(m._start_button.rect.center) == "start"
    pygame.quit()

def test_plant_buttons_use_friendly_labels_but_keep_registry_key_values():
    pygame.init()
    m = MenuScreen(InteractiveConfig(), plants=sorted(PLANTS), start_plant="cart")
    labels = {b.value: b.label for b in m._plant_group.buttons}
    assert labels["cart"] == "Cart 2 Pendulum"
    assert labels["cartpole"] == "Cart 1 Pendulum"
    assert labels["fixed"] == "2 Pendulum"
    assert m.selection["plant"] == "cart"   # value (key) unchanged → plumbing still works
    pygame.quit()

def test_controller_button_labels_zero_as_none():
    from dpend.interactive.menu import controller_label
    assert controller_label("zero") == "None" and controller_label("lqr") == "lqr"
    pygame.init()
    m = MenuScreen(InteractiveConfig(), plants=sorted(PLANTS), start_plant="cartpole")
    labels = {b.value: b.label for b in m._controller_group.buttons}
    assert labels["zero"] == "None"
    assert m.selection["controller"] == "zero"   # value still the key
    pygame.quit()
