"""Manual X11 input-path check — not collected by pytest (no test_ prefix).

Run from the host:  docker compose run --rm dev python tests/manual/manual_x_click_check.py
A window opens; click anywhere inside its axes. On click it prints CLICK OK and
closes itself; exit code 0 = mouse events cross the container boundary — the
load-bearing guarantee for the interactive cart-drag GUI (X11 is bidirectional:
the socket that carries draw requests out carries input events back in).
"""
import sys

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

clicks = []


def on_click(event):
    # event.xdata/ydata: click position in data coords (None if outside axes)
    clicks.append((event.xdata, event.ydata))
    print(f"CLICK OK at data coords ({event.xdata}, {event.ydata})")
    plt.close(event.canvas.figure)


fig, ax = plt.subplots()
ax.set_title("dpend input check: click anywhere in the axes")
ax.plot([0, 1], [0, 1])
fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()  # blocks until on_click closes the figure (or user closes without clicking)

sys.exit(0 if clicks else 1)
