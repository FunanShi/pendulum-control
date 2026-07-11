"""controllers — swappable control laws.

Each law lives in its own file and implements ``base.Controller``. Adding a
controller is a new file, not surgery. Controllers depend on ``model`` (for
linearization / equilibria) and ``util`` only — never ``sim`` — so the same law
runs unchanged on hardware.
"""
