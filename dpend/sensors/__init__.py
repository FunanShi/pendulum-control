"""sensors — measurement models in the signal path.

A Sensor maps the true state to a measurement ``y = h(x) + noise`` that the
estimator consumes. Swappable like controllers. Depends on ``model`` and
``util`` only — never ``sim``.
"""
