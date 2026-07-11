"""estimation — state estimators that close the loop on an estimate.

An Estimator turns measurements (and the last control) into a full-state
estimate x̂ the controller uses. The default is the identity (full-state
feedback); LQG uses the Kalman filter. Depends on ``model`` and ``util`` only —
never ``sim``.
"""
