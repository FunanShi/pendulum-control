"""sim — the simulation engine: *how we roll the plant forward in time*.

Owns the integrator and the multi-rate loop, wiring model, sensors, estimation,
controllers, and telemetry. Controllers/estimators/sensors never import it, so
the same control code runs unchanged on hardware.
"""
