# scenarios

Each file here is a small module that exposes a single ``scenario`` object
(a `dpend.config.Scenario`). Run one with:

```bash
docker compose run --rm dev python batch.py scenarios/<name>.py
```

A scenario picks the plant actuation config, the controller / estimator / sensor
(by name), the initial condition, horizon, and rates. It is pure configuration —
no logic — so experiments are reproducible and diff-able.
