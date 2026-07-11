"""telemetry — the recording contract between sim (producer) and viz (consumer).

A Telemetry record holds time-aligned arrays of every signal in the loop
(t_ns, x_true, x_hat, y, u, energy, …). The simulator writes it; viz and the
benchmark tooling read it; it persists to disk. Depends on ``util`` only.
"""
