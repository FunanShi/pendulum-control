"""Cart end-to-end: headless batch.py artifacts + energy conservation,
end-stop pipeline through simulate(), work-energy audit of the disturbance hook."""
from __future__ import annotations

import numpy as np


def test_cart_drop_headless_produces_artifacts(tmp_path):
    """Headless batch.py on cart_drop_uncontrolled: artifacts written, 2000
    rows (10 s at 200 Hz), energy flat to <1e-6 relative.
    1e-6 basis: RK4 drift measured 5.7e-8 over 10 s (test_integrators.py) —
    ~17x headroom over honest drift, while a term/sign error overshoots by
    integer factors."""
    import batch as run_mod

    out = tmp_path / "cart_drop"
    run_mod.main(["scenarios/cart_drop_uncontrolled.py", "--headless", "--out", str(out)])

    npz = out / "telemetry.npz"
    png = out / "dashboard.png"
    assert npz.exists() and png.exists() and png.stat().st_size > 5_000

    from dpend.telemetry.formats import load_npz

    tel = load_npz(npz)
    assert tel.x_true.shape[0] == 2000  # 10 s at 200 Hz (ctrl_dt_s = 5e-3)

    E = tel.energy_J
    E_rel_drift = np.max(np.abs(E - E[0])) / abs(E[0])
    print(f"\nCart drop: energy relative drift = {E_rel_drift:.3e} (budget: 1e-6)")
    assert E_rel_drift < 1e-6, f"energy drift too large: {E_rel_drift}"


def test_cart_end_stop_pipeline_simulate():
    """End-stop pipeline via simulate() from x0=(1.2, 0, 0, 2.0, 0, 0): three
    impacts, max|x| < 1.75 m (empirical bound for this z0 — measured 1.7327 m,
    rationale in test_cart_oracles.py), energy monotone non-increasing within
    each contact episode, then flat (<1e-8 rel) on the final in-rail second."""
    from dpend.model.cart_params import CartParams
    from dpend.sim.simulator import simulate
    from dpend.model.plant import cart_plant
    from dpend.registry import SENSORS, ESTIMATORS, CONTROLLERS

    cp = CartParams()
    plant = cart_plant(cp)

    sensor = SENSORS["perfect"](plant, {})
    estimator = ESTIMATORS["identity"](plant, {})
    controller = CONTROLLERS["zero"](plant, {})

    x0 = np.array([1.2, 0.0, 0.0, 2.0, 0.0, 0.0])  # [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]

    tel = simulate(
        plant=plant, x0=x0, duration_s=5.0, sim_dt_s=1e-3, ctrl_dt_s=5e-3,
        sensor=sensor, estimator=estimator, controller=controller, seed=0
    )

    # Assertion 1: max|x| < empirical bound (1.75 m for this z0)
    x_traj = tel.x_true[:, 0]
    x_max = np.max(np.abs(x_traj))
    print(f"\nEnd-stop: max(|x|) = {x_max:.4f} m (limit: 1.75 m)")
    assert x_max < 1.75, f"overshoot exceeded: {x_max}"

    # Assertion 2: energy monotone non-increasing within each contact episode
    E_traj = tel.energy_J
    idx = np.where(np.abs(x_traj) > cp.L_rail)[0]
    if len(idx) > 0:
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        worst_intra_dE = -np.inf
        for run in runs:
            if len(run) > 1:
                dE = np.diff(E_traj[run])
                worst_intra_dE = max(worst_intra_dE, float(np.max(dE)))
                assert np.all(dE < 1e-10), \
                    "energy not monotone non-increasing within contact episode"
        print(f"End-stop: contact episodes: {len(runs)}; worst intra-episode dE: {worst_intra_dE:.3e}")

    # Assertion 3: flat energy on final in-rail second (t ∈ [4, 5] s)
    final_indices = np.where(tel.t_ns >= 4.0 * 1e9)[0]
    if len(final_indices) > 0:
        E_final = E_traj[final_indices]
        drift_final = np.max(np.abs(E_final - E_final[0])) / max(abs(E_final[0]), 1.0)
        print(f"End-stop: final-second energy drift (t ∈ [4, 5]): {drift_final:.3e}")
        assert drift_final < 1e-8


def test_work_energy_audit_cart_disturbance():
    """3 N cart disturbance for t<1 s, validated against the work-energy
    theorem dE = ∫ F·ẋ dt, not just a sign check. disturbance_fn returns the
    full ℝ³ tau_ext [F, 0, 0] (a shape-(1,) return would numpy-broadcast
    across all three coordinates — ControlTicker.tick's shape guard rejects
    that). Defaults are frictionless and the cart stays in-rail (asserted),
    so the disturbance's work is the only energy change."""
    import numpy as np
    from dpend.sim.simulator import simulate
    from dpend.model.plant import cart_plant
    from dpend.registry import SENSORS, ESTIMATORS, CONTROLLERS

    plant = cart_plant()  # frictionless defaults: bc=0, pend b1=b2=0

    sensor = SENSORS["perfect"](plant, {})
    estimator = ESTIMATORS["identity"](plant, {})
    controller = CONTROLLERS["zero"](plant, {})

    def disturbance_fn(t, x):
        """3 N step force on the cart (τ_ext ∈ ℝ³ = [F_x, τ_θ1, τ_θ2]) for
        t < 1.0 s, then zero."""
        if t < 1.0:
            return np.array([3.0, 0.0, 0.0])
        return np.zeros(3)

    x0 = np.zeros(6)
    ctrl_dt_s = 5e-3

    tel = simulate(
        plant=plant, x0=x0, duration_s=2.0, sim_dt_s=1e-3, ctrl_dt_s=ctrl_dt_s,
        sensor=sensor, estimator=estimator, controller=controller,
        seed=0, disturbance=disturbance_fn
    )

    # Sanity: the cart must stay in-rail throughout — end-stop contact would
    # inject c_stop damping and break the clean work-energy identity below.
    x_max = np.max(np.abs(tel.x_true[:, 0]))
    assert x_max < 1.5, f"cart left the rail (contact would add damping): {x_max}"

    # Work-energy theorem: dE == ∫ F·ẋ dt (trapezoid over the ctrl_dt_s grid);
    # ẋ is column 3 of z = [x, θ1, θ2, ẋ, θ̇1, θ̇2]. np.trapezoid: NumPy 2.x
    # removed the np.trapz alias — same quadrature.
    W = np.trapezoid(tel.tau_ext[:, 0] * tel.x_true[:, 3], dx=ctrl_dt_s)
    dE = tel.energy_J[-1] - tel.energy_J[0]
    rel_err = abs(dE - W) / abs(W)

    print(f"\nWork-energy audit: dE = {dE:.6f} J, W = {W:.6f} J, "
          f"rel_err = {rel_err:.4e}")

    # Force must have done real work (not a near-zero/wrong-channel no-op).
    assert W > 1.0, f"disturbance did too little work: {W} J"
    # 2% tolerance: the t=1.0 s force step breaks trapezoid smoothness,
    # missing O(ctrl_dt_s) of area at the jump — generous against that
    # discretization error, tight against channel/sign bugs (integer-factor errors).
    assert rel_err < 0.02, f"work-energy mismatch: dE={dE}, W={W}, rel_err={rel_err}"
