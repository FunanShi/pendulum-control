"""Headless interactive core (no pygame): the hand_force law, the RealtimeLoop
accumulator on a scripted clock, and the live ≡ batch equivalence gate."""
from __future__ import annotations

import numpy as np
import pytest

from dpend.controllers.zero import ZeroController
from dpend.estimation.identity import IdentityEstimator
from dpend.interactive.ui_config import InteractiveConfig
from dpend.interactive.input import hand_force
from dpend.interactive.loop import RealtimeLoop
from dpend.model.plant import cart_plant, fixed_pivot_plant
from dpend.sensors.perfect import PerfectSensor
from dpend.sim.simulator import simulate
from dpend.sim.ticker import ControlTicker
from dpend.telemetry.recorder import Recorder

CFG = InteractiveConfig()  # defaults: k_drag=60, c_drag=12, f_key=8, f_max=60


# ---------------------------------------------------------------------------
# hand_force: pure force law
# ---------------------------------------------------------------------------

def test_hand_force_idle_returns_zero():
    """No drag, no key gives exactly 0 N even with a stale mouse offset and
    nonzero cart velocity lying around."""
    f = hand_force(dragging=False, x_mouse=5.0, x_cart=0.0, xdot_cart=3.0,
                    key_dir=0, cfg=CFG)
    assert f == 0.0


def test_hand_force_spring_pulls_toward_mouse():
    """Spring sign follows (x_mouse − x_cart); zero velocity and key isolate
    the spring term alone."""
    right = hand_force(dragging=True, x_mouse=0.2, x_cart=0.0, xdot_cart=0.0,
                        key_dir=0, cfg=CFG)
    left = hand_force(dragging=True, x_mouse=-0.2, x_cart=0.0, xdot_cart=0.0,
                       key_dir=0, cfg=CFG)
    assert right == pytest.approx(60.0 * 0.2)    # +12 N, toward the mouse
    assert left == pytest.approx(60.0 * -0.2)    # -12 N, toward the mouse


def test_hand_force_damping_opposes_velocity_only_while_dragging():
    """Damping −c_drag·ẋ applies only while dragging; the same velocity adds
    nothing when not dragging (checked via a key-press case, not idle's 0)."""
    dragging = hand_force(dragging=True, x_mouse=0.0, x_cart=0.0, xdot_cart=2.0,
                           key_dir=0, cfg=CFG)
    not_dragging = hand_force(dragging=False, x_mouse=0.0, x_cart=0.0, xdot_cart=5.0,
                               key_dir=1, cfg=CFG)
    assert dragging == pytest.approx(-12.0 * 2.0)   # -24 N: opposes rightward ẋ
    assert not_dragging == pytest.approx(8.0)       # key term only; ẋ=5 ignored


def test_hand_force_key_adds_signed_constant_force():
    """An arrow key adds ±f_key; with no drag and zero spring/damper inputs
    the key term is the entire result."""
    plus = hand_force(dragging=False, x_mouse=0.0, x_cart=0.0, xdot_cart=0.0,
                       key_dir=1, cfg=CFG)
    minus = hand_force(dragging=False, x_mouse=0.0, x_cart=0.0, xdot_cart=0.0,
                        key_dir=-1, cfg=CFG)
    assert plus == pytest.approx(8.0)
    assert minus == pytest.approx(-8.0)


def test_hand_force_stale_mouse_exerts_no_pull_when_not_dragging():
    """Spring is gated on dragging itself, not just the idle short-circuit: a
    stale x_mouse 3 m away plus a held key yields exactly +f_key."""
    stale = hand_force(dragging=False, x_mouse=3.0, x_cart=0.0, xdot_cart=0.0,
                        key_dir=1, cfg=CFG)
    assert stale == CFG.f_key


def test_hand_force_clamps_combined_total():
    """The clamp applies to the spring+key sum (128 N → ±f_max=60), not to
    each term separately."""
    high = hand_force(dragging=True, x_mouse=2.0, x_cart=0.0, xdot_cart=0.0,
                       key_dir=1, cfg=CFG)
    low = hand_force(dragging=True, x_mouse=-2.0, x_cart=0.0, xdot_cart=0.0,
                      key_dir=-1, cfg=CFG)
    assert high == pytest.approx(60.0)
    assert low == pytest.approx(-60.0)


# ---------------------------------------------------------------------------
# RealtimeLoop: wall-clock accumulator (scripted fake clock — no real timing)
# ---------------------------------------------------------------------------

class _ScriptedClock:
    """now_fn stand-in: pops scripted absolute wall-clock timestamps [s], one
    per call — exact elapsed times without a real clock."""

    def __init__(self, times):
        self._times = list(times)

    def __call__(self) -> float:
        return self._times.pop(0)


def _make_ticker(*, ctrl_dt_s=5e-3, sim_dt_s=1e-3):
    """Minimal fixed-pivot ControlTicker for accumulator-arithmetic tests;
    physics content is irrelevant (ZeroController, zero tau_ext)."""
    plant = fixed_pivot_plant()
    return ControlTicker(plant=plant, sensor=PerfectSensor(),
                          estimator=IdentityEstimator(plant.n),
                          controller=ZeroController(m=plant.m),
                          ctrl_dt_s=ctrl_dt_s, sim_dt_s=sim_dt_s)


def _zero_tau(t, x):
    return np.zeros(2)  # fixed-pivot plant.n // 2


def test_advance_consumes_whole_ticks_and_carries_remainder():
    """12.5 ms at ctrl_dt=5 ms yields exactly 2 ticks with the 2.5 ms remainder
    carried: feeding 2.5 ms more produces exactly one more tick."""
    ticker = _make_ticker()
    clock = _ScriptedClock([0.0, 0.0125, 0.015])  # construct@0; +12.5ms; +2.5ms
    loop = RealtimeLoop(ticker=ticker, x0=np.zeros(4), ctrl_dt_s=5e-3, now_fn=clock)

    first = loop.advance(_zero_tau)
    second = loop.advance(_zero_tau)

    assert len(first) == 2     # floor(12.5 / 5) = 2
    assert len(second) == 1    # carried 2.5 ms + new 2.5 ms = 5.0 ms = 1 more tick
    assert loop.dropped_s == 0.0


def test_advance_caps_ticks_on_hiccup_and_tracks_dropped_time():
    """A 1.0 s hiccup is capped at 20 ticks (100 substeps // n_sub=5) with the
    other 0.9 s dropped into dropped_s, not carried — a zero-elapsed follow-up
    runs no more ticks (spiral-of-death guard)."""
    ticker = _make_ticker()
    clock = _ScriptedClock([0.0, 1.0, 1.0])  # construct@0; +1.0s hiccup; +0.0s
    loop = RealtimeLoop(ticker=ticker, x0=np.zeros(4), ctrl_dt_s=5e-3, now_fn=clock)

    first = loop.advance(_zero_tau)
    assert len(first) == 20                          # cap: 100 // n_sub(5)
    assert loop.dropped_s == pytest.approx(0.9)       # (200-20) ticks * 5 ms
    assert loop.t == pytest.approx(0.1)               # only 20 ticks' worth of sim time

    second = loop.advance(_zero_tau)                  # +0.0 s elapsed
    assert len(second) == 0                           # no backlog left to run
    assert loop.dropped_s == pytest.approx(0.9)        # unchanged: nothing new dropped


def test_reset_mid_gap_starts_fresh_without_catchup_burst():
    """reset() during a long wall-clock stall re-anchors the clock — the next
    advance() runs exactly 1 tick, not a capped catch-up burst — and zeroes
    dropped_s with x/t re-armed to (x0, 0) before any tick runs."""
    ticker = _make_ticker()
    # construct@0; frame@10ms (2 ticks); frame@1.010s (1s hiccup: 20 ticks,
    # 0.9s dropped, so "reset zeroes dropped_s" is meaningful); reset@3.0
    # (mid-gap); frame@3.005 (+5ms).
    clock = _ScriptedClock([0.0, 0.010, 1.010, 3.0, 3.005])
    x0_new = np.array([0.1, 0.05, 0.0, 0.0])
    loop = RealtimeLoop(ticker=ticker, x0=np.array([0.3, -0.2, 0.0, 0.0]),
                         ctrl_dt_s=5e-3, now_fn=clock)

    assert len(loop.advance(_zero_tau)) == 2           # session underway
    assert len(loop.advance(_zero_tau)) == 20          # hiccup: capped burst
    assert loop.dropped_s == pytest.approx(0.9)         # nonzero pre-reset

    loop.reset(x0_new)                                 # during the 2 s gap
    np.testing.assert_array_equal(loop.x, x0_new)       # re-anchored to x0...
    assert loop.t == 0.0                                # ...at sim t=0...
    assert loop.dropped_s == 0.0                        # ...drift counter fresh

    post = loop.advance(_zero_tau)                     # only 5 ms SINCE reset
    assert len(post) == 1                              # no burst: exactly 1 tick
    assert post[0]["t_s"] == 0.0                       # fresh session's tick 0
    np.testing.assert_array_equal(post[0]["x_true"], x0_new)  # starts from x0
    assert loop.dropped_s == 0.0                        # nothing dropped since


# ---------------------------------------------------------------------------
# The gate: live (RealtimeLoop, fake clock) ≡ batch (simulate()), bit-for-bit
# ---------------------------------------------------------------------------

def test_live_equals_batch_bit_identical():
    """A fake clock advancing exactly ctrl_dt_s per frame, zero forcing:
    RealtimeLoop reproduces simulate()'s Telemetry bit-for-bit (np.array_equal,
    no tolerance) — both paths call the identical ControlTicker.tick()."""
    plant = cart_plant()
    x0 = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    ctrl_dt_s, sim_dt_s, duration_s, seed = 5e-3, 1e-3, 0.5, 0
    n_ticks = round(duration_s / ctrl_dt_s)

    def zero_tau(t, x):
        return np.zeros(plant.n // 2)

    batch_tel = simulate(
        plant=plant, x0=x0, duration_s=duration_s, sim_dt_s=sim_dt_s,
        ctrl_dt_s=ctrl_dt_s, sensor=PerfectSensor(),
        estimator=IdentityEstimator(plant.n), controller=ZeroController(m=plant.m),
        seed=seed,
    )

    ticker = ControlTicker(
        plant=plant, sensor=PerfectSensor(), estimator=IdentityEstimator(plant.n),
        controller=ZeroController(m=plant.m), ctrl_dt_s=ctrl_dt_s, sim_dt_s=sim_dt_s,
        rng=np.random.default_rng(seed),
    )
    clock = _ScriptedClock([k * ctrl_dt_s for k in range(n_ticks + 1)])
    loop = RealtimeLoop(ticker=ticker, x0=x0, ctrl_dt_s=ctrl_dt_s, now_fn=clock)

    rec = Recorder()
    for _ in range(n_ticks):
        for record in loop.advance(zero_tau):
            rec.append(**record)
    live_tel = rec.finalize()

    for field in ("t_ns", "x_true", "x_hat", "y", "u", "tau_ext", "energy_J"):
        live_val = getattr(live_tel, field)
        batch_val = getattr(batch_tel, field)
        assert np.array_equal(live_val, batch_val), f"{field} differs between live and batch"


# ---------------------------------------------------------------------------
# ControlTicker.controller_provider: swap which controller answers update()
# at runtime on one shared ticker (shared sensor/estimator/physics state).
# ---------------------------------------------------------------------------

def test_ticker_uses_fixed_controller_when_no_provider_given():
    """Backward-compat pin: `controller=` alone resolves `ticker.controller`
    to that exact object, with no provider."""
    plant = fixed_pivot_plant()
    c = ZeroController(m=plant.m)
    ticker = ControlTicker(plant=plant, sensor=PerfectSensor(),
                           estimator=IdentityEstimator(plant.n), controller=c,
                           ctrl_dt_s=5e-3, sim_dt_s=1e-3)
    assert ticker.controller is c


def test_ticker_controller_provider_consulted_fresh_each_tick():
    """tick() re-resolves controller_provider() on every tick, not cached —
    proven by flipping a mutable selector between two tick() calls."""
    plant = fixed_pivot_plant()

    class _Tagging:
        """update() returns a constant `tag` and logs its call times, so the
        test can tell which controller answered which tick."""

        def __init__(self, tag):
            self.tag = tag
            self.seen_t = []

        def reset(self, t0, x0):
            pass

        def update(self, t, x_hat):
            self.seen_t.append(t)
            return np.full(plant.m, self.tag)

    a, b = _Tagging(1.0), _Tagging(2.0)
    selected = {"which": a}
    ticker = ControlTicker(plant=plant, sensor=PerfectSensor(),
                           estimator=IdentityEstimator(plant.n),
                           controller_provider=lambda: selected["which"],
                           ctrl_dt_s=5e-3, sim_dt_s=1e-3)
    ticker.reset(np.zeros(4))

    _, row0 = ticker.tick(0.0, np.zeros(4))
    assert row0["u"][0] == 1.0                       # `a` consulted this tick
    assert a.seen_t == [0.0] and b.seen_t == []       # `b` untouched so far

    selected["which"] = b                             # mode switch, mid-session
    _, row1 = ticker.tick(5e-3, np.zeros(4))
    assert row1["u"][0] == 2.0                        # now `b` is consulted
    assert b.seen_t == [5e-3] and a.seen_t == [0.0]   # `a` got no more calls


def test_ticker_requires_controller_or_provider():
    """Supplying neither `controller` nor `controller_provider` raises at
    construction, not as a silent AttributeError inside tick()."""
    plant = fixed_pivot_plant()
    with pytest.raises(ValueError):
        ControlTicker(plant=plant, sensor=PerfectSensor(),
                     estimator=IdentityEstimator(plant.n),
                     ctrl_dt_s=5e-3, sim_dt_s=1e-3)


def test_ticker_rejects_both_controller_and_provider():
    """Supplying both is ambiguous (which wins?) and is rejected at
    construction with a message naming the conflict."""
    plant = fixed_pivot_plant()
    c = ZeroController(m=plant.m)
    with pytest.raises(ValueError, match="not both"):
        ControlTicker(plant=plant, sensor=PerfectSensor(),
                     estimator=IdentityEstimator(plant.n),
                     controller=c, controller_provider=lambda: c,
                     ctrl_dt_s=5e-3, sim_dt_s=1e-3)


# ---------------------------------------------------------------------------
# ControlTicker.tick's tau_ext shape guard — every tau_ext producer passes
# through this one seam (see dpend/sim/ticker.py).
# ---------------------------------------------------------------------------

def test_ticker_tick_rejects_wrong_shape_tau_ext():
    """A tau_ext not shaped (plant.n // 2,) raises a ValueError naming given
    vs expected — never silently numpy-broadcast across all coordinates."""
    ticker = _make_ticker()  # fixed_pivot_plant(): n=4, n // 2 == 2
    ticker.reset(np.zeros(4))
    with pytest.raises(ValueError, match=r"\(1,\).*\(2,\)"):
        ticker.tick(0.0, np.zeros(4), tau_ext=np.array([1.0]))
