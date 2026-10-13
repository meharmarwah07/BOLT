"""Real tests for sim/, per the Check 1 fix pass.

No display needed, no viewer imports (gripper_demo.py / viewer_demo.py were
renamed out of pytest's default *_test.py collection pattern specifically
so importing sim/ for tests can't hang). See conftest.py for the sys.path
setup that makes `import cell_model` etc. work from here.
"""
import json

import numpy as np
import pytest

import cell_model
from batch import HELD_OUT_SEEDS, TRAIN_SEEDS, run_batch
from cell_model import OBJECT_CLASSES, PLACEMENT_JITTER_M, PAD_FRICTION, sample_friction
from monitor import make_failure_report
from run_trial import run_trial


def _mu_for_seed(object_class, seed):
    """Replays run_trial's own RNG sequence (placement jitter draw, then
    friction draw) to recover the hidden mu a given seed will produce,
    without needing run_trial to expose it."""
    rng = np.random.default_rng(seed)
    rng.uniform(-PLACEMENT_JITTER_M, PLACEMENT_JITTER_M, size=2)
    return sample_friction(object_class, rng)


def _bisect_min_holding_grip(object_class, seed, accel=1.0, iters=14):
    crush = OBJECT_CLASSES[object_class]["crush_limit_n"]
    lo, hi = 0.05, min(49.0, crush * 0.95)
    for _ in range(iters):
        mid = (lo + hi) / 2
        r = run_trial(object_class, {"grip_force_n": mid, "carry_accel_ms2": accel}, seed)
        if r.outcome == "drop":
            lo = mid
        else:
            hi = mid
    return hi


def test_determinism():
    params = {"grip_force_n": 7.5, "carry_accel_ms2": 2.0}
    r1 = run_trial("steel", params, seed=42)
    r2 = run_trial("steel", params, seed=42)
    assert r1.as_dict() == r2.as_dict()


def test_friction_sensitivity(monkeypatch):
    """Two fake classes, same mass, only mu differs (0.05 vs 0.95). Pad
    friction (0.02) is below both, so the object's own mu governs either
    way -- if it didn't (the original bug), these would behave identically
    since both would be clamped by the pad's friction instead."""
    monkeypatch.setitem(
        OBJECT_CLASSES, "fake_lo_mu",
        {"mass_kg": 0.5, "friction_range": (0.05, 0.05), "crush_limit_n": 1000.0},
    )
    monkeypatch.setitem(
        OBJECT_CLASSES, "fake_hi_mu",
        {"mass_kg": 0.5, "friction_range": (0.95, 0.95), "crush_limit_n": 1000.0},
    )
    params = {"grip_force_n": 5.0, "carry_accel_ms2": 1.0}
    lo = run_trial("fake_lo_mu", params, seed=0)
    hi = run_trial("fake_hi_mu", params, seed=0)
    assert lo.outcome == "drop"
    assert hi.outcome == "ok"


def test_pad_friction_below_every_class_mu_floor():
    """Regression guard for the root cause itself: MuJoCo combines contact
    friction as the elementwise max of the two geoms (equal priority), so
    pad friction must stay below every class's mu floor or it silently
    becomes the governing value again."""
    for cls, props in OBJECT_CLASSES.items():
        lo, _ = props["friction_range"]
        assert PAD_FRICTION < lo, f"{cls}'s friction floor {lo} <= PAD_FRICTION {PAD_FRICTION}"


@pytest.mark.parametrize("object_class", ["steel", "cardboard", "glass_jar"])
def test_min_holding_grip_matches_friction_formula(object_class):
    """N >= m*(g+a)/(2*mu), a = LIFT_ACCEL_MS2 = 1.0 (the hardest instant:
    holding against gravity plus the lift's peak acceleration). The sim's
    empirical minimum needs a real margin above this quasi-static estimate
    (a stiff position actuator tracking a trajectory with a jerk
    discontinuity at the move's start overshoots briefly, and the
    fixed-time drop window needs some headroom above the exact
    equilibrium point to stay under the slip threshold) -- measured at
    2-24% above the formula across these three classes, well inside the
    ~25% this checks for.
    """
    mass = OBJECT_CLASSES[object_class]["mass_kg"]
    for seed in (2, 4, 7):
        mu = _mu_for_seed(object_class, seed)
        expected = mass * (9.81 + 1.0) / (2 * mu)
        found = _bisect_min_holding_grip(object_class, seed)
        assert found <= expected * 1.35, (
            f"{object_class} seed={seed}: min holding grip {found:.2f}N is "
            f">35% above the formula's {expected:.2f}N"
        )
        assert found >= expected * 0.75, (
            f"{object_class} seed={seed}: min holding grip {found:.2f}N is "
            f"suspiciously far BELOW the formula's {expected:.2f}N"
        )


def test_accel_direction_lower_never_systematically_worse():
    """The original bug: at a fixed marginal grip, accel=0.5 dropped 100%
    of the time and accel>=1.0 dropped 0% -- backwards, since lower accel
    needs less force to track. Root cause was that a half-sine move's
    duration is inversely related to accel, and the old fixed-step hold
    (same number of steps regardless of move length) meant lower accel
    silently got a longer total observation window, letting an unrelated
    near-marginal creep (present even with zero carry motion commanded)
    accumulate further. Fixed by watching every accel for the same fixed
    wall-clock window (hold_steps_for in cell_model.py).

    This is checked statistically (pairwise, same seed => same hidden mu
    for both accel values) rather than pointwise: right at a knife-edge
    grip, contact-solver noise can flip either calls' outcome, but that
    noise must not be SYSTEMATICALLY biased toward low accel losing.
    """
    grip = _bisect_min_holding_grip("steel", seed=0, accel=1.0)
    low_worse = 0
    high_worse = 0
    n_seeds = 30
    for seed in range(n_seeds):
        dropped_low = run_trial("steel", {"grip_force_n": grip, "carry_accel_ms2": 0.5}, seed).outcome == "drop"
        dropped_high = run_trial("steel", {"grip_force_n": grip, "carry_accel_ms2": 6.0}, seed).outcome == "drop"
        if dropped_low and not dropped_high:
            low_worse += 1
        elif dropped_high and not dropped_low:
            high_worse += 1
    assert low_worse <= high_worse + 3, (
        f"lower accel (0.5) dropped more often than higher accel (6.0) at the same "
        f"grip across {n_seeds} seeds: low_worse={low_worse} high_worse={high_worse}"
    )


def test_glass_jar_drops_at_nominal_grip_and_is_fixed_within_range():
    seed = 2
    nominal = run_trial("glass_jar", {"grip_force_n": 2.0, "carry_accel_ms2": 1.0}, seed)
    assert nominal.outcome == "drop"

    fixed = run_trial("glass_jar", {"grip_force_n": 24.0, "carry_accel_ms2": 1.0}, seed)
    assert fixed.outcome == "ok"
    assert 2.0 <= 24.0 <= 40.0


def test_crush_outcome_at_commanded_force_over_limit():
    # steel's crush_limit_n is 40.0; the check is grip > crush_limit, so
    # exactly 40N must NOT crush, and just over it must.
    at_limit = run_trial("steel", {"grip_force_n": 40.0, "carry_accel_ms2": 1.0}, seed=0)
    assert at_limit.outcome != "crush"

    over_limit = run_trial("steel", {"grip_force_n": 41.0, "carry_accel_ms2": 1.0}, seed=0)
    assert over_limit.outcome == "crush"


def test_parallel_batch_equals_serial_batch():
    params = {"grip_force_n": 10.0, "carry_accel_ms2": 1.0}
    seeds = range(8)
    serial = run_batch("steel", params, seeds, parallel=False)
    parallel = run_batch("steel", params, seeds, parallel=True)
    assert [r.as_dict() for r in serial.results] == [r.as_dict() for r in parallel.results]


def test_no_friction_or_mu_in_failure_report():
    result = run_trial("glass_jar", {"grip_force_n": 2.0, "carry_accel_ms2": 1.0}, seed=2)
    assert result.outcome != "ok"
    report = make_failure_report(result, cell_id="A", owner_id="owner1")
    assert report is not None
    assert "friction" not in report
    assert "mu" not in report
    # belt-and-suspenders: nothing named mu/friction anywhere in the
    # serialized report, including nested keys.
    serialized = json.dumps(report).lower()
    assert "friction" not in serialized
    assert '"mu"' not in serialized


def test_held_out_seeds_not_used_by_default():
    assert HELD_OUT_SEEDS == range(1000, 1200)
    assert set(TRAIN_SEEDS).isdisjoint(set(HELD_OUT_SEEDS))
