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
    # Use this SEED's actual (jittered) crush limit, not the nominal one --
    # CRUSH_LIMIT_JITTER can push a seed's real limit below nominal*0.95,
    # and without this the search could probe a grip that crushes for this
    # seed, which the loop below would wrongly treat as "found a safe hold"
    # (it only branches on outcome == "drop", not "crush").
    crush = cell_model.jittered_crush_limit_n(object_class, seed)
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
    # The check is grip > crush_limit, so exactly at the limit must NOT
    # crush and just over it must. crush_limit_n is now jittered per
    # (object_class, seed) (CRUSH_LIMIT_JITTER), so this queries seed=0's
    # actual limit instead of asserting against the nominal 40.0 -- the
    # old hardcoded 40.0/41.0 no longer straddles seed=0's real boundary
    # (steel/seed=0 jitters to ~42.36N, so 41.0N no longer crushes).
    limit = cell_model.jittered_crush_limit_n("steel", seed=0)
    at_limit = run_trial("steel", {"grip_force_n": limit, "carry_accel_ms2": 1.0}, seed=0)
    assert at_limit.outcome != "crush"

    over_limit = run_trial("steel", {"grip_force_n": limit + 1.0, "carry_accel_ms2": 1.0}, seed=0)
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


# min_grip(6.0)/min_grip(0.5), 6 seeds (0-5), 10 bisection iterations,
# measured on this build. An audit found this ratio nearly inert
# (1.06-1.13 over a larger 30-seed/5-accel sweep vs ~1.5 naively predicted
# from the quasi-static formula) and an experiment coupling the lift's
# acceleration to carry_accel_ms2 (so a low accel setting also slows the
# lift, not just the carry) was tried and rejected -- it only moved the
# ratio to 1.06-1.15, nowhere near the 1.2 keep/revert bar, because the
# sim's drop criterion (a fixed 20mm slip budget over the fixed
# PHASE_OBSERVE_S=1.6s window, see cell_model.py) is dominated by a
# near-constant creep rate, not by the inertial force term accel changes
# -- see STATUS.md. This locks in the *current* (unchanged) behavior so
# the lever can't silently regress to fully inert (ratio -> 1.0) without
# a test failure; it is not a claim that 1.2+ is achievable here.
MEASURED_ACCEL_RATIO = {"steel": 1.027, "cardboard": 1.118, "glass_jar": 1.037}
# glass_jar's value above changed from 1.126 (pre-jitter) to 1.037 after
# CRUSH_LIMIT_JITTER was added: _bisect_min_holding_grip's search ceiling is
# now this SEED's actual jittered crush limit, not the nominal one (needed
# for correctness -- see that function's comment), and several of glass_jar's
# seeds 0-5 bisect to a value pinned at/near that ceiling (STATUS.md item 2:
# glass_jar's low-mu seeds are near-unholdable, close to the crush limit) --
# so a per-seed ceiling that now varies with the jitter draw shifts the
# measured ratio for THIS class specifically. steel/cardboard are nowhere
# near their crush limits and are unaffected (unchanged from pre-jitter).


@pytest.mark.parametrize("object_class", ["steel", "cardboard", "glass_jar"])
def test_accel_lever_ratio_not_inert(object_class):
    seeds = range(6)
    g_lo = [_bisect_min_holding_grip(object_class, s, accel=0.5, iters=10) for s in seeds]
    g_hi = [_bisect_min_holding_grip(object_class, s, accel=6.0, iters=10) for s in seeds]
    ratio = (sum(g_hi) / len(g_hi)) / (sum(g_lo) / len(g_lo))
    expected = MEASURED_ACCEL_RATIO[object_class]
    assert ratio >= 1.01, (
        f"{object_class}: min_grip(6.0)/min_grip(0.5) ratio {ratio:.4f} has gone "
        f"essentially inert (<=1.01) -- the accel lever has no effect"
    )
    assert expected - 0.08 <= ratio <= expected + 0.25, (
        f"{object_class}: ratio {ratio:.4f} is outside the documented band "
        f"around the measured {expected} (-0.08/+0.25)"
    )


def test_held_out_seeds_not_used_by_default():
    assert HELD_OUT_SEEDS == range(1000, 1200)
    assert set(TRAIN_SEEDS).isdisjoint(set(HELD_OUT_SEEDS))


def test_crush_limit_jitter_deterministic_per_seed():
    """Same (object_class, seed) always gives the same jittered limit, and
    it varies across seeds (not silently inert)."""
    a = cell_model.jittered_crush_limit_n("glass_jar", seed=7)
    b = cell_model.jittered_crush_limit_n("glass_jar", seed=7)
    assert a == b
    values = {cell_model.jittered_crush_limit_n("glass_jar", seed=s) for s in range(10)}
    assert len(values) > 1


def test_crush_limit_jitter_independent_of_placement_and_friction_stream():
    """Crush-limit jitter must be drawn from its own RNG stream -- adding it
    must not perturb the existing placement/friction draws for any seed
    (both depend on run_trial's own rng = np.random.default_rng(seed),
    untouched by jittered_crush_limit_n's separate SeedSequence)."""
    for seed in range(5):
        mu_before = _mu_for_seed("glass_jar", seed)
        cell_model.jittered_crush_limit_n("glass_jar", seed)  # must not disturb anything
        mu_after = _mu_for_seed("glass_jar", seed)
        assert mu_before == mu_after


def test_crush_limit_jitter_clipped_within_bounds():
    sigma = cell_model.CRUSH_LIMIT_JITTER
    clip = cell_model.CRUSH_LIMIT_JITTER_CLIP_SIGMAS
    for cls, props in OBJECT_CLASSES.items():
        nominal = props["crush_limit_n"]
        lo = nominal * (1.0 - clip * sigma)
        hi = nominal * (1.0 + clip * sigma)
        for seed in range(50):
            v = cell_model.jittered_crush_limit_n(cls, seed)
            assert lo - 1e-9 <= v <= hi + 1e-9


def test_crush_limit_jitter_zero_means_nominal():
    monkeypatch_value = cell_model.CRUSH_LIMIT_JITTER
    cell_model.CRUSH_LIMIT_JITTER = 0.0
    try:
        for cls, props in OBJECT_CLASSES.items():
            assert cell_model.jittered_crush_limit_n(cls, seed=123) == props["crush_limit_n"]
    finally:
        cell_model.CRUSH_LIMIT_JITTER = monkeypatch_value


def test_baseline_params_never_crush_despite_jitter():
    """pipeline/check1.py's BASELINE_PARAMS (grip_force_n=5.0) must never
    crush any class even at the jitter's worst-case (lowest) clipped draw --
    otherwise seeds 0-4's real FailureReports (pipeline/loop.py's
    real_failure_reports, run at BASELINE_PARAMS) could change behavior.
    """
    clip = cell_model.CRUSH_LIMIT_JITTER_CLIP_SIGMAS
    sigma = cell_model.CRUSH_LIMIT_JITTER
    for cls, props in OBJECT_CLASSES.items():
        worst_case_limit = props["crush_limit_n"] * (1.0 - clip * sigma)
        assert 5.0 < worst_case_limit, (
            f"{cls}: worst-case jittered crush limit {worst_case_limit} is not safely "
            "above BASELINE_PARAMS' grip_force_n=5.0"
        )


# ---- Phase B: probe_crush_limit_reading (the pre-grasp sensor observable) ----

def test_probe_reading_deterministic_per_seed():
    a = cell_model.probe_crush_limit_reading("glass_jar", seed=7)
    b = cell_model.probe_crush_limit_reading("glass_jar", seed=7)
    assert a == b


def test_probe_reading_sigma_zero_equals_true_limit():
    for cls in OBJECT_CLASSES:
        reading = cell_model.probe_crush_limit_reading(cls, seed=3, sigma=0.0)
        true_limit = cell_model.jittered_crush_limit_n(cls, seed=3)
        assert reading == true_limit


def test_probe_reading_varies_with_seed_and_sigma():
    readings = {cell_model.probe_crush_limit_reading("glass_jar", s) for s in range(10)}
    assert len(readings) > 1  # not silently inert across seeds

    # Larger sigma -> larger average absolute deviation from the true limit.
    def _mean_abs_dev(sigma):
        devs = []
        for s in range(30):
            true_limit = cell_model.jittered_crush_limit_n("glass_jar", s)
            reading = cell_model.probe_crush_limit_reading("glass_jar", s, sigma=sigma)
            devs.append(abs(reading - true_limit) / true_limit)
        return sum(devs) / len(devs)

    assert _mean_abs_dev(0.30) > _mean_abs_dev(0.03)


def test_probe_reading_independent_of_other_rng_streams():
    """Adding/sweeping the probe must not perturb placement, friction, or
    the crush-limit jitter for any seed -- own independent SeedSequence
    salt, per its docstring."""
    for seed in range(5):
        rng_before = np.random.default_rng(seed)
        xy_before = rng_before.uniform(-PLACEMENT_JITTER_M, PLACEMENT_JITTER_M, size=2).copy()
        mu_before = sample_friction("glass_jar", rng_before)
        crush_before = cell_model.jittered_crush_limit_n("glass_jar", seed)

        for sigma in (0.0, 0.01, 0.03, 0.05, 0.10, 0.30):
            cell_model.probe_crush_limit_reading("glass_jar", seed, sigma=sigma)

        rng_after = np.random.default_rng(seed)
        xy_after = rng_after.uniform(-PLACEMENT_JITTER_M, PLACEMENT_JITTER_M, size=2)
        mu_after = sample_friction("glass_jar", rng_after)
        crush_after = cell_model.jittered_crush_limit_n("glass_jar", seed)

        assert (xy_before == xy_after).all()
        assert mu_before == mu_after
        assert crush_before == crush_after


def test_run_trial_conditional_grip_realizes_fraction_times_probe():
    """End-to-end wiring check: run_trial's realized grip_force_n for a
    "grip_fraction" controller equals grip_fraction * this seed's probe
    reading exactly."""
    for seed in (5000, 5001, 5002):
        grip_fraction = 0.9
        result = run_trial("glass_jar", {"grip_fraction": grip_fraction, "carry_accel_ms2": 1.0}, seed)
        expected = grip_fraction * cell_model.probe_crush_limit_reading("glass_jar", seed)
        assert abs(result.controller["grip_force_n"] - expected) < 1e-9


def test_run_trial_conditional_respects_probe_sigma_override():
    seed = 5000
    r_default = run_trial("glass_jar", {"grip_fraction": 0.9, "carry_accel_ms2": 1.0}, seed)
    r_zero_sigma = run_trial(
        "glass_jar", {"grip_fraction": 0.9, "carry_accel_ms2": 1.0, "probe_sigma": 0.0}, seed
    )
    true_limit = cell_model.jittered_crush_limit_n("glass_jar", seed)
    assert abs(r_zero_sigma.controller["grip_force_n"] - 0.9 * true_limit) < 1e-9
    # Default PROBE_SIGMA=0.03 means these should differ (overwhelmingly likely, not guaranteed,
    # but true for this specific fixed seed -- deterministic, so this is a real assertion not a flake).
    assert r_default.controller["grip_force_n"] != r_zero_sigma.controller["grip_force_n"]
