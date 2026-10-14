"""Check 1: simulation replay for a candidate Rule (CONTEXT.md section 4
step 4 / section 8).

check1(rule, train_seeds) replays the target object_class with and
without the rule, via the real sim/run_trial.py + sim/batch.py (never
hand-simulated), and judges PASS/FAIL against team-default thresholds
fixed below. Returns a Check1Outcome: a pipeline-level report that wraps
a REAL rules.lifecycle.Check1Record (not a hand-rolled substitute) plus
the extra diagnostics CONTEXT.md section 8 asks for (Wilson intervals,
paired fixed/newly-broken seeds, margin, slowdown) that Check1Record's
own fixed shape has no room for.

Does not touch rules/lifecycle.py's audit log itself -- that is
pipeline/loop.py's job (it decides which attempts are worth logging).
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(ROOT, "sim")
LLM_DIR = os.path.join(ROOT, "llm")
for _p in (SIM_DIR, LLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from batch import BatchSummary, run_batch  # sim/batch.py
from cell_model import phase_duration  # sim/cell_model.py
from run_trial import CARRY_DISTANCE_M  # sim/run_trial.py

import stats as llm_stats  # llm/stats.py -- Wilson interval, per spec

from rules.apply import RuleRejected, apply
from rules.lifecycle import Check1Record
from rules.limits import CRUSH_LIMIT_N, OBJECT_CLASSES
from rules.validate import validate

# ---- Team defaults (PASS/FAIL thresholds) -----------------------------
# These are pre-registered: fixed here, before any held-out run, per
# CONTEXT.md section 8's own methodology ("held-out seeds used only for
# the final measurement"). Changing them after looking at a held-out
# result would defeat the point of having a held-out set at all.
#
# MIN_FIXED_RATE_CI95_LOWER is checked against the Wilson 95% LOWER bound
# of (seeds_fixed / baseline_drops), not the raw point estimate -- a rule
# that fixes, say, 6/10 baseline drops (60% point estimate) has a wide,
# low-confidence interval at n=10 (Wilson 95%: ~0.31-0.85) and should not
# pass as confidently as one that fixes 60/100 (Wilson 95%: ~0.50-0.69).
# Requiring the LOWER bound to clear 0.5 means check1 only passes a rule
# once there is 95% confidence the true fixed-rate is at least 50%, which
# also makes DEFAULT_TRAIN_SEEDS' size (below) matter: more seeds narrow
# the interval and make a real 50%+ rule easier to confidently clear.
MIN_FIXED_RATE_CI95_LOWER = 0.5  # team default: 95% confident fixed-rate >= 50%.
MAX_CRUSH_COUNT = 0  # team default: zero tolerance for crushing the part.
MAX_NEWLY_BROKEN = 0  # team default: zero tolerance for regressing a previously-ok seed.

# Default number of TRAIN seeds check1() replays on, if the caller doesn't
# pass its own. 100 (seeds 0-99) was chosen so MIN_FIXED_RATE_CI95_LOWER's
# Wilson interval is narrow enough to be a meaningful gate (see above) --
# never the held-out range (sim/batch.py's HELD_OUT_SEEDS = 1000-1199).
# Configurable per call: pass a different `train_seeds` to check1().
DEFAULT_TRAIN_SEEDS = range(100)

# ---- Check 1 v2 ---------------------------------------------------------
# pre-registered v2, written before any 3000+ run.
#
# v1's gate (above) was found too weak after seeing held-out 2000-2199: it
# only required the Wilson 95% LOWER bound of (seeds_fixed / baseline_drops)
# to clear 0.5, which says nothing about the rule's own absolute after-rule
# drop rate -- a rule with ~28% drop rate still passed. v2 replaces that
# fixed-rate gate with a direct ceiling on the after-rule drop rate's Wilson
# UPPER bound, plus a cap on the slowdown proxy. MAX_CRUSH_COUNT,
# MAX_NEWLY_BROKEN, and the other-classes-unchanged check are unchanged
# from v1.
#
# CHECK1_VERSION picks the gate check1() uses when the caller doesn't pass
# its own `version`. Set to 2 (v2 is now the default); pass version=1 to
# any check1() call to recover the old gate -- v1's code path is kept,
# not deleted.
CHECK1_VERSION = 2
MAX_AFTER_DROP_RATE_CI95_UPPER = 0.10  # Wilson 95% upper bound of drop_rate_after must be <= this.
MAX_SLOWDOWN_PROXY = 1.5  # after/before carry-phase duration ratio must be <= this.

# "Without the rule" baseline controller settings -- the nominal, untuned
# setting a rule is supposed to fix glass_jar at, without touching the
# other two classes.
#
# sim/ has exactly one thing labeled "nominal": sim/tests/test_sim.py's
# test_glass_jar_drops_at_nominal_grip_and_is_fixed_within_range uses
# grip_force_n=2.0, carry_accel_ms2=1.0 (also monitor.py's/run_trial.py's
# own CLI defaults). Checked first, as instructed, but measured on
# TRAIN_SEEDS 0-99 (this pipeline's own run, not in sim/) it does NOT hold
# for the other two classes: steel drops 100% (n=100) and cardboard drops
# 29% (n=100) at that setting -- only glass_jar's own single-seed check in
# test_sim.py was ever verified, not a 100-seed sweep, and not cross-class.
# That setting is "nominal" in the sense of "a round number someone
# tried", not "the setting at which only glass_jar is broken", which is
# what check1's baseline needs to mean for its before/after comparison
# and the "other classes unchanged" regression check to tell a true story.
#
# Chosen instead (measured on TRAIN_SEEDS 0-99, this pipeline, before any
# held-out run): grip_force_n=5.0 N, carry_accel_ms2=1.0 m/s^2.
#   steel:      0/100 drop  (5.0N clears sim/STATUS.md's measured mean
#               min holding grip of ~3.8-4.2N with margin)
#   cardboard:  0/100 drop  (5.0N clears its measured mean min holding
#               grip of ~1.9-2.2N with a lot of margin)
#   glass_jar: 100/100 drop (5.0N is far below its measured mean min
#               holding grip of ~17-20N)
# A clean split: steel and cardboard always hold, glass_jar always drops.
# Fixed here, before any held-out run; never re-measured against or tuned
# on HELD_OUT_SEEDS (1000+).
BASELINE_PARAMS = {"grip_force_n": 5.0, "carry_accel_ms2": 1.0}
NOMINAL_PARAMS = BASELINE_PARAMS  # alias: this IS "the nominal setting" referred to elsewhere (loop.py, evaluate_heldout.py)


def _class_value(object_class) -> str:
    return object_class.value if hasattr(object_class, "value") else object_class


@dataclass
class Check1Outcome:
    passed: bool
    reasons: list = field(default_factory=list)
    record: Optional[Check1Record] = None  # the REAL rules.lifecycle.Check1Record
    drop_rate_before: float = 0.0
    drop_rate_after: float = 0.0
    drop_rate_before_ci95: tuple = (0.0, 0.0)
    drop_rate_after_ci95: tuple = (0.0, 0.0)
    seeds_fixed: list = field(default_factory=list)
    seeds_newly_broken: list = field(default_factory=list)
    fixed_rate_ci95: Optional[tuple] = None  # Wilson 95% CI of fixed/baseline_drops; see MIN_FIXED_RATE_CI95_LOWER
    crush_count_after: int = 0
    margin: Optional[float] = None  # 1 - grip_force_min_n / crush_limit
    slowdown_proxy: Optional[float] = None  # carry-phase duration ratio, after/before
    n_trials: int = 0
    version: int = 1  # which CHECK1_VERSION gate produced this outcome
    intervention_cost_grip: Optional[float] = None  # |after.grip - nominal.grip| / nominal.grip
    intervention_cost_accel: Optional[float] = None  # |after.accel - nominal.accel| / nominal.accel


def _failed(reasons, record=None) -> Check1Outcome:
    return Check1Outcome(passed=False, reasons=reasons, record=record)


def _v1_primary_gate(seeds_fixed, n_baseline_drops, fixed_rate_ci95):
    """v1's own gate: Wilson 95% CI lower bound of fixed/baseline_drops >= MIN_FIXED_RATE_CI95_LOWER.
    Pure function of already-computed summary numbers -- no sim call -- so it's directly unit-testable.
    """
    if n_baseline_drops == 0:
        return False, ["baseline has zero drops for this class/seeds -- nothing to fix, cannot assess fixed rate"]
    fix_fraction = len(seeds_fixed) / n_baseline_drops
    if fixed_rate_ci95[0] < MIN_FIXED_RATE_CI95_LOWER:
        return False, [
            f"fixed {len(seeds_fixed)}/{n_baseline_drops} baseline drops ({fix_fraction:.0%} point "
            f"estimate), but the Wilson 95% CI lower bound {fixed_rate_ci95[0]:.3f} is below the "
            f"required {MIN_FIXED_RATE_CI95_LOWER}"
        ]
    return True, []


def _v2_primary_gate(drop_rate_after, n_drop_after, n_trials_after, drop_rate_after_ci95, slowdown_proxy):
    """v2's own gate: Wilson 95% CI upper bound of the after-rule drop rate <= MAX_AFTER_DROP_RATE_CI95_UPPER,
    and slowdown_proxy <= MAX_SLOWDOWN_PROXY. Pure function of already-computed summary numbers -- no sim
    call -- so it's directly unit-testable.
    """
    reasons = []
    ok = True
    if drop_rate_after_ci95[1] > MAX_AFTER_DROP_RATE_CI95_UPPER:
        reasons.append(
            f"after-rule drop rate {drop_rate_after:.3f} ({n_drop_after}/{n_trials_after}), Wilson 95% CI "
            f"upper bound {drop_rate_after_ci95[1]:.3f} exceeds the required <= {MAX_AFTER_DROP_RATE_CI95_UPPER}"
        )
        ok = False
    if slowdown_proxy > MAX_SLOWDOWN_PROXY:
        reasons.append(f"slowdown_proxy {slowdown_proxy:.3f} exceeds the required <= {MAX_SLOWDOWN_PROXY}")
        ok = False
    return ok, reasons


def _common_gates(crush_count_after, seeds_newly_broken, other_classes_unchanged):
    """Gates shared by v1 and v2, unchanged between the two versions."""
    reasons = []
    ok = True
    if crush_count_after > MAX_CRUSH_COUNT:
        reasons.append(f"{crush_count_after} crush(es) after the rule (must be <= {MAX_CRUSH_COUNT})")
        ok = False
    if len(seeds_newly_broken) > MAX_NEWLY_BROKEN:
        reasons.append(f"{len(seeds_newly_broken)} previously-ok seed(s) newly broken: {seeds_newly_broken}")
        ok = False
    if not other_classes_unchanged:
        reasons.append("apply()/replay changed results for a non-target object class")
        ok = False
    return ok, reasons


def check1(
    rule,
    train_seeds=None,
    parallel: bool = True,
    precomputed_before: Optional[BatchSummary] = None,
    version: Optional[int] = None,
) -> Check1Outcome:
    """Replay `rule` on `train_seeds` (default DEFAULT_TRAIN_SEEDS, i.e.
    seeds 0-99 -- pass a different range/list to use fewer, e.g. for a
    fast test) and judge PASS/FAIL.

    `precomputed_before` lets a caller (e.g. a grid search trying many
    candidate rules for the same object_class back to back) reuse one
    already-run baseline batch instead of re-running the identical
    BASELINE_PARAMS replay for every candidate.

    `version` selects the gate: 1 for the original fixed-rate-lower-bound
    gate, 2 for the v2 gate (see the "Check 1 v2" block above). Defaults to
    CHECK1_VERSION.
    """
    if train_seeds is None:
        train_seeds = DEFAULT_TRAIN_SEEDS
    if version is None:
        version = CHECK1_VERSION
    if version not in (1, 2):
        raise ValueError(f"unknown check1 version {version!r}; must be 1 or 2")

    val = validate(rule)
    if not val.ok:
        return _failed([f"rules.validate failed: {val.error_text}"])

    target_class = _class_value(rule.object_class)
    before_params = dict(BASELINE_PARAMS)
    try:
        after_params = apply(rule, before_params, target_class)
    except RuleRejected as e:
        return _failed([f"apply() rejected the rule: {e}"])

    before = precomputed_before
    if before is None or before.object_class != target_class or before.controller != before_params:
        before = run_batch(target_class, before_params, train_seeds, parallel=parallel)
    after = run_batch(target_class, after_params, train_seeds, parallel=parallel)

    before_by_seed = {r.seed: r.outcome for r in before.results}
    after_by_seed = {r.seed: r.outcome for r in after.results}
    seeds_fixed = sorted(
        s for s, o in before_by_seed.items() if o == "drop" and after_by_seed.get(s) == "ok"
    )
    seeds_newly_broken = sorted(
        s for s, o in before_by_seed.items() if o == "ok" and after_by_seed.get(s) in ("drop", "crush")
    )
    crush_count_after = after.n_crush

    # Regression check on the OTHER two classes (CONTEXT.md section 8 item
    # 2: a rule must not "regress another class"). apply() is defined to
    # be a no-op whenever rule.object_class doesn't match, so this is a
    # live, empirical confirmation of that guarantee via the real sim,
    # not a re-statement of it.
    other_classes_unchanged = True
    for other_class in OBJECT_CLASSES:
        if other_class == target_class:
            continue
        other_after_params = apply(rule, before_params, other_class)
        if other_after_params != before_params:
            other_classes_unchanged = False
            continue
        other_before = run_batch(other_class, before_params, train_seeds, parallel=parallel)
        other_after = run_batch(other_class, other_after_params, train_seeds, parallel=parallel)
        if [r.as_dict() for r in other_before.results] != [r.as_dict() for r in other_after.results]:
            other_classes_unchanged = False

    crush_limit = CRUSH_LIMIT_N[target_class]
    margin = 1.0 - rule.grip_force_min_n / crush_limit

    before_duration = phase_duration(CARRY_DISTANCE_M, before_params["carry_accel_ms2"])
    after_duration = phase_duration(CARRY_DISTANCE_M, after_params["carry_accel_ms2"])
    slowdown_proxy = after_duration / before_duration

    drop_rate_after_ci95 = llm_stats.wilson(after.n_drop, after.n_trials)

    intervention_cost_grip = abs(after_params["grip_force_n"] - before_params["grip_force_n"]) / before_params["grip_force_n"]
    intervention_cost_accel = abs(after_params["carry_accel_ms2"] - before_params["carry_accel_ms2"]) / before_params["carry_accel_ms2"]

    n_baseline_drops = before.n_drop
    fixed_rate_ci95 = llm_stats.wilson(len(seeds_fixed), n_baseline_drops) if n_baseline_drops > 0 else None

    if version == 1:
        primary_ok, primary_reasons = _v1_primary_gate(seeds_fixed, n_baseline_drops, fixed_rate_ci95)
    else:
        primary_ok, primary_reasons = _v2_primary_gate(after.drop_rate, after.n_drop, after.n_trials, drop_rate_after_ci95, slowdown_proxy)
    common_ok, common_reasons = _common_gates(crush_count_after, seeds_newly_broken, other_classes_unchanged)

    reasons = [
        f"intervention cost: grip {intervention_cost_grip:+.1%} of nominal "
        f"({before_params['grip_force_n']}N -> {after_params['grip_force_n']}N), accel "
        f"{intervention_cost_accel:+.1%} of nominal "
        f"({before_params['carry_accel_ms2']}m/s^2 -> {after_params['carry_accel_ms2']}m/s^2)"
    ] + primary_reasons + common_reasons
    ok = primary_ok and common_ok

    if ok:
        if version == 1:
            reasons.append(
                f"passed (v1): validate ok, crush_count=0, newly_broken=0, fixed {len(seeds_fixed)}/{n_baseline_drops} "
                f"(Wilson 95% CI lower bound {fixed_rate_ci95[0]:.3f} >= {MIN_FIXED_RATE_CI95_LOWER})"
            )
        else:
            reasons.append(
                f"passed (v2): validate ok, crush_count=0, newly_broken=0, drop_rate_after={after.drop_rate:.3f} "
                f"(Wilson 95% CI upper bound {drop_rate_after_ci95[1]:.3f} <= {MAX_AFTER_DROP_RATE_CI95_UPPER}), "
                f"slowdown_proxy={slowdown_proxy:.3f} <= {MAX_SLOWDOWN_PROXY}"
            )

    record = Check1Record(
        target_class=target_class,
        drops_before=before.n_drop,
        drops_after=after.n_drop,
        other_classes_unchanged=other_classes_unchanged,
        crush_count=crush_count_after,
        cited_report_ids=list(rule.source_report_ids),
        intervention_cost_grip=intervention_cost_grip,
        intervention_cost_accel=intervention_cost_accel,
    )

    return Check1Outcome(
        passed=ok,
        reasons=reasons,
        record=record,
        drop_rate_before=before.drop_rate,
        drop_rate_after=after.drop_rate,
        drop_rate_before_ci95=llm_stats.wilson(before.n_drop, before.n_trials),
        drop_rate_after_ci95=drop_rate_after_ci95,
        seeds_fixed=seeds_fixed,
        seeds_newly_broken=seeds_newly_broken,
        fixed_rate_ci95=fixed_rate_ci95,
        crush_count_after=crush_count_after,
        margin=margin,
        slowdown_proxy=slowdown_proxy,
        n_trials=before.n_trials,
        version=version,
        intervention_cost_grip=intervention_cost_grip,
        intervention_cost_accel=intervention_cost_accel,
    )


if __name__ == "__main__":
    import argparse
    import datetime as dt

    from rules.schema import Rule

    p = argparse.ArgumentParser(description="Run Check 1 on a hand-built glass_jar rule, for a quick manual check.")
    p.add_argument("--grip-force-min-n", type=float, default=14.0)
    p.add_argument("--carry-accel-max-ms2", type=float, default=2.5)
    p.add_argument("--object-class", default="glass_jar")
    p.add_argument("--n-seeds", type=int, default=len(DEFAULT_TRAIN_SEEDS))
    args = p.parse_args()

    rule = Rule(
        rule_id="r_manual_check1",
        object_class=args.object_class,
        grip_force_min_n=args.grip_force_min_n,
        carry_accel_max_ms2=args.carry_accel_max_ms2,
        proposed_by="manual",
        created_at=dt.datetime.now(dt.timezone.utc),
        rationale="manual check1.py invocation",
        source_report_ids=["manual"],
    )
    outcome = check1(rule, range(args.n_seeds))
    print(f"PASSED: {outcome.passed}")
    for r in outcome.reasons:
        print(f"  - {r}")
    print(f"drop_rate_before={outcome.drop_rate_before} (CI95 {outcome.drop_rate_before_ci95})")
    print(f"drop_rate_after={outcome.drop_rate_after} (CI95 {outcome.drop_rate_after_ci95})")
    print(f"seeds_fixed={outcome.seeds_fixed}")
    print(f"seeds_newly_broken={outcome.seeds_newly_broken}")
    print(f"margin={outcome.margin:.3f} slowdown_proxy={outcome.slowdown_proxy:.3f}")
    print(f"record={outcome.record}")
