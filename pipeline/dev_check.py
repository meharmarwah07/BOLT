"""Dev-only probe: does ANY (grip, accel) rule for glass_jar clear Check 1
v2 (pipeline/check1.py) now that sim/cell_model.py's CRUSH_LIMIT_JITTER is
in play?

Runs the real simulator (never hand-simulated) against three reference
rules -- the latest pipeline/runs/accepted_rule.json, a naive
22.5N/0.5m/s^2 heuristic, and a 24.9N/0.5m/s^2 "ceiling" heuristic -- plus
a coarse grid over (grip_force_min_n, carry_accel_max_ms2), and prints a
table of drop rate (with Wilson 95% CI), crush count, slowdown_proxy, and
PASS/FAIL per rule.

Seeds are DEV_SEEDS = range(5000, 5100) ONLY -- this project's reserved
dev/iteration range. Never TRAIN (0-199) and never any held-out range
(1000-1199, 2000-2199, 3000-3199): this script answers a methodology
question (is the 2-parameter space even capable of clearing v2?), not a
real Check 1 run or a held-out measurement, so it must not touch either.
"""
import argparse
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(ROOT, "sim")
LLM_DIR = os.path.join(ROOT, "llm")
for _p in (PIPELINE_DIR, SIM_DIR, LLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check1 as check1_mod  # pipeline/check1.py
from batch import run_batch  # sim/batch.py
from cell_model import phase_duration  # sim/cell_model.py
from run_trial import CARRY_DISTANCE_M  # sim/run_trial.py
import stats as llm_stats  # llm/stats.py

from rules.schema import Rule

DEV_SEEDS = range(5000, 5100)
TARGET_CLASS = "glass_jar"
RUNS_DIR = os.path.join(PIPELINE_DIR, "runs")
ACCEPTED_RULE_JSON = os.path.join(RUNS_DIR, "accepted_rule.json")

# Reference rules named in the task: a naive "0.9x crush limit" style
# heuristic and a "right at the ceiling" heuristic, both at the datasheet's
# minimum legal accel (the most aggressive legal choice on that axis).
NAIVE_GRIP, NAIVE_ACCEL = 22.5, 0.5
CEILING_GRIP, CEILING_ACCEL = 24.9, 0.5

# Coarse grid: grip from comfortably-safe up to just under the nominal
# crush limit (25.0N -- rules.validate() rejects anything above it
# regardless of jitter), accel across the full datasheet range.
GRID_GRIPS = [14.0, 16.0, 18.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.9]
GRID_ACCELS = [0.5, 1.0, 2.0, 3.0, 6.0]

# Finer grid for --frontier (A2): 0.5N steps from 14.0 to 24.5, plus the
# same 24.9 "near ceiling" endpoint used above -- fine enough to find the
# drop/crush trade-off's actual minimum (not just bracket it).
FRONTIER_GRIPS = [14.0 + 0.5 * i for i in range(22)] + [24.9]  # 14.0, 14.5, ..., 24.5, 24.9

# A rule's carry_accel_max_ms2 is a CAP, combined with the baseline via
# rules/apply.py's effective_settings: after_accel = min(current, rule's
# cap). Since BASELINE_PARAMS' accel is 1.0, any cap >= 1.0 clamps to
# exactly 1.0 (no effect) -- a Rule can only ever RESTRICT accel below the
# baseline, never raise it. So the only physically-reachable-by-a-real-
# Rule accel range is [CARRY_ACCEL_MIN_MS2, 1.0]; values above 1.0 would
# measure the sim's raw sensitivity to literally commanding a faster carry
# phase, which is informative on its own but is NOT what any Rule's accel
# parameter can actually produce. FRONTIER_ACCELS therefore only sweeps
# within [0.5, 1.0] (an earlier version of this grid swept 0.5-6.0 raw,
# uncapped -- a real bug, since it reported different numbers for
# equivalent-after-clamping accel caps 1.0/2.0/3.0/6.0; fixed by
# _frontier_row itself clamping before running the batch, as a second,
# belt-and-suspenders safeguard).
FRONTIER_ACCELS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _rule(rule_id, grip, accel, rationale):
    return Rule(
        rule_id=rule_id, object_class=TARGET_CLASS,
        grip_force_min_n=grip, carry_accel_max_ms2=accel,
        proposed_by="dev_check", created_at=dt.datetime.now(dt.timezone.utc),
        rationale=rationale, source_report_ids=["dev_check"],
    )


def _reference_rules():
    rules = []
    if os.path.exists(ACCEPTED_RULE_JSON):
        with open(ACCEPTED_RULE_JSON) as f:
            data = json.load(f)
        rule = Rule.model_validate(data["rule"])
        if check1_mod._class_value(rule.object_class) == TARGET_CLASS:
            rules.append(("accepted_rule.json", rule))
        else:
            print(
                f"NOTE: {ACCEPTED_RULE_JSON}'s rule targets "
                f"{check1_mod._class_value(rule.object_class)!r}, not {TARGET_CLASS!r} -- skipping it.",
                file=sys.stderr,
            )
    else:
        print(f"NOTE: {ACCEPTED_RULE_JSON} does not exist -- skipping that reference rule.", file=sys.stderr)

    rules.append(("naive", _rule("r_dev_naive", NAIVE_GRIP, NAIVE_ACCEL, "naive 0.9x-crush-limit-style reference")))
    rules.append(("ceiling", _rule("r_dev_ceiling", CEILING_GRIP, CEILING_ACCEL, "right-at-the-ceiling reference")))
    return rules


def _grid_rules():
    return [
        (f"grid_{grip:.1f}_{accel:.1f}", _rule(f"r_dev_grid_{grip:.2f}_{accel:.2f}", grip, accel, "grid search candidate"))
        for grip in GRID_GRIPS
        for accel in GRID_ACCELS
    ]


def _row(label, rule, before_batch):
    outcome = check1_mod.check1(rule, DEV_SEEDS, precomputed_before=before_batch)
    gate_reasons = [r for r in outcome.reasons if not r.startswith("intervention cost")]
    return {
        "label": label,
        "grip": rule.grip_force_min_n,
        "accel": rule.carry_accel_max_ms2,
        "drop_rate": outcome.drop_rate_after,
        "ci95": outcome.drop_rate_after_ci95,
        "crush_count": outcome.crush_count_after,
        "slowdown_proxy": outcome.slowdown_proxy,
        "passed": outcome.passed,
        "reasons": gate_reasons,
    }


def _frontier_row(grip, accel, seeds):
    """A single (grip, accel) point's achievable frontier numbers -- raw
    sim measurement via run_batch, NOT check1() (no validate()/other-class
    regression check -- this is a research sweep over raw params, not a
    rule being gated). drop and crush are mutually exclusive per-trial
    outcomes (sim/run_trial.py's TrialResult.outcome is exactly one of
    "ok"/"drop"/"crush"), so combined failure = n_drop + n_crush.

    `accel` is clamped to the baseline accel (rules/apply.py's own
    effective_settings: after = min(current, rule's cap)) before it's
    actually run -- see FRONTIER_ACCELS' comment for why: it's a Rule's
    carry_accel_max_ms2, not a literal commanded accel.
    """
    nominal_accel = check1_mod.BASELINE_PARAMS["carry_accel_ms2"]
    effective_accel = min(nominal_accel, accel)
    batch = run_batch(TARGET_CLASS, {"grip_force_n": grip, "carry_accel_ms2": effective_accel}, seeds, parallel=True)
    n = batch.n_trials
    n_fail = batch.n_drop + batch.n_crush
    duration_s = phase_duration(CARRY_DISTANCE_M, effective_accel)
    slowdown_proxy = duration_s / phase_duration(CARRY_DISTANCE_M, nominal_accel)
    return {
        "grip": grip,
        "accel": accel,
        "effective_accel": effective_accel,
        "n_trials": n,
        "drop_rate": batch.drop_rate,
        "drop_ci95": llm_stats.wilson(batch.n_drop, n),
        "crush_rate": round(batch.n_crush / n, 4) if n else 0.0,
        "crush_ci95": llm_stats.wilson(batch.n_crush, n),
        "combined_rate": round(n_fail / n, 4) if n else 0.0,
        "combined_ci95": llm_stats.wilson(n_fail, n),
        "mean_duration_s": round(duration_s, 4),
        "slowdown_proxy": round(slowdown_proxy, 4),
    }


def _pareto_front(rows, minimize=("combined_rate", "mean_duration_s")):
    """Points not dominated by any other point on ALL of `minimize`'s keys
    (dominated: some other row is <= on every key and < on at least one).
    """
    front = []
    for r in rows:
        dominated = any(
            all(other[k] <= r[k] for k in minimize) and any(other[k] < r[k] for k in minimize)
            for other in rows if other is not r
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: (r["combined_rate"], r["mean_duration_s"]))


def _print_frontier_table(rows):
    header = (
        f"{'grip':>6}{'accel':>7}{'drop':>9}{'drop CI95':>18}{'crush':>9}{'crush CI95':>18}"
        f"{'combined':>10}{'comb CI95':>18}{'duration_s':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        dci, cci, kci = r["drop_ci95"], r["crush_ci95"], r["combined_ci95"]
        print(
            f"{r['grip']:>6.1f}{r['accel']:>7.1f}{r['drop_rate']:>9.3f}"
            f"  [{dci[0]:.3f},{dci[1]:.3f}]{r['crush_rate']:>9.3f}  [{cci[0]:.3f},{cci[1]:.3f}]"
            f"{r['combined_rate']:>10.3f}  [{kci[0]:.3f},{kci[1]:.3f}]{r['mean_duration_s']:>12.3f}"
        )


def run_frontier(grips=FRONTIER_GRIPS, accels=FRONTIER_ACCELS, seeds=DEV_SEEDS):
    """A2: the achievable drop/crush/combined-failure frontier for a FIXED
    (grip, accel) rule, independent of any gate's thresholds. Returns the
    full row list (also used to print the table/Pareto front from main()
    and reused by callers that want the raw numbers, e.g. a future
    conditional-rule comparison)."""
    return [_frontier_row(grip, accel, seeds) for grip in grips for accel in accels]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-grid", action="store_true", help="only run the 3 reference rules, skip the grid")
    ap.add_argument("--frontier", action="store_true",
                     help="instead of the v2 pass/fail probe, compute the raw achievable "
                          "drop/crush/combined-failure frontier over a finer grid (A2)")
    args = ap.parse_args()

    if args.frontier:
        print(
            f"Fixed-rule frontier, {TARGET_CLASS}, DEV seeds {DEV_SEEDS.start}-{DEV_SEEDS.stop - 1} "
            f"ONLY ({len(DEV_SEEDS)} seeds), {len(FRONTIER_GRIPS)} grips x {len(FRONTIER_ACCELS)} accels "
            f"= {len(FRONTIER_GRIPS) * len(FRONTIER_ACCELS)} points\n"
        )
        rows = run_frontier()
        _print_frontier_table(rows)
        front = _pareto_front(rows)
        print(f"\nPareto front (minimizing combined failure rate, then mean duration), {len(front)} point(s):")
        _print_frontier_table(front)
        best = min(rows, key=lambda r: r["combined_rate"])
        print(
            f"\nBEST ACHIEVABLE combined failure for any fixed rule: {best['combined_rate']:.3f} "
            f"(Wilson 95% CI {best['combined_ci95']}) at grip={best['grip']}, accel={best['accel']} "
            f"(drop={best['drop_rate']:.3f}, crush={best['crush_rate']:.3f})."
        )
        slowdowns = sorted({r["slowdown_proxy"] for r in rows})
        print(
            f"\nSlowdown_proxy values achieved across this grid: {slowdowns} "
            f"(max possible given CARRY_ACCEL_MIN_MS2=0.5 is sqrt(2)~=1.41421)."
        )
        return

    print(
        f"Check 1 v{check1_mod.CHECK1_VERSION} probe, {TARGET_CLASS}, DEV seeds "
        f"{DEV_SEEDS.start}-{DEV_SEEDS.stop - 1} ONLY ({len(DEV_SEEDS)} seeds)\n"
    )

    before_batch = run_batch(TARGET_CLASS, check1_mod.BASELINE_PARAMS, DEV_SEEDS, parallel=True)
    print(f"baseline (no rule, {check1_mod.BASELINE_PARAMS}): {before_batch.n_drop}/{before_batch.n_trials} drop\n")

    rows = [_row(label, rule, before_batch) for label, rule in _reference_rules()]
    if not args.skip_grid:
        rows += [_row(label, rule, before_batch) for label, rule in _grid_rules()]

    header = f"{'label':<16}{'grip':>7}{'accel':>7}{'drop_rate':>11}{'CI95 upper':>12}{'crush':>7}{'slowdown':>10}{'PASS':>7}"
    print(header)
    print("-" * len(header))
    n_passed = 0
    for row in rows:
        print(
            f"{row['label']:<16}{row['grip']:>7.2f}{row['accel']:>7.2f}{row['drop_rate']:>11.3f}"
            f"{row['ci95'][1]:>12.3f}{row['crush_count']:>7}{row['slowdown_proxy']:>10.3f}{str(row['passed']):>7}"
        )
        if row["passed"]:
            n_passed += 1
        else:
            print(f"    FAIL: {'; '.join(row['reasons'])}")

    print()
    if n_passed == 0:
        print(
            f"RESULT: none passes v{check1_mod.CHECK1_VERSION} -- all {len(rows)} rule(s) tried "
            f"(3 reference + {len(rows) - 3 if not args.skip_grid else 0} grid) for {TARGET_CLASS} "
            f"fail Check 1 v{check1_mod.CHECK1_VERSION} on seeds {DEV_SEEDS.start}-{DEV_SEEDS.stop - 1}."
        )
    else:
        print(f"RESULT: {n_passed}/{len(rows)} rule(s) passed v{check1_mod.CHECK1_VERSION}.")


if __name__ == "__main__":
    main()
