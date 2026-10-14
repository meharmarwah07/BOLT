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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-grid", action="store_true", help="only run the 3 reference rules, skip the grid")
    args = ap.parse_args()

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
