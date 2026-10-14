"""Held-out measurement (CONTEXT.md section 4 steps 6-7 / section 8).

For Cell B (same owner as A) and Cell C (different owner), measures drop
rate with vs without the accepted rule on held-out seeds never used while
developing the rule, gated by the REAL rules.policy.can_share() decision
and the real audit log -- a denied Cell C is evaluated unchanged (no
rule effect), matching what actually happens in the real pipeline. Also
reports three baselines on the same seeds for context: no rule, a naive
"grip 0.9x crush limit with minimum accel" heuristic, and a grid-searched
"mildest passing rule" found via check1() alone (no LLM).

Refuses to run without --held-out (CONTEXT.md: "held-out seeds ... used
only for the final measurement"). Every actual run is appended to
pipeline/runs/heldout_access.log, written BEFORE any held-out seed is
touched, so an aborted run still leaves a record that an attempt was
made.
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
import stats as llm_stats  # llm/stats.py

from rules import lifecycle
from rules.apply import CellState, RuleRejected, apply
from rules.limits import (
    CARRY_ACCEL_MAX_MS2,
    CARRY_ACCEL_MIN_MS2,
    CRUSH_LIMIT_N,
    GRIP_FORCE_MAX_N,
    GRIP_FORCE_MIN_N,
)
from rules.policy import Owner, can_share
from rules.schema import Rule

RUNS_DIR = os.path.join(PIPELINE_DIR, "runs")
ACCESS_LOG_PATH = os.path.join(RUNS_DIR, "heldout_access.log")

CELL_B_SEEDS_DEFAULT = range(1000, 1100)
CELL_C_SEEDS_DEFAULT = range(1100, 1200)

# Permission model (CONTEXT.md section 7): A and B share an owner, C has a
# different one. Not configurable beyond the sharing_enabled flag below --
# that's what section 7's permission model fixes, not something this
# script should invent variations of.
OWNER_A_B = "owner1"
OWNER_C = "owner2"

# Grid search ("mildest passing rule", no LLM) on TRAIN seeds only -- never
# held-out, matching CONTEXT.md's "never use [held-out seeds] while
# iterating on a rule". Resolution/seed count are kept small so this
# script stays fast; this is a comparison baseline, not Check 1 itself.
GRID_TRAIN_SEEDS = range(15)
GRID_RESOLUTION = 6

# "Ceiling" baseline: grip just under the class's crush limit (as close
# to the datasheet edge as this script will go without actually risking
# a crush), minimum legal accel. CEILING_GRIP_MARGIN_N is this script's
# own choice of "just under" -- not a measured or agreed value.
CEILING_GRIP_MARGIN_N = 0.1

STUB_LABEL = "STUB RULE - not a model result"


def _class_value(object_class) -> str:
    return object_class.value if hasattr(object_class, "value") else object_class


def _grid_search_mildest_passing_rule(target_class, source_report_ids):
    """Grid-search (grip, accel) with check1() alone, no LLM. Returns the
    PASSING candidate with the smallest grip_force_min_n, tie-broken by
    the largest carry_accel_max_ms2 -- i.e. the least invasive rule that
    still clears Check 1. Returns None if no grid point passes.
    """
    crush_limit = CRUSH_LIMIT_N[target_class]
    grip_hi = min(GRIP_FORCE_MAX_N, crush_limit)
    grips = [
        GRIP_FORCE_MIN_N + i * (grip_hi - GRIP_FORCE_MIN_N) / (GRID_RESOLUTION - 1)
        for i in range(GRID_RESOLUTION)
    ]
    accels = [
        CARRY_ACCEL_MIN_MS2 + i * (CARRY_ACCEL_MAX_MS2 - CARRY_ACCEL_MIN_MS2) / (GRID_RESOLUTION - 1)
        for i in range(GRID_RESOLUTION)
    ]

    before_batch = run_batch(target_class, check1_mod.BASELINE_PARAMS, GRID_TRAIN_SEEDS, parallel=True)

    best = None
    now = dt.datetime.now(dt.timezone.utc)
    for grip in grips:
        for accel in accels:
            candidate = Rule(
                rule_id=f"grid_{grip:.2f}_{accel:.2f}",
                object_class=target_class,
                grip_force_min_n=round(grip, 3),
                carry_accel_max_ms2=round(accel, 3),
                proposed_by="grid-search",
                created_at=now,
                rationale="grid search candidate (no LLM)",
                source_report_ids=list(source_report_ids) or ["grid-search"],
            )
            outcome = check1_mod.check1(candidate, GRID_TRAIN_SEEDS, precomputed_before=before_batch)
            if not outcome.passed:
                continue
            key = (candidate.grip_force_min_n, -candidate.carry_accel_max_ms2)
            if best is None or key < best[0]:
                best = (key, candidate)
    return best[1] if best else None


def _ensure_approved(audit_path, rule, actor):
    state = lifecycle.current_state(audit_path, rule.rule_id)
    if state is None:
        raise RuntimeError(
            f"rule {rule.rule_id} has no PROPOSED event in {audit_path} -- "
            "run pipeline/loop.py first (or pass a matching --audit-path)"
        )
    if state == lifecycle.State.APPROVED:
        return
    if state == lifecycle.State.CHECK1_PASSED:
        print(
            f"NOTE: simulating Check 2 (human review) approval of {rule.rule_id} for this "
            "non-interactive demo script -- a real deployment requires a human reviewer "
            "(rules/review_cli.py), not this script."
        )
        lifecycle.transition(
            audit_path, rule.rule_id, lifecycle.State.APPROVED, actor,
            {"note": "auto-approved by pipeline/evaluate_heldout.py for a non-interactive demo run"},
        )
        return
    raise RuntimeError(f"rule {rule.rule_id} is in state {state.value}, cannot evaluate held-out performance")


def _paired_counts(before_results, after_results):
    before_by_seed = {r.seed: r.outcome for r in before_results}
    after_by_seed = {r.seed: r.outcome for r in after_results}
    fixed = sorted(s for s, o in before_by_seed.items() if o != "ok" and after_by_seed.get(s) == "ok")
    newly_broken = sorted(s for s, o in before_by_seed.items() if o == "ok" and after_by_seed.get(s) != "ok")
    return fixed, newly_broken


def _measure(target_class, cell_label, seeds, conditions):
    """conditions: {name: params or None}. None means "same as baseline,
    not evaluated" (used for a denied share). Returns {name: row dict}."""
    rows = {}
    baseline_batch = run_batch(target_class, conditions["none"], seeds, parallel=True)
    rows["none"] = _row(baseline_batch, baseline_batch)
    for name, params in conditions.items():
        if name == "none":
            continue
        batch = run_batch(target_class, params, seeds, parallel=True) if params is not None else baseline_batch
        rows[name] = _row(baseline_batch, batch)
    return rows


def _row(baseline_batch, batch):
    fixed, newly_broken = _paired_counts(baseline_batch.results, batch.results)
    return {
        "n_trials": batch.n_trials,
        "n_drop": batch.n_drop,
        "n_crush": batch.n_crush,
        "drop_rate": batch.drop_rate,
        "drop_rate_ci95": list(llm_stats.wilson(batch.n_drop, batch.n_trials)),
        "seeds_fixed_vs_baseline": fixed,
        "seeds_newly_broken_vs_baseline": newly_broken,
    }


def _markdown_table(cell_label, rows, controller_by_condition):
    lines = [
        f"### Cell {cell_label}",
        "",
        "| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | fixed | newly broken |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in rows.items():
        params = controller_by_condition.get(name) or {}
        ci = row["drop_rate_ci95"]
        lines.append(
            f"| {name} | {params.get('grip_force_n', '-')} | {params.get('carry_accel_ms2', '-')} | "
            f"{row['n_trials']} | {row['n_drop']} | {row['n_crush']} | {row['drop_rate']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {len(row['seeds_fixed_vs_baseline'])} | {len(row['seeds_newly_broken_vs_baseline'])} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--held-out", action="store_true", help="required: confirms this run is allowed to touch held-out seeds")
    p.add_argument("--rule-json", default=os.path.join(RUNS_DIR, "accepted_rule.json"))
    p.add_argument("--audit-path", default=os.path.join(RUNS_DIR, "loop_demo_audit.jsonl"))
    p.add_argument("--cell-b-start", type=int, default=CELL_B_SEEDS_DEFAULT.start)
    p.add_argument("--cell-b-end", type=int, default=CELL_B_SEEDS_DEFAULT.stop)
    p.add_argument("--cell-c-start", type=int, default=CELL_C_SEEDS_DEFAULT.start)
    p.add_argument("--cell-c-end", type=int, default=CELL_C_SEEDS_DEFAULT.stop)
    p.add_argument("--actor", default="pipeline-evaluate-heldout")
    args = p.parse_args()

    if not args.held_out:
        print(
            "REFUSED: --held-out was not passed. This script only runs against held-out "
            "seeds, which must never be touched while iterating on a rule. Re-run with "
            "--held-out if this is really the final measurement.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(args.rule_json):
        print(
            f"REFUSED: {args.rule_json} does not exist. Run pipeline/loop.py first -- it "
            "writes this file once a candidate reaches CHECK1_PASSED. Pass --rule-json to "
            "point at a different file.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(args.rule_json) as f:
        accepted_data = json.load(f)
    rule_label = accepted_data.get("label")
    rule = Rule.model_validate(accepted_data["rule"])

    stub_banner = rule_label or (STUB_LABEL if accepted_data.get("is_stub") else None)
    if stub_banner:
        print(stub_banner)
    top_notice = (
        "NOTICE: this is a non-interactive demo script. Check 2 (human approval) is "
        "AUTO-APPROVED here for the purpose of this held-out evaluation -- a real "
        "deployment requires an actual human reviewer (rules/review_cli.py), not this "
        "script. The baseline ('none' / 'without the rule') condition in every table "
        f"below is the nominal setting {dict(check1_mod.BASELINE_PARAMS)} "
        "(grip_force_n, carry_accel_ms2) -- see pipeline/check1.py's BASELINE_PARAMS "
        "for how and why it was chosen."
    )
    print(top_notice)
    print()

    _ensure_approved(args.audit_path, rule, args.actor)

    cell_b_seeds = range(args.cell_b_start, args.cell_b_end)
    cell_c_seeds = range(args.cell_c_start, args.cell_c_end)

    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(ACCESS_LOG_PATH, "a") as f:
        f.write(json.dumps({
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "actor": args.actor,
            "rule_id": rule.rule_id,
            "cell_b_seeds": [cell_b_seeds.start, cell_b_seeds.stop],
            "cell_c_seeds": [cell_c_seeds.start, cell_c_seeds.stop],
        }) + "\n")

    target_class = _class_value(rule.object_class)
    cell_a = CellState("A", OWNER_A_B, check1_mod.BASELINE_PARAMS)
    cell_b = CellState("B", OWNER_A_B, check1_mod.BASELINE_PARAMS)
    cell_c = CellState("C", OWNER_C, check1_mod.BASELINE_PARAMS)

    # Cell C is evaluated under BOTH real policy decisions, each a REAL
    # can_share() call against the real audit log (two per-cell SHARED/
    # BLOCKED transitions on the same (rule_id, "C") pair -- both legal,
    # rules/lifecycle.py's _PER_CELL_TRANSITIONS allows BLOCKED<->SHARED
    # either direction). "denied" is the realistic default (Owner's own
    # sharing_enabled default is False); "allowed" shows what the same
    # rule, same seeds, would do if owner1 opted in to sharing.
    owners_denied = {OWNER_A_B: Owner(OWNER_A_B, sharing_enabled=False), OWNER_C: Owner(OWNER_C)}
    owners_allowed = {OWNER_A_B: Owner(OWNER_A_B, sharing_enabled=True), OWNER_C: Owner(OWNER_C)}

    decision_b = can_share(args.audit_path, rule, cell_a, cell_b, owners_denied)
    decision_c_denied = can_share(args.audit_path, rule, cell_a, cell_c, owners_denied)
    decision_c_allowed = can_share(args.audit_path, rule, cell_a, cell_c, owners_allowed)

    try:
        rule_after_params = apply(rule, check1_mod.BASELINE_PARAMS, target_class)
    except RuleRejected as e:
        print(f"REFUSED: apply() rejected the accepted rule: {e}", file=sys.stderr)
        sys.exit(1)

    grid_rule = _grid_search_mildest_passing_rule(target_class, rule.source_report_ids)

    crush_limit = CRUSH_LIMIT_N[target_class]
    naive_params = {
        "grip_force_n": round(0.9 * crush_limit, 3),
        "carry_accel_ms2": CARRY_ACCEL_MIN_MS2,
    }
    ceiling_params = {
        "grip_force_n": round(crush_limit - CEILING_GRIP_MARGIN_N, 3),
        "carry_accel_ms2": CARRY_ACCEL_MIN_MS2,
    }

    conditions = {
        "none": dict(check1_mod.BASELINE_PARAMS),
        "naive_0.9x_crush_min_accel": naive_params,
        "ceiling_near_crush_min_accel": ceiling_params,
        "grid_mildest_passing": (
            apply(grid_rule, check1_mod.BASELINE_PARAMS, target_class) if grid_rule is not None else None
        ),
    }

    results = {}
    controller_by_condition_by_cell = {}
    cell_runs = (
        ("B", cell_b_seeds, decision_b),
        ("C_sharing_denied", cell_c_seeds, decision_c_denied),
        ("C_sharing_allowed", cell_c_seeds, decision_c_allowed),
    )
    for cell_label, seeds, decision in cell_runs:
        cell_conditions = dict(conditions)
        cell_conditions["accepted_rule"] = rule_after_params if decision.allowed else None
        cell_controller_by_condition = dict(conditions)
        cell_controller_by_condition["accepted_rule"] = rule_after_params if decision.allowed else conditions["none"]
        controller_by_condition_by_cell[cell_label] = cell_controller_by_condition
        rows = _measure(target_class, cell_label, seeds, cell_conditions)
        results[cell_label] = {
            "seeds": [seeds.start, seeds.stop],
            "can_share_decision": {"allowed": decision.allowed, "reason": decision.reason},
            "rows": rows,
        }

    report = {
        "stub_label": stub_banner,
        "notice": top_notice,
        "baseline_params": dict(check1_mod.BASELINE_PARAMS),
        "rule": rule.model_dump(mode="json"),
        "target_class": target_class,
        "grid_mildest_passing_rule": grid_rule.model_dump(mode="json") if grid_rule is not None else None,
        "naive_baseline_params": naive_params,
        "ceiling_params": ceiling_params,
        "results": results,
    }

    os.makedirs(RUNS_DIR, exist_ok=True)
    json_path = os.path.join(RUNS_DIR, f"heldout_eval_{rule.rule_id}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [f"# Held-out evaluation: {rule.rule_id} ({target_class})", ""]
    if stub_banner:
        md_lines += [stub_banner, ""]
    md_lines += [top_notice, ""]
    for cell_label, _seeds, _decision in cell_runs:
        md_lines.append(
            f"Cell {cell_label} can_share: allowed={results[cell_label]['can_share_decision']['allowed']} "
            f"({results[cell_label]['can_share_decision']['reason']})"
        )
        md_lines.append("")
        md_lines.append(_markdown_table(cell_label, results[cell_label]["rows"], controller_by_condition_by_cell[cell_label]))
        md_lines.append("")
    md_path = os.path.join(RUNS_DIR, f"heldout_eval_{rule.rule_id}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print("\n".join(md_lines))
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
