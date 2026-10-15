"""Held-out measurement (CONTEXT.md section 4 steps 6-7 / section 8).

For Cell B (same owner as A) and Cell C (different owner), measures drop
rate with vs without the accepted rule on held-out seeds never used while
developing the rule, gated by the REAL rules.policy.can_share() decision
and the real audit log -- a denied Cell C is evaluated unchanged (no
rule effect), matching what actually happens in the real pipeline. Also
reports three baselines on the same seeds for context: no rule, a naive
"grip 0.9x crush limit with minimum accel" heuristic, and a grid-searched
"mildest passing rule" found via check1() alone (no LLM) -- that grid
search now uses check1()'s default gate (CHECK1_VERSION); if no grid point
clears it, the JSON/.md report say so explicitly ("none passes vN")
instead of silently reporting the unchanged baseline as if it were a rule.

Per-seed outcome/crush/duration (not just aggregates) is stored for every
condition in the JSON output, and the .md output adds a paired exact
McNemar table (accepted_rule vs each baseline, task success) per cell,
plus a task-success-rate and mean-duration column on the main table.

Refuses to run without --held-out or --dev (CONTEXT.md: "held-out seeds
... used only for the final measurement"). A --held-out run is appended to
the access log (--access-log-path, default pipeline/runs/heldout_access.log),
written BEFORE any held-out seed is touched, so an aborted run still
leaves a record that an attempt was made.

A --held-out run also REFUSES outright (no override) if its seeds overlap
an already-spent range (docs/FINAL_PROTOCOL.md: 1000-1199, 2000-2199,
5000-5199), and refuses a SECOND run touching the final held-out range
3000-3199 once the access log already shows a prior one, unless
--override-reason is given (itself recorded in the log). See
SPENT_SEED_RANGES / FINAL_HELDOUT_SEED_RANGE below.

--dev is a separate, safe-to-iterate path: it requires every seed in
DEV_SEEDS_RANGE (5000-5199), never writes heldout_access.log, and stamps
"DEV RUN" in its own output files (heldout_eval_<rule_id>_DEV.{json,md},
never overwriting a real --held-out run's output). --held-out and --dev
are mutually exclusive; the real --held-out path is otherwise unchanged.
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

# --dev mode (never touches held-out seeds or heldout_access.log): dev/
# iteration seeds for this project are reserved as 5000-5199. --dev
# refuses outside that range; see main()'s validation.
DEV_SEEDS_RANGE = range(5000, 5200)
DEV_CELL_B_SEEDS_DEFAULT = range(5000, 5100)
DEV_CELL_C_SEEDS_DEFAULT = range(5100, 5200)

# docs/FINAL_PROTOCOL.md: "Spent seeds: 1000-1199, 2000-2199, 5000-5199" --
# already used (the pre-revision baseline measurement; 5000-5199 is also
# DEV_SEEDS_RANGE, legitimately reused for --dev iteration, but must never
# be reused as if it were a fresh --held-out measurement). A --held-out
# run touching any of these is refused outright, with no override -- this
# is a hard guard against contaminating a held-out comparison with seeds
# whose outcome is already known, not a "used once" budget (see
# FINAL_HELDOUT_SEED_RANGE below for that).
SPENT_SEED_RANGES = (range(1000, 1200), range(2000, 2200), range(5000, 5200))

# docs/FINAL_PROTOCOL.md: "Seeds 3000-3199 ... are used once. No rule,
# prompt, gate or selection change after the first look." Enforced below
# as a single-use guard keyed off heldout_access.log: a --held-out run
# touching this range is refused once the log already shows a prior run
# that touched it, unless --override-reason is given (itself recorded in
# the log for audit, e.g. the first attempt was an infrastructure error).
FINAL_HELDOUT_SEED_RANGE = range(3000, 3200)

# Permission model (CONTEXT.md section 7): A and B share an owner, C has a
# different one. Not configurable beyond the sharing_enabled flag below --
# that's what section 7's permission model fixes, not something this
# script should invent variations of.
OWNER_A_B = "owner1"
OWNER_C = "owner2"

# Grid search ("mildest passing rule", no LLM) on TRAIN seeds only -- never
# held-out, matching CONTEXT.md's "never use [held-out seeds] while
# iterating on a rule". This is a comparison baseline, not Check 1 itself.
#
# Was range(15): too few for check1()'s now-default v2 gate to ever pass at
# all, regardless of the candidate -- v2's MAX_AFTER_DROP_RATE_CI95_UPPER
# (0.10) Wilson upper bound can't clear 0.10 at n=15 even at zero observed
# drops (the Wilson upper bound at k=0 only drops below 0.10 once n>=35).
# At range(15), "no grid point passes v2" would reflect a statistics
# floor, not a real finding about the grid. 50 clears that floor with a
# margin (Wilson upper at k=0,n=50 is ~0.071) while keeping the grid's
# GRID_RESOLUTION**2 batches reasonably fast.
GRID_TRAIN_SEEDS = range(50)
GRID_RESOLUTION = 6

# "Ceiling" baseline: grip just under the class's crush limit (as close
# to the datasheet edge as this script will go without actually risking
# a crush), minimum legal accel. CEILING_GRIP_MARGIN_N is this script's
# own choice of "just under" -- not a measured or agreed value.
CEILING_GRIP_MARGIN_N = 0.1

STUB_LABEL = "STUB RULE - not a model result"


def _class_value(object_class) -> str:
    return object_class.value if hasattr(object_class, "value") else object_class


def _overlaps(start, end, target_range) -> bool:
    """True if the half-open seed range [start, end) overlaps `target_range`."""
    return start < target_range.stop and end > target_range.start


def check_spent_seed_guard(cell_ranges):
    """cell_ranges: [(name, start, end), ...]. Returns a REFUSED message
    string if any range overlaps an already-spent range (no override --
    see SPENT_SEED_RANGES), else None. A pure function of its arguments --
    no file I/O, no sim call -- so it is directly unit-testable without
    going through main() or touching any real seed."""
    for name, start, end in cell_ranges:
        for spent in SPENT_SEED_RANGES:
            if _overlaps(start, end, spent):
                return (
                    f"REFUSED: --{name} [{start}, {end}) overlaps an already-spent seed range "
                    f"[{spent.start}, {spent.stop}) (docs/FINAL_PROTOCOL.md: spent seeds are "
                    "1000-1199, 2000-2199, 5000-5199) -- a --held-out run must use a fresh "
                    "range (e.g. 3000-3199) instead."
                )
    return None


def check_final_range_single_use_guard(cell_ranges, access_log_path, override_reason, final_range=None):
    """cell_ranges: [(name, start, end), ...]. Returns a REFUSED message
    string if any range overlaps `final_range` (default
    FINAL_HELDOUT_SEED_RANGE, i.e. 3000-3199) AND `access_log_path` already
    records a prior run touching it AND no override_reason was given,
    else None (including when access_log_path has no prior record at all
    -- "the first run is allowed").

    The only file I/O here is a read of access_log_path -- no sim call, no
    write -- so a caller can safely pass a throwaway final_range (e.g.
    range(8000, 8010)) and a temp access_log_path to test this fully
    in-process, including against a fabricated "prior run" entry, without
    ever touching the real pipeline/runs/heldout_access.log or spending
    any of the real 3000-3199 budget.
    """
    if final_range is None:
        final_range = FINAL_HELDOUT_SEED_RANGE
    touches_final_range = any(_overlaps(start, end, final_range) for _, start, end in cell_ranges)
    if not touches_final_range or override_reason is not None:
        return None
    prior = _prior_runs_touching_range(access_log_path, final_range)
    if not prior:
        return None
    return (
        f"REFUSED: seeds {final_range.start}-{final_range.stop - 1} are the final held-out range "
        "(docs/FINAL_PROTOCOL.md: 'used once. No rule, prompt, gate or selection change after the "
        f"first look.'), and {access_log_path} already records {len(prior)} prior run(s) touching "
        f"it (most recent: {prior[-1].get('ts')}, rule_id={prior[-1].get('rule_id')!r}). Pass "
        "--override-reason \"...\" to proceed anyway (recorded in the log for audit)."
    )


def _prior_runs_touching_range(access_log_path, target_range):
    """Every prior --held-out entry in `access_log_path` (if it exists)
    whose cell_b_seeds or cell_c_seeds overlaps `target_range`. Read-only:
    never writes, never touches the real log unless the caller's own
    access_log_path points at it."""
    if not os.path.exists(access_log_path):
        return []
    hits = []
    with open(access_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for key in ("cell_b_seeds", "cell_c_seeds"):
                seeds = entry.get(key)
                if seeds and _overlaps(seeds[0], seeds[1], target_range):
                    hits.append(entry)
                    break
    return hits


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
    rows["none"] = _row(baseline_batch, baseline_batch, conditions["none"])
    for name, params in conditions.items():
        if name == "none":
            continue
        batch = run_batch(target_class, params, seeds, parallel=True) if params is not None else baseline_batch
        rows[name] = _row(baseline_batch, batch, params if params is not None else conditions["none"])
    return rows


def _row(baseline_batch, batch, controller_params):
    fixed, newly_broken = _paired_counts(baseline_batch.results, batch.results)
    # duration_s is a function of carry_accel_ms2 alone (sim/cell_model.py's
    # phase_duration), so it is identical for every seed under one
    # condition -- stored per seed anyway (not just as one aggregate
    # number) so downstream per-seed analysis doesn't have to re-derive it
    # from controller_params itself.
    duration_s = phase_duration(CARRY_DISTANCE_M, controller_params["carry_accel_ms2"])
    n_ok = sum(1 for r in batch.results if r.outcome == "ok")
    return {
        "n_trials": batch.n_trials,
        "n_drop": batch.n_drop,
        "n_crush": batch.n_crush,
        "drop_rate": batch.drop_rate,
        "drop_rate_ci95": list(llm_stats.wilson(batch.n_drop, batch.n_trials)),
        "task_success_rate": round(n_ok / batch.n_trials, 4) if batch.n_trials else 0.0,
        "mean_duration_s": round(duration_s, 4),
        "seeds_fixed_vs_baseline": fixed,
        "seeds_newly_broken_vs_baseline": newly_broken,
        "outcomes_by_seed": [
            {"seed": r.seed, "outcome": r.outcome, "duration_s": round(duration_s, 4)} for r in batch.results
        ],
    }


def _markdown_table(cell_label, rows, controller_by_condition):
    lines = [
        f"### Cell {cell_label}",
        "",
        "| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | "
        "task success | mean duration (s) | fixed | newly broken |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in rows.items():
        params = controller_by_condition.get(name) or {}
        ci = row["drop_rate_ci95"]
        lines.append(
            f"| {name} | {params.get('grip_force_n', '-')} | {params.get('carry_accel_ms2', '-')} | "
            f"{row['n_trials']} | {row['n_drop']} | {row['n_crush']} | {row['drop_rate']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {row['task_success_rate']:.3f} | {row['mean_duration_s']:.3f} | "
            f"{len(row['seeds_fixed_vs_baseline'])} | {len(row['seeds_newly_broken_vs_baseline'])} |"
        )
    return "\n".join(lines)


def _mcnemar_table(cell_label, rows, accepted_name="accepted_rule"):
    """Paired exact McNemar (two-sided), task-success (outcome == "ok") on
    each seed, accepted_rule vs every other condition measured on this
    cell's own seeds. b/c are the discordant counts stats.mcnemar_exact
    needs; see its docstring. Returns "" if accepted_rule wasn't measured
    on this cell (e.g. this cell's rows only cover the no-rule baseline).
    """
    if accepted_name not in rows:
        return ""
    accepted_success = {o["seed"]: o["outcome"] == "ok" for o in rows[accepted_name]["outcomes_by_seed"]}
    lines = [
        f"### Cell {cell_label}: paired exact McNemar ({accepted_name} vs each baseline, task success)",
        "",
        "| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |",
        "|---|---|---|---|",
    ]
    for name, row in rows.items():
        if name == accepted_name:
            continue
        other_success = {o["seed"]: o["outcome"] == "ok" for o in row["outcomes_by_seed"]}
        common_seeds = sorted(set(accepted_success) & set(other_success))
        b = sum(1 for s in common_seeds if accepted_success[s] and not other_success[s])
        c = sum(1 for s in common_seeds if not accepted_success[s] and other_success[s])
        p = llm_stats.mcnemar_exact(b, c)
        lines.append(f"| {name} | {b} | {c} | {p:.4f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--held-out", action="store_true", help="required (unless --dev): confirms this run is allowed to touch held-out seeds")
    p.add_argument("--dev", action="store_true",
                    help="dev mode: seeds must be in DEV_SEEDS_RANGE (5000-5199), never touches "
                         "heldout_access.log, and stamps DEV RUN in the output. Mutually exclusive with --held-out.")
    p.add_argument("--rule-json", default=os.path.join(RUNS_DIR, "accepted_rule.json"))
    p.add_argument("--audit-path", default=os.path.join(RUNS_DIR, "loop_demo_audit.jsonl"))
    p.add_argument("--cell-b-start", type=int, default=None)
    p.add_argument("--cell-b-end", type=int, default=None)
    p.add_argument("--cell-c-start", type=int, default=None)
    p.add_argument("--cell-c-end", type=int, default=None)
    p.add_argument("--actor", default="pipeline-evaluate-heldout")
    p.add_argument("--access-log-path", default=ACCESS_LOG_PATH,
                    help="path to the held-out access log (default: pipeline/runs/heldout_access.log). "
                         "Tests should point this at a temp path so the real log is never touched.")
    p.add_argument("--override-reason", default=None,
                    help="required to run --held-out a second time against seeds that already "
                         "touch the final held-out range 3000-3199 (docs/FINAL_PROTOCOL.md: "
                         "'used once'); recorded in the access log for audit.")
    args = p.parse_args()

    if args.held_out and args.dev:
        print("REFUSED: --held-out and --dev are mutually exclusive -- choose one.", file=sys.stderr)
        sys.exit(1)

    if not args.held_out and not args.dev:
        print(
            "REFUSED: neither --held-out nor --dev was passed. This script only runs against "
            "held-out seeds (--held-out, the final measurement) or DEV seeds 5000-5199 "
            "(--dev, safe to iterate with). Re-run with one of the two.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dev:
        cell_b_start = args.cell_b_start if args.cell_b_start is not None else DEV_CELL_B_SEEDS_DEFAULT.start
        cell_b_end = args.cell_b_end if args.cell_b_end is not None else DEV_CELL_B_SEEDS_DEFAULT.stop
        cell_c_start = args.cell_c_start if args.cell_c_start is not None else DEV_CELL_C_SEEDS_DEFAULT.start
        cell_c_end = args.cell_c_end if args.cell_c_end is not None else DEV_CELL_C_SEEDS_DEFAULT.stop
        for name, start, end in (("cell-b", cell_b_start, cell_b_end), ("cell-c", cell_c_start, cell_c_end)):
            if not (DEV_SEEDS_RANGE.start <= start and end <= DEV_SEEDS_RANGE.stop):
                print(
                    f"REFUSED: --dev requires every seed in DEV_SEEDS_RANGE "
                    f"[{DEV_SEEDS_RANGE.start}, {DEV_SEEDS_RANGE.stop}) -- got --{name} "
                    f"[{start}, {end}).",
                    file=sys.stderr,
                )
                sys.exit(1)
    else:
        cell_b_start = args.cell_b_start if args.cell_b_start is not None else CELL_B_SEEDS_DEFAULT.start
        cell_b_end = args.cell_b_end if args.cell_b_end is not None else CELL_B_SEEDS_DEFAULT.stop
        cell_c_start = args.cell_c_start if args.cell_c_start is not None else CELL_C_SEEDS_DEFAULT.start
        cell_c_end = args.cell_c_end if args.cell_c_end is not None else CELL_C_SEEDS_DEFAULT.stop

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

    if not args.dev:
        cell_ranges = (("cell-b", cell_b_start, cell_b_end), ("cell-c", cell_c_start, cell_c_end))

        refusal = check_spent_seed_guard(cell_ranges)
        if refusal is not None:
            print(refusal, file=sys.stderr)
            sys.exit(1)

        refusal = check_final_range_single_use_guard(cell_ranges, args.access_log_path, args.override_reason)
        if refusal is not None:
            print(refusal, file=sys.stderr)
            sys.exit(1)

    dev_banner = "DEV RUN -- seeds 5000-5199, not a held-out measurement" if args.dev else None
    if dev_banner:
        print(dev_banner)
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

    cell_b_seeds = range(cell_b_start, cell_b_end)
    cell_c_seeds = range(cell_c_start, cell_c_end)

    os.makedirs(RUNS_DIR, exist_ok=True)
    if not args.dev:
        os.makedirs(os.path.dirname(os.path.abspath(args.access_log_path)) or ".", exist_ok=True)
        with open(args.access_log_path, "a") as f:
            f.write(json.dumps({
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "actor": args.actor,
                "rule_id": rule.rule_id,
                "cell_b_seeds": [cell_b_seeds.start, cell_b_seeds.stop],
                "cell_c_seeds": [cell_c_seeds.start, cell_c_seeds.stop],
                "override_reason": args.override_reason,
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
    grid_none_passes_note = None
    if grid_rule is None:
        grid_none_passes_note = (
            f"none passes v{check1_mod.CHECK1_VERSION} -- no (grip, accel) grid point for "
            f"{target_class!r} cleared Check 1 v{check1_mod.CHECK1_VERSION} on GRID_TRAIN_SEEDS "
            f"{GRID_TRAIN_SEEDS.start}-{GRID_TRAIN_SEEDS.stop - 1} (resolution {GRID_RESOLUTION}x"
            f"{GRID_RESOLUTION}). The 'grid_mildest_passing' row below is the UNCHANGED baseline "
            "(no grid rule applied, not a real passing rule) -- see pipeline/dev_check.py for a "
            "wider, dev-seed grid search of the same question."
        )
        print(f"NOTE: {grid_none_passes_note}", file=sys.stderr)

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
        "dev_run": args.dev,
        "stub_label": stub_banner,
        "notice": top_notice,
        "baseline_params": dict(check1_mod.BASELINE_PARAMS),
        "rule": rule.model_dump(mode="json"),
        "target_class": target_class,
        "grid_mildest_passing_rule": grid_rule.model_dump(mode="json") if grid_rule is not None else None,
        "grid_mildest_passing_note": grid_none_passes_note,
        "naive_baseline_params": naive_params,
        "ceiling_params": ceiling_params,
        "results": results,
    }

    os.makedirs(RUNS_DIR, exist_ok=True)
    file_suffix = "_DEV" if args.dev else ""
    json_path = os.path.join(RUNS_DIR, f"heldout_eval_{rule.rule_id}{file_suffix}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [f"# Held-out evaluation: {rule.rule_id} ({target_class}){' [DEV RUN]' if args.dev else ''}", ""]
    if dev_banner:
        md_lines += [dev_banner, ""]
    if stub_banner:
        md_lines += [stub_banner, ""]
    md_lines += [top_notice, ""]
    if grid_none_passes_note:
        md_lines += [f"NOTE: {grid_none_passes_note}", ""]
    for cell_label, _seeds, _decision in cell_runs:
        md_lines.append(
            f"Cell {cell_label} can_share: allowed={results[cell_label]['can_share_decision']['allowed']} "
            f"({results[cell_label]['can_share_decision']['reason']})"
        )
        md_lines.append("")
        md_lines.append(_markdown_table(cell_label, results[cell_label]["rows"], controller_by_condition_by_cell[cell_label]))
        md_lines.append("")
        mcnemar_md = _mcnemar_table(cell_label, results[cell_label]["rows"])
        if mcnemar_md:
            md_lines.append(mcnemar_md)
            md_lines.append("")
    md_path = os.path.join(RUNS_DIR, f"heldout_eval_{rule.rule_id}{file_suffix}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print("\n".join(md_lines))
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
