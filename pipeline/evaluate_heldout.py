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

Multi-rule path (docs/SUCCESS_CRITERION.md item B/C): pass --rule-file
one or more times (each a gate-passing rule's accepted_rule.json-shaped
file, paired with a matching --audit-path in the same order) instead of
--rule-json to evaluate EVERY given rule in ONE invocation -- shared
guards, ONE access-log entry (rule_ids, plural), baselines computed once
per object_class. Each rule gets its own table plus a item-D verdict
(Cell B drop rate lower with paired exact McNemar p < 0.05/k, Bonferroni-
corrected for k rules evaluated; Cell C drop rate not higher; crush not
increased on either cell). Output: heldout_eval_MULTI_<timestamp>
{_DEV}.{json,md}. The original single-rule path (--rule-json, default)
is unchanged when --rule-file is not given.

Multi-rule hardening, under --held-out (checked before any held-out seed
is touched): a --rule-file marked as a stub (is_stub or a "stub" label)
is refused outright; each rule is independently re-verified against the
CURRENT Check 1 gate on TRAIN seeds (never held-out), refused if it no
longer passes; the same --rule-file path or byte-identical content given
twice is refused as a duplicate. See _is_stub_accepted_rule,
_verify_rule_still_passes_check1, and the path/hash de-duplication in
_main_multi_rule.
"""
import argparse
import datetime as dt
import hashlib
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
DEFAULT_AUDIT_PATH = os.path.join(RUNS_DIR, "loop_demo_audit.jsonl")


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
    # rule_id (single-rule log entries) or rule_ids (multi-rule entries) -- whichever is present.
    prior_rule_ref = prior[-1].get("rule_id") or prior[-1].get("rule_ids")
    return (
        f"REFUSED: seeds {final_range.start}-{final_range.stop - 1} are the final held-out range "
        "(docs/FINAL_PROTOCOL.md: 'used once. No rule, prompt, gate or selection change after the "
        f"first look.'), and {access_log_path} already records {len(prior)} prior run(s) touching "
        f"it (most recent: {prior[-1].get('ts')}, rule_id(s)={prior_rule_ref!r}). Pass "
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


def _measure_baselines(target_class, seeds, baseline_conditions):
    """baseline_conditions: {name: params}, must include "none". Returns
    (baseline_batch, {name: row}) -- the "none" BatchSummary plus every
    baseline's row. Computed once per (target_class, cell seeds) and
    reused across every rule evaluated on them (multi-rule evaluation:
    "baselines once", not once per rule)."""
    rows = {}
    baseline_batch = run_batch(target_class, baseline_conditions["none"], seeds, parallel=True)
    rows["none"] = _row(baseline_batch, baseline_batch, baseline_conditions["none"])
    for name, params in baseline_conditions.items():
        if name == "none":
            continue
        batch = run_batch(target_class, params, seeds, parallel=True) if params is not None else baseline_batch
        rows[name] = _row(baseline_batch, batch, params if params is not None else baseline_conditions["none"])
    return baseline_batch, rows


def _measure_rule(baseline_batch, baseline_params, target_class, seeds, rule_params):
    """One rule's own row on `seeds`, reusing the already-computed
    `baseline_batch` (not recomputing it) for the paired fixed/newly-
    broken comparison. rule_params=None means "not evaluated here" (e.g.
    a denied share) -- same as the baseline itself."""
    batch = run_batch(target_class, rule_params, seeds, parallel=True) if rule_params is not None else baseline_batch
    return _row(baseline_batch, batch, rule_params if rule_params is not None else baseline_params)


def _measure(target_class, cell_label, seeds, conditions):
    """conditions: {name: params or None}, optionally including
    "accepted_rule". None means "same as baseline, not evaluated" (used
    for a denied share). Returns {name: row dict}. Thin wrapper over
    _measure_baselines/_measure_rule -- kept so the single-rule path's
    behavior (incl. key order) is unchanged."""
    baseline_conditions = {k: v for k, v in conditions.items() if k != "accepted_rule"}
    baseline_batch, rows = _measure_baselines(target_class, seeds, baseline_conditions)
    if "accepted_rule" in conditions:
        rows["accepted_rule"] = _measure_rule(
            baseline_batch, conditions["none"], target_class, seeds, conditions["accepted_rule"],
        )
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


def _paired_mcnemar(rows, name_a, name_b):
    """Paired exact McNemar (two-sided), task-success (outcome == "ok") on
    each seed common to both `rows[name_a]` and `rows[name_b]`. Returns
    (b, c, p) -- b/c are the discordant counts stats.mcnemar_exact needs;
    see its docstring."""
    a_success = {o["seed"]: o["outcome"] == "ok" for o in rows[name_a]["outcomes_by_seed"]}
    b_success = {o["seed"]: o["outcome"] == "ok" for o in rows[name_b]["outcomes_by_seed"]}
    common = sorted(set(a_success) & set(b_success))
    b = sum(1 for s in common if a_success[s] and not b_success[s])
    c = sum(1 for s in common if not a_success[s] and b_success[s])
    return b, c, llm_stats.mcnemar_exact(b, c)


def _mcnemar_table(cell_label, rows, accepted_name="accepted_rule"):
    """Paired exact McNemar (two-sided), task-success (outcome == "ok") on
    each seed, accepted_rule vs every other condition measured on this
    cell's own seeds. Returns "" if accepted_rule wasn't measured on this
    cell (e.g. this cell's rows only cover the no-rule baseline).
    """
    if accepted_name not in rows:
        return ""
    lines = [
        f"### Cell {cell_label}: paired exact McNemar ({accepted_name} vs each baseline, task success)",
        "",
        "| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |",
        "|---|---|---|---|",
    ]
    for name in rows:
        if name == accepted_name:
            continue
        b, c, p = _paired_mcnemar(rows, accepted_name, name)
        lines.append(f"| {name} | {b} | {c} | {p:.4f} |")
    return "\n".join(lines)


def _is_stub_accepted_rule(accepted_data) -> bool:
    """True if an accepted_rule.json-shaped dict marks itself as a stub
    (loop.py's is_stub flag, or a "label" that says so -- loop.py only
    ever sets label to STUB_LABEL or None, but this also catches any
    other truthy label mentioning "stub")."""
    if accepted_data.get("is_stub"):
        return True
    label = accepted_data.get("label")
    return bool(label) and "stub" in str(label).lower()


def _verify_rule_still_passes_check1(rule):
    """Provenance hardening: re-run the REAL Check 1 (pipeline/check1.py,
    unchanged) on TRAIN seeds (check1_mod.DEFAULT_TRAIN_SEEDS, i.e. 0-99 --
    never held-out) against the CURRENT registered gate
    (check1_mod.CHECK1_VERSION), independent of whatever audit_path's
    lifecycle log says. Catches a rule file that no longer clears the
    currently-registered gate (stale, hand-edited, or from a since-
    tightened check1 version) even if its own audit log still claims
    CHECK1_PASSED. Returns the real check1_mod.Check1Outcome."""
    return check1_mod.check1(rule, check1_mod.DEFAULT_TRAIN_SEEDS)


def _main_multi_rule(args, cell_b_start, cell_b_end, cell_c_start, cell_c_end, audit_paths):
    """Multi-rule path (--rule-file, one or more): evaluates every given
    gate-passing rule in ONE invocation. docs/SUCCESS_CRITERION.md:
    item B (every gate-passing rule evaluated, no selection among runs),
    item C (one evaluator invocation, seeds 3000-3199, run once), item D
    (per-rule success with a Bonferroni-corrected McNemar threshold),
    item F (zero rules -> no final evaluation, handled by the caller
    simply never invoking this function with an empty --rule-file list).

    Shares the spent-seed/single-use guards and the access-log write with
    the single-rule path (run once here, not per rule), and shares
    baselines across every rule targeting the same object_class (computed
    once per class, not once per rule).
    """
    # Pass 1 (cheap, no sim call): existence + duplicate detection across
    # ALL --rule-file arguments, before anything else -- hardening item 3.
    # The same path (even spelled differently, e.g. a symlink or a
    # relative vs absolute form) or byte-identical content given twice
    # would silently double-count one rule in k (the Bonferroni
    # denominator) and in the access log. Done as its own pass (not
    # folded into pass 2 below) so a duplicate is always caught on its
    # SECOND occurrence regardless of whether the FIRST occurrence would
    # itself be refused by the (expensive, real-sim) provenance re-check
    # in pass 2 -- duplicate detection must not depend on rule validity.
    raw_bytes_by_file = {}
    seen_real_paths = {}  # realpath -> the --rule-file string that first used it
    seen_content_hashes = {}  # sha256 of file bytes -> the --rule-file string that first used it
    for rule_file in args.rule_file:
        if not os.path.exists(rule_file):
            print(
                f"REFUSED: {rule_file} does not exist. Run pipeline/loop.py first -- it writes "
                "this file once a candidate reaches CHECK1_PASSED.",
                file=sys.stderr,
            )
            sys.exit(1)
        real_path = os.path.realpath(rule_file)
        if real_path in seen_real_paths:
            print(
                f"WARNING: --rule-file {rule_file!r} is the same file as the earlier "
                f"--rule-file {seen_real_paths[real_path]!r}.\n"
                "REFUSED: each rule must be evaluated exactly once -- remove the duplicate --rule-file.",
                file=sys.stderr,
            )
            sys.exit(1)
        seen_real_paths[real_path] = rule_file
        with open(rule_file, "rb") as f:
            raw_bytes = f.read()
        raw_bytes_by_file[rule_file] = raw_bytes
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        if content_hash in seen_content_hashes:
            print(
                f"WARNING: --rule-file {rule_file!r} has IDENTICAL content to the earlier "
                f"--rule-file {seen_content_hashes[content_hash]!r} (same sha256, different path).\n"
                "REFUSED: each rule must be evaluated exactly once -- remove the duplicate --rule-file.",
                file=sys.stderr,
            )
            sys.exit(1)
        seen_content_hashes[content_hash] = rule_file

    # Pass 2: parse, stub-check (item 1), provenance re-check (item 2,
    # runs a real sim on TRAIN seeds only -- never held-out).
    rule_entries = []  # (rule_file, audit_path, rule_label, is_stub, rule)
    for rule_file, audit_path in zip(args.rule_file, audit_paths):
        accepted_data = json.loads(raw_bytes_by_file[rule_file])

        if args.held_out and _is_stub_accepted_rule(accepted_data):
            print(
                f"REFUSED: {rule_file} is a STUB rule (is_stub={accepted_data.get('is_stub')!r}, "
                f"label={accepted_data.get('label')!r}) -- stub rules (loop.py's local-stub demo "
                "fallback) cannot be evaluated under --held-out.",
                file=sys.stderr,
            )
            sys.exit(1)

        rule_label = accepted_data.get("label")
        rule = Rule.model_validate(accepted_data["rule"])

        if args.held_out:
            reverify = _verify_rule_still_passes_check1(rule)
            if not reverify.passed:
                print(
                    f"REFUSED: {rule_file} (rule_id={rule.rule_id}) no longer passes Check 1 when "
                    f"re-verified now against the current gate (reasons: {'; '.join(reverify.reasons)}). "
                    "Provenance check failed -- this rule cannot be evaluated under --held-out.",
                    file=sys.stderr,
                )
                sys.exit(1)

        rule_entries.append((rule_file, audit_path, rule_label, accepted_data.get("is_stub"), rule))

    k = len(rule_entries)
    bonferroni_threshold = 0.05 / k

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
    top_notice = (
        "NOTICE: this is a non-interactive demo script. Check 2 (human approval) is "
        "AUTO-APPROVED here for the purpose of this held-out evaluation -- a real "
        "deployment requires an actual human reviewer (rules/review_cli.py), not this "
        "script. The baseline ('none' / 'without the rule') condition in every table "
        f"below is the nominal setting {dict(check1_mod.BASELINE_PARAMS)} "
        "(grip_force_n, carry_accel_ms2) -- see pipeline/check1.py's BASELINE_PARAMS "
        f"for how and why it was chosen. docs/SUCCESS_CRITERION.md item D(i): Bonferroni-"
        f"adjusted significance threshold 0.05/{k} = {bonferroni_threshold:.6g} (k = {k} "
        "rules evaluated in this invocation). Cell C verdicts below use the "
        "'C_sharing_allowed' counterfactual (would this rule hurt/help Cell C if shared), "
        "not the realistic default decision, which is reported separately."
    )
    print(top_notice)
    print()

    for _rule_file, audit_path, _rule_label, _is_stub, rule in rule_entries:
        _ensure_approved(audit_path, rule, args.actor)

    cell_b_seeds = range(cell_b_start, cell_b_end)
    cell_c_seeds = range(cell_c_start, cell_c_end)

    os.makedirs(RUNS_DIR, exist_ok=True)
    if not args.dev:
        os.makedirs(os.path.dirname(os.path.abspath(args.access_log_path)) or ".", exist_ok=True)
        with open(args.access_log_path, "a") as f:
            f.write(json.dumps({
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "actor": args.actor,
                "rule_ids": [rule.rule_id for *_, rule in rule_entries],
                "cell_b_seeds": [cell_b_seeds.start, cell_b_seeds.stop],
                "cell_c_seeds": [cell_c_seeds.start, cell_c_seeds.stop],
                "override_reason": args.override_reason,
            }) + "\n")

    cell_a = CellState("A", OWNER_A_B, check1_mod.BASELINE_PARAMS)
    cell_b = CellState("B", OWNER_A_B, check1_mod.BASELINE_PARAMS)
    cell_c = CellState("C", OWNER_C, check1_mod.BASELINE_PARAMS)
    owners_denied = {OWNER_A_B: Owner(OWNER_A_B, sharing_enabled=False), OWNER_C: Owner(OWNER_C)}
    owners_allowed = {OWNER_A_B: Owner(OWNER_A_B, sharing_enabled=True), OWNER_C: Owner(OWNER_C)}

    # Baselines, once per target_class (not once per rule) -- most/all
    # rules are expected to share one class (CONTEXT.md's glass_jar), but
    # this is correct even if they don't.
    baselines_by_class = {}

    def _baselines_for(target_class):
        if target_class in baselines_by_class:
            return baselines_by_class[target_class]
        grid_rule = _grid_search_mildest_passing_rule(target_class, [f"multi_rule_eval_{target_class}"])
        grid_note = None
        if grid_rule is None:
            grid_note = (
                f"none passes v{check1_mod.CHECK1_VERSION} -- no (grip, accel) grid point for "
                f"{target_class!r} cleared Check 1 v{check1_mod.CHECK1_VERSION} on GRID_TRAIN_SEEDS "
                f"{GRID_TRAIN_SEEDS.start}-{GRID_TRAIN_SEEDS.stop - 1} (resolution {GRID_RESOLUTION}x"
                f"{GRID_RESOLUTION})."
            )
            print(f"NOTE: {grid_note}", file=sys.stderr)
        crush_limit = CRUSH_LIMIT_N[target_class]
        naive_params = {"grip_force_n": round(0.9 * crush_limit, 3), "carry_accel_ms2": CARRY_ACCEL_MIN_MS2}
        ceiling_params = {
            "grip_force_n": round(crush_limit - CEILING_GRIP_MARGIN_N, 3), "carry_accel_ms2": CARRY_ACCEL_MIN_MS2,
        }
        baseline_conditions = {
            "none": dict(check1_mod.BASELINE_PARAMS),
            "naive_0.9x_crush_min_accel": naive_params,
            "ceiling_near_crush_min_accel": ceiling_params,
            "grid_mildest_passing": (
                apply(grid_rule, check1_mod.BASELINE_PARAMS, target_class) if grid_rule is not None else None
            ),
        }
        b_batch, b_rows = _measure_baselines(target_class, cell_b_seeds, baseline_conditions)
        c_batch, c_rows = _measure_baselines(target_class, cell_c_seeds, baseline_conditions)
        entry = {
            "grid_rule": grid_rule, "grid_note": grid_note,
            "naive_params": naive_params, "ceiling_params": ceiling_params,
            "conditions": baseline_conditions,
            "B": (b_batch, b_rows), "C": (c_batch, c_rows),
        }
        baselines_by_class[target_class] = entry
        return entry

    rule_reports = []
    md_blocks = []
    summary_rows = []
    for rule_file, audit_path, rule_label, is_stub, rule in rule_entries:
        target_class = _class_value(rule.object_class)
        baseline = _baselines_for(target_class)
        b_batch, b_baseline_rows = baseline["B"]
        c_batch, c_baseline_rows = baseline["C"]

        try:
            rule_after_params = apply(rule, check1_mod.BASELINE_PARAMS, target_class)
        except RuleRejected as e:
            print(f"REFUSED: apply() rejected rule {rule.rule_id}: {e}", file=sys.stderr)
            sys.exit(1)

        decision_b = can_share(audit_path, rule, cell_a, cell_b, owners_denied)
        decision_c_denied = can_share(audit_path, rule, cell_a, cell_c, owners_denied)
        decision_c_allowed = can_share(audit_path, rule, cell_a, cell_c, owners_allowed)

        rows_b = dict(b_baseline_rows)
        rows_b["accepted_rule"] = _measure_rule(
            b_batch, check1_mod.BASELINE_PARAMS, target_class, cell_b_seeds,
            rule_after_params if decision_b.allowed else None,
        )
        rows_c_denied = dict(c_baseline_rows)
        rows_c_denied["accepted_rule"] = _measure_rule(
            c_batch, check1_mod.BASELINE_PARAMS, target_class, cell_c_seeds,
            rule_after_params if decision_c_denied.allowed else None,
        )
        rows_c_allowed = dict(c_baseline_rows)
        rows_c_allowed["accepted_rule"] = _measure_rule(
            c_batch, check1_mod.BASELINE_PARAMS, target_class, cell_c_seeds,
            rule_after_params if decision_c_allowed.allowed else None,
        )

        # Item D: (i) uses Cell B; (ii)/(iii) use the 'C_sharing_allowed'
        # counterfactual (see top_notice) -- the realistic default
        # ('C_sharing_denied') never applies the rule, so comparing it
        # against itself would be trivially unchanged.
        _, _, p_b = _paired_mcnemar(rows_b, "accepted_rule", "none")
        verdict_i = rows_b["accepted_rule"]["drop_rate"] < rows_b["none"]["drop_rate"] and p_b < bonferroni_threshold
        verdict_ii = rows_c_allowed["accepted_rule"]["drop_rate"] <= rows_c_allowed["none"]["drop_rate"]
        verdict_iii = (
            rows_b["accepted_rule"]["n_crush"] <= rows_b["none"]["n_crush"]
            and rows_c_allowed["accepted_rule"]["n_crush"] <= rows_c_allowed["none"]["n_crush"]
        )
        verdict_d = verdict_i and verdict_ii and verdict_iii

        results = {
            "B": {"seeds": [cell_b_seeds.start, cell_b_seeds.stop],
                  "can_share_decision": {"allowed": decision_b.allowed, "reason": decision_b.reason}, "rows": rows_b},
            "C_sharing_denied": {"seeds": [cell_c_seeds.start, cell_c_seeds.stop],
                                  "can_share_decision": {"allowed": decision_c_denied.allowed, "reason": decision_c_denied.reason},
                                  "rows": rows_c_denied},
            "C_sharing_allowed": {"seeds": [cell_c_seeds.start, cell_c_seeds.stop],
                                   "can_share_decision": {"allowed": decision_c_allowed.allowed, "reason": decision_c_allowed.reason},
                                   "rows": rows_c_allowed},
        }
        stub_label = rule_label or (STUB_LABEL if is_stub else None)
        rule_reports.append({
            "rule_id": rule.rule_id,
            "rule_file": rule_file,
            "stub_label": stub_label,
            "target_class": target_class,
            "rule": rule.model_dump(mode="json"),
            "grid_mildest_passing_rule": baseline["grid_rule"].model_dump(mode="json") if baseline["grid_rule"] is not None else None,
            "grid_mildest_passing_note": baseline["grid_note"],
            "naive_baseline_params": baseline["naive_params"],
            "ceiling_params": baseline["ceiling_params"],
            "results": results,
            "criterion_d": {
                "i_cell_b_drop_rate_lower_mcnemar_p": {"pass": verdict_i, "p_value": p_b, "threshold": bonferroni_threshold},
                "ii_cell_c_drop_rate_not_higher": {"pass": verdict_ii},
                "iii_crush_not_increased": {"pass": verdict_iii},
                "pass": verdict_d,
            },
        })

        summary_rows.append(
            f"| {rule.rule_id} | {target_class} | {rows_b['none']['drop_rate']:.3f} | "
            f"{rows_b['accepted_rule']['drop_rate']:.3f} | {p_b:.4g} | {bonferroni_threshold:.4g} | "
            f"{rows_c_allowed['none']['drop_rate']:.3f} | {rows_c_allowed['accepted_rule']['drop_rate']:.3f} | "
            f"{rows_b['none']['n_crush']}->{rows_b['accepted_rule']['n_crush']} | "
            f"{rows_c_allowed['none']['n_crush']}->{rows_c_allowed['accepted_rule']['n_crush']} | "
            f"{'PASS' if verdict_d else 'FAIL'} |"
        )

        md_blocks.append(f"## Rule {rule.rule_id} ({target_class}){f' [{stub_label}]' if stub_label else ''}")
        md_blocks.append("")
        for cell_label in ("B", "C_sharing_denied", "C_sharing_allowed"):
            cell_result = results[cell_label]
            md_blocks.append(
                f"Cell {cell_label} can_share: allowed={cell_result['can_share_decision']['allowed']} "
                f"({cell_result['can_share_decision']['reason']})"
            )
            md_blocks.append("")
            controller_by_condition = dict(baseline["conditions"])
            controller_by_condition["accepted_rule"] = (
                rule_after_params if cell_result["can_share_decision"]["allowed"] else check1_mod.BASELINE_PARAMS
            )
            md_blocks.append(_markdown_table(cell_label, cell_result["rows"], controller_by_condition))
            md_blocks.append("")
            mcnemar_md = _mcnemar_table(cell_label, cell_result["rows"])
            if mcnemar_md:
                md_blocks.append(mcnemar_md)
                md_blocks.append("")
        md_blocks.append(
            f"criterion D: (i) Cell B drop rate lower & p<{bonferroni_threshold:.4g}: {verdict_i} "
            f"(p={p_b:.4g}); (ii) Cell C drop rate not higher: {verdict_ii}; (iii) crush not increased: "
            f"{verdict_iii} => **{'PASS' if verdict_d else 'FAIL'}**"
        )
        md_blocks.append("")

    report = {
        "dev_run": args.dev,
        "notice": top_notice,
        "baseline_params": dict(check1_mod.BASELINE_PARAMS),
        "k_rules_evaluated": k,
        "bonferroni_threshold": bonferroni_threshold,
        "rules": rule_reports,
    }

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_suffix = "_DEV" if args.dev else ""
    json_path = os.path.join(RUNS_DIR, f"heldout_eval_MULTI_{timestamp}{file_suffix}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [f"# Held-out evaluation (multi-rule, k={k}){' [DEV RUN]' if args.dev else ''}", ""]
    if dev_banner:
        md_lines += [dev_banner, ""]
    md_lines += [top_notice, ""]
    md_lines += [
        "| rule_id | class | B drop (none) | B drop (rule) | B McNemar p | Bonferroni thr. | "
        "C drop (none) | C drop (rule) | B crush none->rule | C crush none->rule | verdict D |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    md_lines += summary_rows
    md_lines.append("")
    md_lines += md_blocks
    md_path = os.path.join(RUNS_DIR, f"heldout_eval_MULTI_{timestamp}{file_suffix}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print("\n".join(md_lines))
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--held-out", action="store_true", help="required (unless --dev): confirms this run is allowed to touch held-out seeds")
    p.add_argument("--dev", action="store_true",
                    help="dev mode: seeds must be in DEV_SEEDS_RANGE (5000-5199), never touches "
                         "heldout_access.log, and stamps DEV RUN in the output. Mutually exclusive with --held-out.")
    p.add_argument("--rule-json", default=os.path.join(RUNS_DIR, "accepted_rule.json"),
                    help="single-rule path (default). Ignored if --rule-file is given.")
    p.add_argument("--rule-file", action="append", default=None,
                    help="multi-rule path: one gate-passing rule's accepted_rule.json-shaped file "
                         "(docs/SUCCESS_CRITERION.md item B: every gate-passing rule from every "
                         "revised run is evaluated, no selection among runs). Repeat for each rule; "
                         "pair with a matching --audit-path in the same order (one per --rule-file). "
                         "Evaluates all of them in ONE invocation: baselines once, one access-log entry.")
    p.add_argument("--audit-path", action="append", default=None,
                    help=f"single-rule path: at most one (default: {DEFAULT_AUDIT_PATH}). Multi-rule "
                         "path (--rule-file): exactly one per --rule-file, same order.")
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

    audit_paths = args.audit_path if args.audit_path else [DEFAULT_AUDIT_PATH]

    if args.rule_file:
        if len(audit_paths) != len(args.rule_file):
            print(
                f"REFUSED: --rule-file was given {len(args.rule_file)} time(s) but --audit-path "
                f"{len(audit_paths)} time(s) -- pass exactly one --audit-path per --rule-file, in "
                "the same order.",
                file=sys.stderr,
            )
            sys.exit(1)
        _main_multi_rule(args, cell_b_start, cell_b_end, cell_c_start, cell_c_end, audit_paths)
        return

    if len(audit_paths) != 1:
        print(
            f"REFUSED: --audit-path was given {len(audit_paths)} times without --rule-file -- the "
            "single-rule path (--rule-json) takes at most one --audit-path.",
            file=sys.stderr,
        )
        sys.exit(1)
    audit_path = audit_paths[0]

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

    _ensure_approved(audit_path, rule, args.actor)

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

    decision_b = can_share(audit_path, rule, cell_a, cell_b, owners_denied)
    decision_c_denied = can_share(audit_path, rule, cell_a, cell_c, owners_denied)
    decision_c_allowed = can_share(audit_path, rule, cell_a, cell_c, owners_allowed)

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
