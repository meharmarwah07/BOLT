"""The propose -> replay -> feedback -> re-propose loop (CONTEXT.md
section 4 steps 3-4), for one failure.

Up to MAX_ROUNDS rounds of CANDIDATES_PER_ROUND candidates each. Every
candidate is replayed for real through pipeline/check1.py (sim/run_trial.py
under the hood). As soon as a round produces at least one PASSING
candidate, the one with the lowest Wilson 95% upper bound of combined
(drop-or-crush) failure on TRAIN seeds is returned (ties broken by lower
crush count, then larger crush margin, then smaller slowdown) -- see
_pick_best_passing's own docstring for why this replaced the original
largest-crush-margin key. If no round ever produces a passing candidate,
returns a real rule_check.NoSafeRule("search exhausted") alongside the
full record of every attempt made.

Even though both llm/solver.py and llm/fake_server.py now assign a fresh
rule_id to every reply (see llm/solver.py's `_assign_unique_rule_id` and
llm/fake_server.py's per-request `r_fake_NNNN`), a real model sampled
against few-shot examples could still echo one of THEIR ids, or otherwise
repeat an id across candidates -- solver.py's overwrite is the actual
fix; rules/lifecycle.py's audit log (keyed by rule_id, PROPOSED only
legal from state None) would reject a second proposal reusing an id
regardless. This loop keeps a defensive fallback for that case: only the
FIRST (round, candidate) to produce a given rule_id gets lifecycle events
logged; later candidates with the same id are still replayed through
check1 (so they remain eligible to be picked) but are marked `note=...`
instead of raising.
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(ROOT, "sim")
LLM_DIR = os.path.join(ROOT, "llm")
for _p in (PIPELINE_DIR, SIM_DIR, LLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check1 as check1_mod  # pipeline/check1.py
import rule_check  # llm/rule_check.py -- real NoSafeRule type
import solver  # llm/solver.py
from monitor import make_failure_report  # sim/monitor.py
from run_trial import run_trial  # sim/run_trial.py

from rules import lifecycle
from rules.schema import Rule

MAX_ROUNDS = 3
CANDIDATES_PER_ROUND = 3
MIN_LLM_INTERVAL_S = 4.0  # minimum seconds between the START of two solve() calls
MAX_LLM_CONCURRENCY = 2  # at most this many solve() calls admitted at once


def real_failure_reports(object_class, seeds, cell_id="A", owner_id="owner1"):
    """Real FailureReports for `object_class`, from actual sim/run_trial.py
    trials at pipeline/check1.py's BASELINE_PARAMS (the nominal setting;
    see check1.py for how/why it was chosen), run through sim/monitor.py's
    real make_failure_report -- never hand-typed. `seeds` must be TRAIN
    seeds: CONTEXT.md reserves held-out seeds for the final measurement
    only, and that applies just as much to the reports that SEED the
    solver's very first proposal as to any later replay.
    """
    reports = []
    for seed in seeds:
        result = run_trial(object_class, check1_mod.BASELINE_PARAMS, seed)
        report = make_failure_report(result, cell_id=cell_id, owner_id=owner_id)
        if report is not None:
            reports.append(report)
    return reports


class _Throttle:
    """Serializes call *starts* to be >= min_interval_s apart, and admits
    at most max_concurrency calls at a time (a simple leaky-bucket rate
    limiter). Thread-safe.
    """

    def __init__(self, min_interval_s: float, max_concurrency: int):
        self._min_interval_s = min_interval_s
        self._sem = threading.Semaphore(max_concurrency)
        self._lock = threading.Lock()
        self._last_start = 0.0

    def run(self, fn, *args, **kwargs):
        self._sem.acquire()
        try:
            with self._lock:
                wait = self._min_interval_s - (time.monotonic() - self._last_start)
                if wait > 0:
                    time.sleep(wait)
                self._last_start = time.monotonic()
            return fn(*args, **kwargs)
        finally:
            self._sem.release()


# solver.py's _METRICS/_LAST_CALL are plain module globals, not
# thread-safe. Calling solve() from >1 thread at once risks corrupted
# per-attempt bookkeeping (not fixed -- out of this pass's scope). This
# lock serializes the actual solve() call so MAX_LLM_CONCURRENCY still
# governs admission/timing (genuinely testable, see test_check1.py)
# without tripping that race in practice.
_SOLVE_CALL_LOCK = threading.Lock()


@dataclass
class AttemptRecord:
    round: int
    candidate_index: int
    rule: Optional[Rule] = None
    check1: Optional["check1_mod.Check1Outcome"] = None
    no_safe_rule_reason: Optional[str] = None
    error: Optional[str] = None
    lifecycle_logged: bool = False
    note: Optional[str] = None
    # solver.get_last_call_info() right after this candidate's solve()
    # call: {attempts, attempt_errors, usage_per_attempt}. usage entries
    # are None against llm/fake_server.py (it reports no token usage);
    # populated for a real OpenAI-compatible backend that sends `usage`.
    cost: Optional[dict] = None


@dataclass
class LoopResult:
    # `check1_passed_rule` is deliberately not named "accepted"/"approved":
    # per rules/lifecycle.py's real state machine, CHECK1_PASSED ->
    # APPROVED is Check 2 (human review), which this automated loop never
    # performs. This is the best rule that cleared Check 1, awaiting a
    # human.
    check1_passed_rule: Optional[Rule] = None
    check1_passed_outcome: Optional["check1_mod.Check1Outcome"] = None
    no_safe_rule: Optional[rule_check.NoSafeRule] = None
    attempts: list = field(default_factory=list)


def _build_feedback(round_attempts) -> str:
    lines = []
    for rec in round_attempts:
        if rec.error is not None:
            lines.append(f"Candidate {rec.candidate_index}: solve() failed ({rec.error}).")
        elif rec.no_safe_rule_reason is not None:
            lines.append(f"Candidate {rec.candidate_index}: proposed no_safe_rule ({rec.no_safe_rule_reason}).")
        elif rec.rule is not None and rec.check1 is not None:
            r, c1 = rec.rule, rec.check1
            obj_class = r.object_class.value if hasattr(r.object_class, "value") else r.object_class
            lines.append(
                f"Candidate {rec.candidate_index}: rule grip_force_min_n={r.grip_force_min_n}, "
                f"carry_accel_max_ms2={r.carry_accel_max_ms2} for {obj_class}. Replay (real simulator, "
                f"{c1.n_trials} seeds): drop rate before={c1.drop_rate_before:.2f}, after={c1.drop_rate_after:.2f}, "
                f"fixed {len(c1.seeds_fixed)} seeds, newly broke {len(c1.seeds_newly_broken)} previously-ok seeds, "
                f"crush_count={c1.crush_count_after}. Rejected: {'; '.join(c1.reasons)}"
            )
    return (
        "Every rule proposed so far was rejected by Check 1 (real simulation replay), not by guesswork. "
        "Details per candidate:\n" + "\n".join(lines) + "\n\n"
        "Propose a different rule for the same failure that fixes more of the baseline drops "
        "without crushing the object or breaking any previously-ok seed."
    )


def _pick_best_passing(round_attempts) -> Optional[AttemptRecord]:
    """Among this round's PASSING candidates, the LOWEST Wilson 95% upper
    bound of combined (drop-or-crush) failure on TRAIN seeds wins -- ties
    broken by lower crush count, then larger crush margin, then smaller
    slowdown_proxy. None if none passed.

    Previously keyed on crush margin alone (largest wins): margin ignores
    the per-trial crush jitter (sim/cell_model.py's CRUSH_LIMIT_JITTER)
    and isn't comparable across rule types (a fixed rule's margin is
    1 - grip/nominal_crush_limit; a conditional rule's is 1 - grip_
    fraction -- different quantities that happen to share a name), so it
    could and did pick the WEAKEST rule that cleared the gate -- the
    held-out run on seeds 2000-2199 found the accepted 20N rule actually
    performed worse than 22.5N/24.9N heuristics never shown to the LLM.
    combined_failure_ci95 (pipeline/check1.py's Check1Outcome field,
    already computed for every passing candidate regardless of version or
    rule_type) is check1()'s own gate statistic, so it's both type-
    agnostic and already reflects the measured jitter, not just a static
    parameter.

    Factored out (not inlined in loop()) so the selection rule itself is
    directly unit-testable without going through solve()/fake_server.
    """
    passing = [r for r in round_attempts if r.rule is not None and r.check1 is not None and r.check1.passed]
    if not passing:
        return None
    return min(
        passing,
        key=lambda r: (
            r.check1.combined_failure_ci95[1],
            r.check1.crush_count_after,
            -r.check1.margin,
            r.check1.slowdown_proxy,
        ),
    )


def loop(
    failure_reports,
    train_seeds,
    audit_path,
    actor: str = "pipeline-loop",
    max_rounds: int = MAX_ROUNDS,
    candidates_per_round: int = CANDIDATES_PER_ROUND,
    min_interval_s: float = MIN_LLM_INTERVAL_S,
    max_concurrency: int = MAX_LLM_CONCURRENCY,
    check1_version: Optional[int] = None,
) -> LoopResult:
    """`check1_version` is forwarded to pipeline/check1.py's check1() as its
    `version` arg; None (the default) means check1's own CHECK1_VERSION
    default (currently 2). Pass 1 to recover the old v1 gate for this run.
    """
    throttle = _Throttle(min_interval_s, max_concurrency)
    seen_rule_ids = set()
    attempts = []
    feedback = None

    for round_idx in range(1, max_rounds + 1):

        def _one_candidate(i):
            rec = AttemptRecord(round=round_idx, candidate_index=i)
            try:
                with _SOLVE_CALL_LOCK:
                    obj = throttle.run(solver.solve, failure_reports, feedback=feedback)
                    # Read right here, still inside the lock: get_last_call_info()
                    # reads the same not-thread-safe module global solve() just
                    # wrote, so this read must be serialized with it too, not
                    # just the call itself.
                    rec.cost = solver.get_last_call_info()
            except Exception as e:  # noqa: BLE001 -- surfaced via the attempt record, not raised
                rec.error = str(e)
                return rec
            if isinstance(obj, rule_check.NoSafeRule):
                rec.no_safe_rule_reason = obj.reason
            else:
                rec.rule = obj
            return rec

        with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
            futures = [ex.submit(_one_candidate, i) for i in range(1, candidates_per_round + 1)]
            round_attempts = [f.result() for f in futures]

        # Lifecycle + Check 1, sequential (cheap; avoids concurrent audit-log appends).
        for rec in round_attempts:
            if rec.rule is None:
                attempts.append(rec)
                continue
            rid = rec.rule.rule_id
            if rid in seen_rule_ids:
                rec.note = (
                    f"duplicate rule_id {rid!r} already proposed earlier this loop run "
                    "(known llm/fake_server.py limitation: canned replies reuse a fixed "
                    "rule_id) -- lifecycle PROPOSED/CHECK1_* events skipped for this "
                    "candidate, but it is still replayed and eligible to be picked"
                )
            else:
                seen_rule_ids.add(rid)
                try:
                    lifecycle.transition(audit_path, rid, lifecycle.State.PROPOSED, actor)
                except lifecycle.LifecycleError as e:
                    rec.note = f"PROPOSED transition failed: {e}"

            rec.check1 = check1_mod.check1(rec.rule, train_seeds, version=check1_version)

            if rec.note is None:
                try:
                    lifecycle.record_check1(audit_path, rid, actor, rec.check1.passed, rec.check1.record)
                    rec.lifecycle_logged = True
                except lifecycle.LifecycleError as e:
                    rec.note = f"record_check1 failed: {e}"
            attempts.append(rec)

        best = _pick_best_passing(round_attempts)
        if best is not None:
            return LoopResult(
                check1_passed_rule=best.rule,
                check1_passed_outcome=best.check1,
                no_safe_rule=None,
                attempts=attempts,
            )

        feedback = _build_feedback(round_attempts)

    return LoopResult(
        check1_passed_rule=None,
        check1_passed_outcome=None,
        no_safe_rule=rule_check.NoSafeRule(no_safe_rule=True, reason="search exhausted"),
        attempts=attempts,
    )


def _check1_outcome_to_dict(outcome) -> Optional[dict]:
    if outcome is None:
        return None
    import dataclasses
    return {
        "passed": outcome.passed,
        "reasons": outcome.reasons,
        "record": dataclasses.asdict(outcome.record) if outcome.record is not None else None,
        "drop_rate_before": outcome.drop_rate_before,
        "drop_rate_after": outcome.drop_rate_after,
        "drop_rate_before_ci95": list(outcome.drop_rate_before_ci95),
        "drop_rate_after_ci95": list(outcome.drop_rate_after_ci95),
        "seeds_fixed": outcome.seeds_fixed,
        "seeds_newly_broken": outcome.seeds_newly_broken,
        "fixed_rate_ci95": list(outcome.fixed_rate_ci95) if outcome.fixed_rate_ci95 is not None else None,
        "crush_count_after": outcome.crush_count_after,
        "margin": outcome.margin,
        "slowdown_proxy": outcome.slowdown_proxy,
        "n_trials": outcome.n_trials,
    }


def attempt_record_to_dict(rec: AttemptRecord) -> dict:
    return {
        "round": rec.round,
        "candidate_index": rec.candidate_index,
        "rule": rec.rule.model_dump(mode="json") if rec.rule is not None else None,
        "no_safe_rule_reason": rec.no_safe_rule_reason,
        "error": rec.error,
        "lifecycle_logged": rec.lifecycle_logged,
        "note": rec.note,
        "cost": rec.cost,
        "check1": _check1_outcome_to_dict(rec.check1),
    }


if __name__ == "__main__":
    import argparse
    import json
    import subprocess

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-failures", type=int, default=5,
                    help="real FailureReports (sim/run_trial.py + sim/monitor.py) from TRAIN seeds "
                         "0..n-1 at the nominal setting (pipeline/check1.py's BASELINE_PARAMS)")
    ap.add_argument("--object-class", default="glass_jar")
    ap.add_argument("--check1-seeds", type=int, default=len(check1_mod.DEFAULT_TRAIN_SEEDS),
                    help="seeds 0..n-1 used for each candidate's check1 replay")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument("--candidates-per-round", type=int, default=CANDIDATES_PER_ROUND)
    ap.add_argument("--min-interval-s", type=float, default=MIN_LLM_INTERVAL_S)
    ap.add_argument("--max-concurrency", type=int, default=MAX_LLM_CONCURRENCY)
    ap.add_argument("--check1-version", type=int, default=None, choices=[1, 2],
                     help="forwarded to check1()'s `version`; default: check1.py's own CHECK1_VERSION")
    args = ap.parse_args()

    RUNS_DIR = os.path.join(PIPELINE_DIR, "runs")
    os.makedirs(RUNS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    audit_path = os.path.join(RUNS_DIR, f"loop_{timestamp}_audit.jsonl")

    # TRAIN seeds only (sim/batch.py's TRAIN_SEEDS = range(0, 200)); never
    # the held-out range (1000+) -- see real_failure_reports' docstring.
    failure_reports = real_failure_reports(args.object_class, range(args.n_failures))
    print(f"{len(failure_reports)} real FailureReport(s) from sim/run_trial.py + sim/monitor.py "
          f"at the nominal setting {check1_mod.BASELINE_PARAMS}, train seeds 0-{args.n_failures - 1}")

    # solver.py reads FLEET_LLM_BASE_URL/FLEET_LLM_MODEL/FLEET_LLM_API_KEY/
    # RADEON_API_KEY from the environment automatically (its own module
    # constants, read at import time -- already imported above). If the
    # caller hasn't pointed this at a real backend, fall back to a local
    # stub (llm/fake_server.py) so the CLI is still runnable standalone --
    # this is the ONLY thing that decides is_stub below, not a guess from
    # the URL's contents.
    is_stub = False
    server = None
    if not os.environ.get("FLEET_LLM_BASE_URL"):
        print(
            "NOTE: FLEET_LLM_BASE_URL is not set -- starting a local stub (llm/fake_server.py) "
            "for this run. Set FLEET_LLM_BASE_URL (and FLEET_LLM_API_KEY/RADEON_API_KEY, "
            "FLEET_LLM_MODEL) in the environment to use a real backend instead."
        )
        is_stub = True
        # Scripted so round 1 (candidates_per_round replies) gets a
        # well-formed-but-measured-insufficient reply and round 2 onward
        # gets a measured-sufficient one -- lets this demo path actually
        # reach CHECK1_PASSED, not just exhaust to NoSafeRule.
        script = [{"content": "good"}] * args.candidates_per_round + [{"content": "strong"}]
        server = subprocess.Popen(
            [sys.executable, "fake_server.py"], cwd=LLM_DIR,
            env={**os.environ, "FAKE_SCRIPT": json.dumps(script)},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)
        solver.BASE_URL = "http://localhost:8000/v1"  # solver.py bound BASE_URL at import time, before we knew this

    print(f"model={solver.MODEL} base_url={solver.BASE_URL} "
          f"api_key_set={bool(solver.API_KEY and solver.API_KEY != 'not-needed')}")

    solver.reset_metrics()
    t0 = time.monotonic()
    try:
        result = loop(
            failure_reports, range(args.check1_seeds), audit_path,
            max_rounds=args.max_rounds, candidates_per_round=args.candidates_per_round,
            min_interval_s=args.min_interval_s, max_concurrency=args.max_concurrency,
            check1_version=args.check1_version,
        )
    finally:
        wall_time_s = time.monotonic() - t0
        if server is not None:
            server.terminate()
            server.wait(timeout=5)
    metrics = solver.get_metrics()

    print(f"\n{len(result.attempts)} attempt(s) made, {metrics['http_requests']} total LLM HTTP "
          f"request(s), {wall_time_s:.1f}s wall time")
    for a in result.attempts:
        tag = "rule" if a.rule else ("no_safe_rule" if a.no_safe_rule_reason else "error")
        print(f"  round {a.round} candidate {a.candidate_index}: {tag}"
              + (f" passed={a.check1.passed}" if a.check1 else "")
              + (f" ({a.note})" if a.note else ""))

    run_report = {
        "timestamp": timestamp,
        "is_stub": is_stub,
        "model": solver.MODEL,
        "base_url": solver.BASE_URL,
        "object_class": args.object_class,
        "n_failure_reports": len(failure_reports),
        "check1_seeds": args.check1_seeds,
        "total_llm_requests": metrics["http_requests"],
        "wall_time_s": round(wall_time_s, 3),
        "attempts": [attempt_record_to_dict(a) for a in result.attempts],
        "chosen_rule": result.check1_passed_rule.model_dump(mode="json") if result.check1_passed_rule is not None else None,
        "no_safe_rule_reason": result.no_safe_rule.reason if result.no_safe_rule is not None else None,
    }
    run_path = os.path.join(RUNS_DIR, f"loop_{timestamp}.json")
    with open(run_path, "w") as f:
        json.dump(run_report, f, indent=2)
    print(f"wrote {run_path}")

    if result.check1_passed_rule is not None:
        label = "STUB RULE - not a model result" if is_stub else None
        print(f"\nCHECK1_PASSED rule: {result.check1_passed_rule.rule_id} "
              f"grip_force_min_n={result.check1_passed_rule.grip_force_min_n} "
              f"carry_accel_max_ms2={result.check1_passed_rule.carry_accel_max_ms2}"
              + (f"  [{label}]" if label else ""))
        accepted_path = os.path.join(RUNS_DIR, "accepted_rule.json")
        with open(accepted_path, "w") as f:
            json.dump({
                "label": label,
                "is_stub": is_stub,
                "rule": result.check1_passed_rule.model_dump(mode="json"),
            }, f, indent=2)
        print(f"wrote {accepted_path}" + (f" (labeled {label!r})" if label else ""))
    else:
        print(f"\nNO_SAFE_RULE: {result.no_safe_rule.reason}")
