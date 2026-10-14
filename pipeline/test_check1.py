"""Hermetic regression tests for pipeline/check1.py, pipeline/loop.py and
pipeline/evaluate_heldout.py's held-out guard.

Same PASS/FAIL harness style as llm/test_solver.py (a flat list of named
checks, each returning True/(False, reason), printed and summarized --
not pytest). Hermetic: FLEET_*/RADEON* env vars are deleted before any
pipeline/llm module is imported, and the only "LLM" ever contacted is
llm/fake_server.py (a local subprocess), never a real API.

Run with:
    env -u PYTHONPATH PYTHONPATH=$HOME/fleet .venv/bin/python3 pipeline/test_check1.py
"""
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

for _k in [k for k in os.environ if k.startswith("FLEET_") or k.startswith("RADEON")]:
    del os.environ[_k]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIM_DIR = os.path.join(ROOT, "sim")
LLM_DIR = os.path.join(ROOT, "llm")
for _p in (HERE, SIM_DIR, LLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check1 as check1_mod
import evaluate_heldout as evaluate_heldout_mod
import loop as loop_mod
from rules import lifecycle
from rules.schema import Rule
import rules.limits as rules_limits

PORT = 8000
TRAIN_SEEDS = range(30)
# Measured safe for glass_jar by an earlier audit (sim/STATUS.md): mean
# min holding grip for glass_jar is ~17-20N depending on accel, so 24N
# at a low 1.0 m/s^2 accel clears it with margin, under the 25N crush limit.
GOOD_GLASS_JAR = dict(grip_force_min_n=24.0, carry_accel_max_ms2=1.0)


def _port_open(port=PORT):
    with socket.socket() as s:
        s.settimeout(0.2)
        try:
            s.connect(("localhost", port))
            return True
        except OSError:
            return False


def _wait_for_port(port=PORT, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.05)
    return False


def _wait_for_port_free(port=PORT, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _port_open(port):
            return True
        time.sleep(0.05)
    return False


def _stop_server(server):
    """Terminate a fake_server.py subprocess and wait for its port to
    actually free up -- without this, the next test's _wait_for_port can
    lose a race against this OS's own socket-release delay."""
    server.terminate()
    try:
        server.wait(timeout=3)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    _wait_for_port_free()


def _fetch_log(port=PORT):
    with urllib.request.urlopen(f"http://localhost:{port}/__log__", timeout=2) as resp:
        return json.loads(resp.read())


def _start_round1_insufficient_round2_strong_server():
    """Round 1's 3 requests get the well-formed-but-insufficient glass_jar
    reply (14N/2.5); request 4 onward (round 2) gets the measured-sufficient
    one (24N/1.0, llm/fake_server.py's "strong")."""
    script = [{"content": "good"}] * loop_mod.CANDIDATES_PER_ROUND + [{"content": "strong"}]
    return subprocess.Popen(
        [sys.executable, "fake_server.py"], cwd=LLM_DIR,
        env={**os.environ, "FAKE_SCRIPT": json.dumps(script)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _rule(**overrides):
    fields = dict(
        rule_id="r_test", object_class="glass_jar",
        grip_force_min_n=14.0, carry_accel_max_ms2=2.5,
        proposed_by="test", created_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        rationale="test rule", source_report_ids=["fr_0001"],
    )
    fields.update(overrides)
    return Rule(**fields)


def test_good_rule_passes():
    outcome = check1_mod.check1(_rule(**GOOD_GLASS_JAR), TRAIN_SEEDS)
    if not outcome.passed:
        return False, f"expected a clearly-good glass_jar rule to pass; reasons={outcome.reasons}"
    if outcome.record.crush_count != 0:
        return False, f"expected crush_count 0, got {outcome.record.crush_count}"
    return True, ""


def test_grip_too_low_fails():
    outcome = check1_mod.check1(_rule(grip_force_min_n=5.0, carry_accel_max_ms2=2.5), TRAIN_SEEDS)
    if outcome.passed:
        return False, "expected a too-low grip rule to fail check1"
    return True, ""


def test_default_train_seeds_is_100():
    """check1() with no train_seeds argument uses DEFAULT_TRAIN_SEEDS
    (0-99, 100 seeds) -- the spec's "100 train seeds by default"."""
    if list(check1_mod.DEFAULT_TRAIN_SEEDS) != list(range(100)):
        return False, f"expected DEFAULT_TRAIN_SEEDS == range(100), got {check1_mod.DEFAULT_TRAIN_SEEDS}"
    outcome = check1_mod.check1(_rule(**GOOD_GLASS_JAR))  # no train_seeds -- uses the default
    if outcome.n_trials != 100:
        return False, f"expected 100 trials when train_seeds is omitted, got {outcome.n_trials}"
    if not outcome.passed:
        return False, f"expected the good rule to still pass at 100 seeds; reasons={outcome.reasons}"
    return True, ""


def test_wilson_lower_bound_gates_pass_not_point_estimate():
    """Measured, not fabricated: grip=17N/accel=1.0 for glass_jar on 10
    seeds fixes 6/10 baseline drops (60% point estimate -- would have
    passed the OLD >= 50% point-estimate rule) but its Wilson 95% CI
    lower bound is 0.313, below MIN_FIXED_RATE_CI95_LOWER (0.5) -- so it
    must fail under the current rule. This is the real behavior change
    item 1 asked for, demonstrated with a real measured case rather than
    asserted against a synthetic number.
    """
    outcome = check1_mod.check1(_rule(grip_force_min_n=17.0, carry_accel_max_ms2=1.0), range(10))
    n_base = outcome.record.drops_before
    n_fixed = len(outcome.seeds_fixed)
    if n_base == 0:
        return False, "expected some baseline drops at n=10 seeds"
    point_estimate = n_fixed / n_base
    if point_estimate < check1_mod.MIN_FIXED_RATE_CI95_LOWER:
        return False, (
            f"expected the point estimate ({n_fixed}/{n_base}={point_estimate:.2f}) to be >= "
            f"{check1_mod.MIN_FIXED_RATE_CI95_LOWER} for this to be a meaningful demonstration"
        )
    if outcome.fixed_rate_ci95[0] >= check1_mod.MIN_FIXED_RATE_CI95_LOWER:
        return False, (
            f"expected the Wilson 95% CI lower bound ({outcome.fixed_rate_ci95[0]:.3f}) to be below "
            f"{check1_mod.MIN_FIXED_RATE_CI95_LOWER} at this small n, for this to be a meaningful demonstration"
        )
    if outcome.passed:
        return False, "expected check1 to fail despite the point estimate clearing 50%, via the CI lower bound"
    if outcome.crush_count_after != 0 or outcome.seeds_newly_broken:
        return False, "expected the ONLY failure reason to be the fixed-rate CI gate"
    return True, ""


def test_grip_above_crush_limit_fails():
    # 30N > glass_jar's crush_limit_n (25.0) -- rules.validate() itself
    # must reject this before any replay runs.
    outcome = check1_mod.check1(_rule(grip_force_min_n=30.0, carry_accel_max_ms2=2.5), TRAIN_SEEDS)
    if outcome.passed:
        return False, "expected a crush-limit-exceeding rule to fail check1"
    if not any("validate" in r for r in outcome.reasons):
        return False, f"expected the failure reason to cite rules.validate; got {outcome.reasons}"
    return True, ""


def test_determinism():
    rule = _rule(**GOOD_GLASS_JAR)
    a = check1_mod.check1(rule, TRAIN_SEEDS)
    b = check1_mod.check1(rule, TRAIN_SEEDS)
    if (a.passed, a.reasons, a.record, a.seeds_fixed, a.seeds_newly_broken) != \
       (b.passed, b.reasons, b.record, b.seeds_fixed, b.seeds_newly_broken):
        return False, "two check1() calls with identical inputs gave different results"
    return True, ""


def test_serial_equals_parallel():
    rule = _rule(**GOOD_GLASS_JAR)
    serial = check1_mod.check1(rule, TRAIN_SEEDS, parallel=False)
    parallel = check1_mod.check1(rule, TRAIN_SEEDS, parallel=True)
    if (serial.passed, serial.reasons, serial.record, serial.seeds_fixed, serial.seeds_newly_broken) != \
       (parallel.passed, parallel.reasons, parallel.record, parallel.seeds_fixed, parallel.seeds_newly_broken):
        return False, "serial and parallel check1() runs disagree"
    return True, ""


def test_heldout_guard_refuses_without_flag():
    access_log = os.path.join(HERE, "runs", "heldout_access.log")
    before = open(access_log).read() if os.path.exists(access_log) else None

    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "evaluate_heldout.py")],
        cwd=HERE, env={**os.environ}, capture_output=True, text=True, timeout=30,
    )
    after = open(access_log).read() if os.path.exists(access_log) else None

    if result.returncode == 0:
        return False, f"expected a non-zero exit without --held-out, got 0 (stdout={result.stdout!r})"
    if "REFUSED" not in result.stderr:
        return False, f"expected a REFUSED message on stderr, got {result.stderr!r}"
    if before != after:
        return False, "heldout_access.log changed even though --held-out was never passed"
    return True, ""


def test_heldout_refuses_without_rule_json():
    rule_json = os.path.join(HERE, "runs", "test_missing_accepted_rule.json")
    if os.path.exists(rule_json):
        os.remove(rule_json)
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "evaluate_heldout.py"), "--held-out", "--rule-json", rule_json],
        cwd=HERE, env={**os.environ}, capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        return False, f"expected a non-zero exit when --rule-json is missing, got 0 (stdout={result.stdout!r})"
    if "REFUSED" not in result.stderr or "does not exist" not in result.stderr:
        return False, f"expected a clear REFUSED/'does not exist' message on stderr; got {result.stderr!r}"
    return True, ""


def _write_demo_accepted_rule(rule_json_path, audit_path, is_stub=True, grip=24.0, accel=1.0):
    """Hand-builds a CHECK1_PASSED rule + matching audit log + accepted_rule.json
    (the same wrapper shape loop.py writes), so evaluate_heldout.py tests don't
    need to run the full loop()/fake_server dance just to get a starting point."""
    rule = Rule(
        rule_id="r_heldout_test", object_class="glass_jar",
        grip_force_min_n=grip, carry_accel_max_ms2=accel,
        proposed_by="test", created_at=dt.datetime.now(dt.timezone.utc),
        rationale="test", source_report_ids=["fr_0001"],
    )
    if os.path.exists(audit_path):
        os.remove(audit_path)
    lifecycle.transition(audit_path, rule.rule_id, lifecycle.State.PROPOSED, "test")
    outcome = check1_mod.check1(rule, range(30))
    lifecycle.record_check1(audit_path, rule.rule_id, "test", outcome.passed, outcome.record)
    with open(rule_json_path, "w") as f:
        json.dump({
            "label": evaluate_heldout_mod.STUB_LABEL if is_stub else None,
            "is_stub": is_stub,
            "rule": rule.model_dump(mode="json"),
        }, f)
    return rule


def test_heldout_shows_both_cell_c_scenarios_ceiling_row_and_stub_label():
    rule_json = os.path.join(HERE, "runs", "test_heldout_rule.json")
    audit_path = os.path.join(HERE, "runs", "test_heldout_audit.jsonl")
    rule = _write_demo_accepted_rule(rule_json, audit_path, is_stub=True)

    result = subprocess.run(
        [
            sys.executable, os.path.join(HERE, "evaluate_heldout.py"), "--held-out",
            "--rule-json", rule_json, "--audit-path", audit_path,
            "--cell-b-start", "1000", "--cell-b-end", "1010",
            "--cell-c-start", "1100", "--cell-c-end", "1110",
        ],
        cwd=HERE, env={**os.environ}, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return False, f"evaluate_heldout.py failed (rc={result.returncode}): {result.stderr.strip()!r}"
    if evaluate_heldout_mod.STUB_LABEL not in result.stdout:
        return False, f"expected {evaluate_heldout_mod.STUB_LABEL!r} in stdout; got {result.stdout[:300]!r}"

    json_path = os.path.join(HERE, "runs", f"heldout_eval_{rule.rule_id}.json")
    with open(json_path) as f:
        report = json.load(f)
    if report.get("stub_label") != evaluate_heldout_mod.STUB_LABEL:
        return False, f"expected stub_label={evaluate_heldout_mod.STUB_LABEL!r} in the written JSON; got {report.get('stub_label')!r}"

    md_path = os.path.join(HERE, "runs", f"heldout_eval_{rule.rule_id}.md")
    if evaluate_heldout_mod.STUB_LABEL not in open(md_path).read():
        return False, f"expected {evaluate_heldout_mod.STUB_LABEL!r} in the written .md"

    results = report["results"]
    if "C_sharing_denied" not in results or "C_sharing_allowed" not in results:
        return False, f"expected both C_sharing_denied and C_sharing_allowed; got {list(results)}"
    if results["C_sharing_denied"]["can_share_decision"]["allowed"]:
        return False, "expected C_sharing_denied to be denied"
    if not results["C_sharing_allowed"]["can_share_decision"]["allowed"]:
        return False, "expected C_sharing_allowed to be allowed"

    for cell_label, cell_result in results.items():
        if "ceiling_near_crush_min_accel" not in cell_result["rows"]:
            return False, f"expected a ceiling_near_crush_min_accel row for {cell_label}"
    ceiling_grip = report["ceiling_params"]["grip_force_n"]
    crush_limit = rules_limits.CRUSH_LIMIT_N["glass_jar"]
    if not (crush_limit - 1.0 < ceiling_grip < crush_limit):
        return False, f"expected the ceiling grip just under the crush limit ({crush_limit}); got {ceiling_grip}"
    return True, ""


def test_loop_no_safe_rule_for_unfixable_seeds():
    """glass_jar + llm/fake_server.py's canned reply (grip=14N) is, per the
    real simulator, not actually enough to hold glass_jar reliably (measured
    min holding grip ~17-20N, sim/STATUS.md) -- so every candidate across
    every round should fail check1 the same way, every round, and the loop
    must exhaust to NoSafeRule rather than loop forever or crash.
    """
    server = subprocess.Popen(
        [sys.executable, "fake_server.py"], cwd=LLM_DIR,
        env={**os.environ, "FAKE_ALWAYS_GOOD": "1"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_port():
            return False, "fake_server.py never started"

        os.environ["FLEET_LLM_BASE_URL"] = f"http://localhost:{PORT}/v1"
        # Real sim trials (run_trial + monitor), train seeds only -- see
        # loop.real_failure_reports' docstring.
        failure_reports = loop_mod.real_failure_reports("glass_jar", range(5))
        audit_path = os.path.join(HERE, "runs", "test_loop_audit.jsonl")
        if os.path.exists(audit_path):
            os.remove(audit_path)

        result = loop_mod.loop(
            failure_reports, range(15), audit_path,
            max_rounds=3, candidates_per_round=3, min_interval_s=0.0,  # 0.0: test only, prod default 4.0s
        )
        if result.check1_passed_rule is not None:
            return False, "expected NoSafeRule (the fake server's rule is too weak for glass_jar), got an accepted rule"
        if result.no_safe_rule is None or result.no_safe_rule.reason != "search exhausted":
            return False, f"expected NoSafeRule('search exhausted'), got {result.no_safe_rule}"
        if len(result.attempts) != 9:
            return False, f"expected 9 attempts (3 rounds x 3 candidates), got {len(result.attempts)}"
        return True, ""
    finally:
        os.environ.pop("FLEET_LLM_BASE_URL", None)
        _stop_server(server)


def _run_round1_insufficient_round2_strong():
    """Shared setup for the next three tests: round 1 (3 candidates) all
    get the insufficient reply, round 2 (3 candidates) all get the
    sufficient one. Returns (result, audit_path, server_log)."""
    server = _start_round1_insufficient_round2_strong_server()
    try:
        if not _wait_for_port():
            raise RuntimeError("fake_server.py never started")
        os.environ["FLEET_LLM_BASE_URL"] = f"http://localhost:{PORT}/v1"
        failure_reports = loop_mod.real_failure_reports("glass_jar", range(5))
        audit_path = os.path.join(HERE, "runs", "test_loop_round2_audit.jsonl")
        if os.path.exists(audit_path):
            os.remove(audit_path)
        result = loop_mod.loop(
            failure_reports, range(15), audit_path,
            max_rounds=3, candidates_per_round=3, min_interval_s=0.0,
        )
        server_log = _fetch_log()
        return result, audit_path, server_log
    finally:
        os.environ.pop("FLEET_LLM_BASE_URL", None)
        _stop_server(server)


def test_feedback_reaches_second_request():
    _, _, server_log = _run_round1_insufficient_round2_strong()
    messages = server_log["messages"]
    if len(messages) < 4:
        return False, f"expected >= 4 requests (round 1's 3 + round 2's 1st), got {len(messages)}"
    # Request 4 is round 2's first candidate -- the first one that could
    # possibly carry feedback from round 1's rejections.
    last_user_msgs = [m for m in messages[3] if m.get("role") == "user"]
    if not last_user_msgs:
        return False, "no user message in request 4"
    content = last_user_msgs[-1].get("content", "")
    if "rejected by Check 1" not in content:
        return False, f"expected round 1's feedback text in request 4's user message; got {content!r}"
    # Requests 1-3 (round 1) must NOT have it -- there's no feedback yet.
    for i in range(3):
        first_user_msgs = [m for m in messages[i] if m.get("role") == "user"]
        if first_user_msgs and "rejected by Check 1" in first_user_msgs[-1].get("content", ""):
            return False, f"request {i + 1} (round 1) unexpectedly already carries feedback"
    return True, ""


def test_loop_returns_passing_rule_with_check1_passed():
    result, audit_path, _ = _run_round1_insufficient_round2_strong()
    if result.check1_passed_rule is None:
        return False, f"expected a CHECK1_PASSED rule in round 2; got NoSafeRule={result.no_safe_rule}"
    if not result.check1_passed_outcome.passed:
        return False, "check1_passed_rule's own outcome.passed is False"
    state = lifecycle.current_state(audit_path, result.check1_passed_rule.rule_id)
    if state != lifecycle.State.CHECK1_PASSED:
        return False, f"expected lifecycle state CHECK1_PASSED for {result.check1_passed_rule.rule_id}, got {state}"
    return True, ""


def test_lifecycle_logged_once_for_passing_rule():
    result, audit_path, _ = _run_round1_insufficient_round2_strong()
    if result.check1_passed_rule is None:
        return False, "no passing rule to check lifecycle logging for"
    rid = result.check1_passed_rule.rule_id
    events = [e for e in lifecycle.read_events(audit_path) if e.rule_id == rid]
    proposed = [e for e in events if e.event == "PROPOSED"]
    check1_passed = [e for e in events if e.event == "CHECK1_PASSED"]
    if len(proposed) != 1:
        return False, f"expected exactly 1 PROPOSED event for {rid}, got {len(proposed)}"
    if len(check1_passed) != 1:
        return False, f"expected exactly 1 CHECK1_PASSED event for {rid}, got {len(check1_passed)}"
    return True, ""


def test_selection_picks_largest_crush_margin():
    """Pure unit test of loop._pick_best_passing -- no server needed. Two
    hand-built passing candidates with different margins; the smaller
    grip_force_min_n (larger margin = 1 - grip/crush_limit) must win even
    though it is listed second and even though the other has a smaller
    slowdown_proxy (margin is the primary key, slowdown only breaks ties)."""
    def _make(rule_id, margin, slowdown_proxy):
        rec = loop_mod.AttemptRecord(round=1, candidate_index=1)
        rec.rule = _rule(rule_id=rule_id)
        rec.check1 = check1_mod.Check1Outcome(passed=True, margin=margin, slowdown_proxy=slowdown_proxy)
        return rec

    worse_margin_faster = _make("r_a", margin=0.04, slowdown_proxy=1.0)
    better_margin_slower = _make("r_b", margin=0.40, slowdown_proxy=2.0)
    best = loop_mod._pick_best_passing([worse_margin_faster, better_margin_slower])
    if best is None or best.rule.rule_id != "r_b":
        return False, f"expected r_b (margin 0.40) to win over r_a (margin 0.04); got {best.rule.rule_id if best else None}"

    # Tie-break: equal margin, smallest slowdown_proxy wins.
    tie_a = _make("r_c", margin=0.20, slowdown_proxy=3.0)
    tie_b = _make("r_d", margin=0.20, slowdown_proxy=1.5)
    best_tie = loop_mod._pick_best_passing([tie_a, tie_b])
    if best_tie is None or best_tie.rule.rule_id != "r_d":
        return False, f"expected r_d (slowdown 1.5) to win the margin tie over r_c (slowdown 3.0); got {best_tie.rule.rule_id if best_tie else None}"
    return True, ""


def test_sim_crush_limits_match_rules_limits():
    import cell_model  # sim/cell_model.py
    for cls in rules_limits.OBJECT_CLASSES:
        sim_limit = cell_model.OBJECT_CLASSES[cls]["crush_limit_n"]
        rules_limit = rules_limits.CRUSH_LIMIT_N[cls]
        if sim_limit != rules_limit:
            return False, f"{cls}: sim crush_limit_n={sim_limit} != rules.limits.CRUSH_LIMIT_N={rules_limit}"
    return True, ""


def test_no_api_key_in_loop_output():
    secret = "sk-test-SECRET-should-never-leak-99887766"
    server = subprocess.Popen(
        [sys.executable, "fake_server.py"], cwd=LLM_DIR,
        env={**os.environ, "FAKE_ALWAYS_GOOD": "1"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_port():
            return False, "fake_server.py never started"
        result = subprocess.run(
            [
                sys.executable, os.path.join(HERE, "loop.py"),
                # Speed knobs only -- FLEET_LLM_BASE_URL set below means
                # loop.py uses the server this test already started
                # instead of launching its own (and the production
                # 4.0s/100-seed defaults would make this leak-check slow).
                "--check1-seeds", "10", "--min-interval-s", "0.0",
            ],
            cwd=HERE, env={**os.environ, "FLEET_LLM_API_KEY": secret, "FLEET_LLM_BASE_URL": f"http://localhost:{PORT}/v1"},
            capture_output=True, text=True, timeout=60,
        )
        if secret in result.stdout or secret in result.stderr:
            return False, "API key leaked into pipeline/loop.py stdout/stderr"
        runs_dir = os.path.join(HERE, "runs")
        for name in os.listdir(runs_dir):
            path = os.path.join(runs_dir, name)
            if os.path.isfile(path):
                with open(path, errors="ignore") as f:
                    if secret in f.read():
                        return False, f"API key leaked into {path}"
        return True, ""
    finally:
        _stop_server(server)


def test_mutation_crush_check_catches_regression():
    """Mutation test (required by spec): flip the crush check and confirm
    a test that must catch crushing goes red, proving the gate is
    load-bearing, then restore both mutations.

    rules.validate()'s crush-limit check and rules.apply()'s own both read
    the SAME shared rules.limits.CRUSH_LIMIT_N dict (both do `from
    rules.limits import CRUSH_LIMIT_N`, binding the identical dict object),
    so mutating it in place fools both at once and lets an over-grip rule
    (30N for glass_jar, above the real 25N crush limit used independently
    by sim/cell_model.py) all the way through to a real simulator replay,
    where it genuinely crushes every trial.

    Note: run_trial.py's crush decision (`grip_force_n > crush_limit_n`) is
    a pure function of the controller params and object class, not of the
    per-trial seed -- so a crushing rule crushes EVERY trial, never some.
    A crushed trial never counts as "fixed" either (its outcome is
    "crush", not "ok"), so a crushing rule always independently fails the
    >= MIN_FIXED_RATE_CI95_LOWER gate too, and the aggregate `passed` boolean can
    never isolate the crush gate's own marginal effect (defeating it alone
    cannot flip `passed` from False to True here). This checks for the
    crush-specific reason string's presence/absence instead, which does
    isolate it.
    """
    CRUSH_REASON_MARKER = "crush(es) after the rule"
    original_crush = rules_limits.CRUSH_LIMIT_N["glass_jar"]
    rules_limits.CRUSH_LIMIT_N["glass_jar"] = 999.0
    try:
        rule = _rule(rule_id="r_crush_mutation", grip_force_min_n=30.0, carry_accel_max_ms2=1.0)
        real_outcome = check1_mod.check1(rule, range(10))
        if real_outcome.crush_count_after == 0:
            return False, (
                "expected grip=30N for glass_jar to crush every trial (real sim crush_limit_n="
                "25.0) even with rules.limits.CRUSH_LIMIT_N mutated to 999.0"
            )
        if not any(CRUSH_REASON_MARKER in r for r in real_outcome.reasons):
            return False, f"expected a crush-count failure reason; got {real_outcome.reasons}"

        original_max = check1_mod.MAX_CRUSH_COUNT
        check1_mod.MAX_CRUSH_COUNT = 10**6  # mutation: flip the crush check off
        try:
            mutated_outcome = check1_mod.check1(rule, range(10))
        finally:
            check1_mod.MAX_CRUSH_COUNT = original_max  # restore immediately

        if any(CRUSH_REASON_MARKER in r for r in mutated_outcome.reasons):
            return False, (
                "flipping MAX_CRUSH_COUNT did not remove the crush-count failure reason -- "
                "this mutation test is not actually exercising the crush_count_after gate"
            )
        return True, ""
    finally:
        rules_limits.CRUSH_LIMIT_N["glass_jar"] = original_crush


UNIT_TESTS = [
    ("a clearly good glass_jar rule passes check1", test_good_rule_passes),
    ("grip too low fails check1", test_grip_too_low_fails),
    ("check1's default train_seeds is 100 (0-99)", test_default_train_seeds_is_100),
    ("Wilson 95% lower bound gates pass, not the point estimate", test_wilson_lower_bound_gates_pass_not_point_estimate),
    ("grip above the crush limit fails check1", test_grip_above_crush_limit_fails),
    ("check1 is deterministic", test_determinism),
    ("check1 serial == parallel", test_serial_equals_parallel),
    ("evaluate_heldout.py refuses without --held-out", test_heldout_guard_refuses_without_flag),
    ("evaluate_heldout.py refuses when accepted_rule.json is missing", test_heldout_refuses_without_rule_json),
    ("evaluate_heldout.py shows both Cell C scenarios, ceiling row, stub label", test_heldout_shows_both_cell_c_scenarios_ceiling_row_and_stub_label),
    ("loop returns NoSafeRule for an unfixable seed set", test_loop_no_safe_rule_for_unfixable_seeds),
    ("feedback reaches the second request", test_feedback_reaches_second_request),
    ("loop returns the passing rule with CHECK1_PASSED", test_loop_returns_passing_rule_with_check1_passed),
    ("lifecycle events logged exactly once for the passing rule", test_lifecycle_logged_once_for_passing_rule),
    ("selection picks the largest crush margin", test_selection_picks_largest_crush_margin),
    ("sim crush limits match rules.limits.CRUSH_LIMIT_N", test_sim_crush_limits_match_rules_limits),
    ("no API key in any pipeline output", test_no_api_key_in_loop_output),
    ("mutation: flipping the crush check must fail a test, then restore", test_mutation_crush_check_catches_regression),
]


def run_unit(name, fn):
    try:
        result = fn()
        ok, reason = result if isinstance(result, tuple) else (result, "")
    except Exception as e:
        ok, reason = False, f"unexpected {type(e).__name__}: {e}"
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f" ({reason})"))
    return ok


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    results = [run_unit(name, fn) for name, fn in UNIT_TESTS]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
