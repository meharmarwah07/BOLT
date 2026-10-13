"""Tests for smoke_real.py and its fixtures. Run with: python test_smoke.py"""
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(HERE, "fixtures")
RUNS_DIR = os.path.join(HERE, "runs")
PORT = 8000

CONTRACTS_V0_REPORT_FIELDS = {
    "report_id", "cell_id", "owner_id", "object_class", "controller",
    "outcome", "drop_phase", "slip_mm_before_drop", "est_mass_kg", "seed",
}
CONTRACTS_V0_CONTROLLER_FIELDS = {"grip_force_n", "carry_accel_ms2"}
HELD_OUT_SEEDS = range(1000, 1200)  # 1000..1199 inclusive
G = 9.81


def load_reports():
    with open(os.path.join(FIXTURES_DIR, "synthetic_reports.json"), "r", encoding="utf-8") as f:
        return json.load(f)["reports"]


def load_truth():
    with open(os.path.join(FIXTURES_DIR, "truth.json"), "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def run_check(name, fn):
    try:
        ok, reason = fn()
    except Exception as e:
        ok, reason = False, f"unexpected {type(e).__name__}: {e}"
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f" ({reason})"))
    return ok


def test_reports_match_contracts_v0_shape():
    reports = load_reports()
    if len(reports) != 18:
        return False, f"expected 18 reports, got {len(reports)}"
    for report in reports:
        keys = set(report.keys())
        if keys != CONTRACTS_V0_REPORT_FIELDS:
            return False, f"{report.get('report_id')}: fields {keys} != {CONTRACTS_V0_REPORT_FIELDS}"
        controller_keys = set(report["controller"].keys())
        if controller_keys != CONTRACTS_V0_CONTROLLER_FIELDS:
            return False, f"{report['report_id']}: controller fields {controller_keys} != {CONTRACTS_V0_CONTROLLER_FIELDS}"
        # True friction is hidden by contract -- it must never appear in a
        # report that gets sent to the model, under any spelling.
        for forbidden in ("mu", "hidden_mu", "friction", "friction_coeff", "friction_coefficient"):
            if forbidden in report or forbidden in report["controller"]:
                return False, f"{report['report_id']}: contains hidden-friction field {forbidden!r}"
    return True, ""


def test_report_ids_unique():
    reports = load_reports()
    ids = [r["report_id"] for r in reports]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        return False, f"duplicate report ids: {dupes}"
    return True, ""


def test_no_held_out_seeds():
    reports = load_reports()
    bad = [r["report_id"] for r in reports if r["seed"] in HELD_OUT_SEEDS]
    if bad:
        return False, f"reports using a held-out seed (1000-1199): {bad}"
    return True, ""


def test_truth_covers_every_report_and_recomputes():
    from rules.limits import CRUSH_LIMIT_N, GRIP_FORCE_MAX_N

    reports = load_reports()
    truth = load_truth()
    report_ids = {r["report_id"] for r in reports}
    truth_ids = set(truth.keys())
    if report_ids != truth_ids:
        return False, f"truth.json ids != report ids; missing={report_ids - truth_ids} extra={truth_ids - report_ids}"

    reports_by_id = {r["report_id"]: r for r in reports}
    for report_id, case in truth.items():
        report = reports_by_id[report_id]
        mu = case["hidden_mu"]
        m = report["est_mass_kg"]
        expected_floor = m * (G + 0.5) / (2 * mu)
        if abs(expected_floor - case["min_grip_at_accel_floor"]) > 0.01:
            return False, f"{report_id}: min_grip_at_accel_floor={case['min_grip_at_accel_floor']} != recomputed {expected_floor:.3f}"

        expected_feasible = expected_floor <= min(GRIP_FORCE_MAX_N, CRUSH_LIMIT_N[report["object_class"]])
        if expected_feasible != case["feasible"]:
            return False, f"{report_id}: feasible={case['feasible']} but formula gives {expected_feasible}"
    return True, ""


def test_scoring_function_hand_written_cases():
    import smoke_real
    from rule_check import NoSafeRule, Rule

    report = {"report_id": "t1", "est_mass_kg": 1.0, "object_class": "steel"}
    # required grip at mu=0.5, accel=2.0, m=1.0 kg: 1.0*(9.81+2.0)/(2*0.5) = 11.81 N
    truth_feasible = {"hidden_mu": 0.5, "feasible": True, "min_grip_at_accel_floor": 10.31}
    truth_infeasible = {"hidden_mu": 0.5, "feasible": False, "min_grip_at_accel_floor": 10.31}

    def make_rule(grip, accel):
        return Rule(
            rule_id="r_t", object_class="steel", grip_force_min_n=grip, carry_accel_max_ms2=accel,
            proposed_by="test", created_at="2026-01-01T00:00:00+00:00", rationale="x",
            source_report_ids=["t1"],
        )

    no_safe = NoSafeRule(no_safe_rule=True, reason="test")

    # (label, rule_or_none, truth_case, error_text, expected outcome, effective, refusal_correct)
    checks = [
        ("effective", make_rule(12.0, 2.0), truth_feasible, None, "valid_rule", True, None),
        ("ineffective", make_rule(5.0, 2.0), truth_feasible, None, "valid_rule", False, None),
        ("correct refusal", no_safe, truth_infeasible, None, "no_safe_rule", None, True),
        ("wrong refusal", no_safe, truth_feasible, None, "no_safe_rule", None, False),
        ("error", None, truth_feasible, "boom", "error", None, None),
    ]
    for label, rule_or_none, truth_case, error_text, exp_outcome, exp_effective, exp_refusal in checks:
        verdict = smoke_real.score_call(report, truth_case, rule_or_none, error_text)
        if verdict["outcome"] != exp_outcome:
            return False, f"{label}: outcome={verdict['outcome']!r}, expected {exp_outcome!r}"
        if verdict["effective"] != exp_effective:
            return False, f"{label}: effective={verdict['effective']!r}, expected {exp_effective!r}"
        if verdict["refusal_correct"] != exp_refusal:
            return False, f"{label}: refusal_correct={verdict['refusal_correct']!r}, expected {exp_refusal!r}"
    return True, ""


def port_is_open(port, host="localhost"):
    with socket.socket() as s:
        s.settimeout(0.2)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def wait_for_port(port, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        if port_is_open(port):
            return True
        time.sleep(0.05)
    return False


def wait_for_port_free(port, timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        if not port_is_open(port):
            return True
        time.sleep(0.05)
    return False


def test_smoke_real_end_to_end_against_stub():
    if port_is_open(PORT):
        return False, f"port {PORT} is already in use, cannot start the stub"

    server_env = {**os.environ, "FAKE_ALWAYS_GOOD": "1"}
    server = subprocess.Popen(
        [sys.executable, "fake_server.py"], cwd=HERE, env=server_env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_port(PORT):
            return False, "stub server never started"

        os.makedirs(RUNS_DIR, exist_ok=True)
        before = set(os.listdir(RUNS_DIR))

        solver_env = {**os.environ, "FLEET_LLM_RETRY_SLEEP_S": "0", "FLEET_MIN_INTERVAL_S": "0"}
        result = subprocess.run(
            [sys.executable, "smoke_real.py"], cwd=HERE, env=solver_env,
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return False, f"smoke_real.py exited {result.returncode}: {result.stderr.strip()[:300]!r}"

        new_files = set(os.listdir(RUNS_DIR)) - before
        if not new_files:
            return False, "smoke_real.py did not write a new file to llm/runs/"

        results_path = os.path.join(RUNS_DIR, sorted(new_files)[-1])
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data.get("records", [])) != 18:
            return False, f"expected 18 records in {results_path}, got {len(data.get('records', []))}"
        return True, ""
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        wait_for_port_free(PORT)


def test_run_metadata_and_no_api_key_leak():
    if port_is_open(PORT):
        return False, f"port {PORT} is already in use"
    secret = "sk-test-SECRET-should-never-leak-99887766"
    server_env = {**os.environ, "FAKE_ALWAYS_GOOD": "1"}
    server = subprocess.Popen(
        [sys.executable, "fake_server.py"], cwd=HERE, env=server_env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_port(PORT):
            return False, "stub server never started"

        os.makedirs(RUNS_DIR, exist_ok=True)
        before = set(os.listdir(RUNS_DIR))

        solver_env = {
            **os.environ,
            "FLEET_LLM_RETRY_SLEEP_S": "0",
            "FLEET_MIN_INTERVAL_S": "0",
            "FLEET_LLM_API_KEY": secret,
            "FLEET_LLM_REASONING_EFFORT": "medium",
        }
        result = subprocess.run(
            [sys.executable, "smoke_real.py"], cwd=HERE, env=solver_env,
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return False, f"smoke_real.py exited {result.returncode}: {result.stderr.strip()[:300]!r}"
        if secret in result.stdout or secret in result.stderr:
            return False, "API key leaked into smoke_real.py stdout/stderr"

        new_files = set(os.listdir(RUNS_DIR)) - before
        if not new_files:
            return False, "smoke_real.py did not write a new file to llm/runs/"
        results_path = os.path.join(RUNS_DIR, sorted(new_files)[-1])
        with open(results_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        if secret in raw_text:
            return False, f"API key leaked into {results_path}"
        data = json.loads(raw_text)

        for field in ("model", "base_url", "utc_date", "reasoning_effort",
                      "http_requests_made", "http_429_count", "finish_reason_counts"):
            if field not in data:
                return False, f"{field!r} missing from {results_path}"
        if data["reasoning_effort"] != "medium":
            return False, f"expected reasoning_effort='medium', got {data['reasoning_effort']!r}"
        if data["http_requests_made"] < len(data.get("records", [])):
            return False, (
                f"http_requests_made={data['http_requests_made']} is less than "
                f"number of records={len(data.get('records', []))}"
            )
        return True, ""
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        wait_for_port_free(PORT)


def test_min_interval_enforced_between_requests():
    if port_is_open(PORT):
        return False, f"port {PORT} is already in use"
    server_env = {**os.environ, "FAKE_ALWAYS_GOOD": "1"}
    server = subprocess.Popen(
        [sys.executable, "fake_server.py"], cwd=HERE, env=server_env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_port(PORT):
            return False, "stub server never started"

        fast_env = {**os.environ, "FLEET_LLM_RETRY_SLEEP_S": "0", "FLEET_MIN_INTERVAL_S": "0"}
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "smoke_real.py"], cwd=HERE, env=fast_env,
            capture_output=True, text=True, timeout=60,
        )
        fast_elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return False, f"(fast) exited {result.returncode}: {result.stderr.strip()[:300]!r}"

        slow_env = {**os.environ, "FLEET_LLM_RETRY_SLEEP_S": "0", "FLEET_MIN_INTERVAL_S": "0.2"}
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "smoke_real.py"], cwd=HERE, env=slow_env,
            capture_output=True, text=True, timeout=60,
        )
        slow_elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return False, f"(throttled) exited {result.returncode}: {result.stderr.strip()[:300]!r}"

        expected_min_gap = 17 * 0.2  # 18 fixture reports -> 17 gaps between request starts
        if (slow_elapsed - fast_elapsed) < expected_min_gap * 0.8:
            return False, (
                f"expected throttled run to take >= {expected_min_gap * 0.8:.2f}s longer than unthrottled "
                f"(fast={fast_elapsed:.2f}s, slow={slow_elapsed:.2f}s)"
            )
        return True, ""
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        wait_for_port_free(PORT)


TESTS = [
    ("fixtures match contracts-v0 shape, no hidden friction field", test_reports_match_contracts_v0_shape),
    ("fixture report ids are unique", test_report_ids_unique),
    ("no fixture uses a held-out seed (1000-1199)", test_no_held_out_seeds),
    ("truth.json covers every report id and recomputes correctly", test_truth_covers_every_report_and_recomputes),
    ("score_call gives the right verdict on 5 hand-written cases", test_scoring_function_hand_written_cases),
    ("smoke_real.py runs end to end against the stub and writes a results file", test_smoke_real_end_to_end_against_stub),
    ("runs/*.json carries model/base_url/date/reasoning_effort/counts, no API key leak", test_run_metadata_and_no_api_key_leak),
    ("FLEET_MIN_INTERVAL_S throttles request starts", test_min_interval_enforced_between_requests),
]


if __name__ == "__main__":
    results = [run_check(name, fn) for name, fn in TESTS]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
