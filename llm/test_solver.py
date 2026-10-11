"""Regression tests for solver.py against fake_server.py.

Each case starts a fresh fake_server.py subprocess in a given failure mode,
runs solver.py as a subprocess against it, and checks the result, including
the exact number of HTTP requests the server received (via the server's
GET /__log__ endpoint). Run with:
    python test_solver.py
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8000


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


def fetch_log():
    with urllib.request.urlopen(f"http://localhost:{PORT}/__log__", timeout=2) as resp:
        return json.loads(resp.read())


def run_case(name, start_server, server_env, solver_env, check, expected_requests=None, extra_check=None):
    server = None
    if start_server:
        env = {**os.environ, **server_env}
        server = subprocess.Popen(
            [sys.executable, "fake_server.py"], cwd=HERE, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not wait_for_port(PORT):
            print(f"FAIL {name}: stub server never started")
            server.terminate()
            return False
    else:
        if not wait_for_port_free(PORT):
            print(f"FAIL {name}: port {PORT} is unexpectedly occupied, cannot test connection-refused")
            return False

    try:
        env = {**os.environ, **solver_env, "FLEET_LLM_RETRY_SLEEP_S": "0"}
        result = subprocess.run(
            [sys.executable, "solver.py"], cwd=HERE, env=env,
            capture_output=True, text=True, timeout=30,
        )
        ok, reason = check(result)

        log = None
        if ok and start_server:
            try:
                log = fetch_log()
            except Exception as e:
                ok, reason = False, f"could not fetch request log: {e}"

        if ok and start_server and expected_requests is not None and log["count"] != expected_requests:
            ok, reason = False, f"expected {expected_requests} requests, server saw {log['count']}"

        if ok and extra_check is not None:
            ok, reason = extra_check(log)

        print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f" ({reason})"))
        if not ok:
            print(f"     stdout: {result.stdout.strip()!r}")
            print(f"     stderr: {result.stderr.strip()!r}")
        return ok
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            wait_for_port_free(PORT)


def accepted(result):
    if result.returncode != 0:
        return False, f"returncode={result.returncode}"
    if "Rule " not in result.stdout:
        return False, "no Rule in stdout"
    return True, ""


def rejected(expect_substring):
    def check(result):
        if result.returncode != 1:
            return False, f"returncode={result.returncode}"
        if expect_substring not in result.stderr:
            return False, f"expected {expect_substring!r} in stderr"
        return True, ""
    return check


def second_max_tokens_larger(log):
    mt = log["max_tokens"]
    if len(mt) < 2 or mt[0] is None or mt[1] is None or not (mt[1] > mt[0]):
        return False, f"expected request 2's max_tokens > request 1's, got {mt}"
    return True, ""


def script_env(steps):
    return {"FAKE_SCRIPT": json.dumps(steps)}


def prompt_and_examples_reach_server(log):
    import solver as solver_mod

    requests = log["messages"]
    if not requests or not requests[0]:
        return False, "no messages recorded on first request"
    msgs = requests[0]
    if not msgs or msgs[0].get("role") != "system":
        return False, "first message sent to the server is not role=system"
    if "no_safe_rule" not in msgs[0].get("content", ""):
        return False, "system message sent to the server does not mention no_safe_rule"

    expected_len = 1 + 2 * len(solver_mod.EXAMPLES) + 1
    if len(msgs) != expected_len:
        return False, f"expected {expected_len} messages (system + {len(solver_mod.EXAMPLES)} examples + real request), got {len(msgs)}"

    for i, ex in enumerate(solver_mod.EXAMPLES):
        user_msg, assistant_msg = msgs[1 + 2 * i], msgs[2 + 2 * i]
        if user_msg.get("role") != "user" or json.loads(user_msg["content"]) != ex["failure_reports"]:
            return False, f"example {i} user message does not match prompts/examples.json"
        if assistant_msg.get("role") != "assistant" or json.loads(assistant_msg["content"]) != ex["response"]:
            return False, f"example {i} assistant message does not match prompts/examples.json"
    return True, ""


CASES = [
    ("good reply (no retry)", True, {"FAKE_ALWAYS_GOOD": "1"}, {}, accepted, 1, None),
    ("bad-then-good (one retry)", True, {}, {}, accepted, 2, None),
    ("all 3 attempts bad (exhausted retries)", True, {"FAKE_ALWAYS_BAD": "1"}, {}, rejected("Extra inputs are not permitted"), 3, None),
    ("server down (connection refused)", False, {}, {}, rejected("request failed"), None, None),
    ("HTTP 500", True, {"FAKE_HTTP_STATUS": "500"}, {}, rejected("500"), 3, None),
    ("empty/missing content", True, {"FAKE_EMPTY_CONTENT": "1"}, {}, rejected("empty or missing message content"), 3, None),

    ("length-then-ok", True,
     script_env([{"finish_reason": "length", "content": "truncated"}, {"finish_reason": "stop", "content": "good"}]),
     {}, accepted, 2, second_max_tokens_larger),

    ("length-twice", True,
     script_env([{"finish_reason": "length", "content": "truncated"}]),
     {}, rejected("finish_reason=length"), 2, None),

    ("HTTP 400", True, script_env([{"status": 400}]), {}, rejected("HTTP 400"), 1, None),

    ("HTTP 429-then-ok", True,
     script_env([{"status": 429}, {"finish_reason": "stop", "content": "good"}]),
     {}, accepted, 2, None),

    ("missing choices then ok", True,
     script_env([{"no_choices": True}, {"finish_reason": "stop", "content": "good"}]),
     {}, accepted, 2, None),

    ("unknown source id then ok", True,
     script_env([{"finish_reason": "stop", "content": "unknown_source"}, {"finish_reason": "stop", "content": "good"}]),
     {}, accepted, 2, None),

    ("unknown source id three times", True,
     script_env([{"finish_reason": "stop", "content": "unknown_source"}]),
     {}, rejected("unknown source_report_ids"), 3, None),

    ("prompt includes system message and examples", True, {"FAKE_ALWAYS_GOOD": "1"}, {}, accepted, 1, prompt_and_examples_reach_server),
]


def run_unit(name, fn):
    try:
        ok, reason = fn()
    except Exception as e:
        ok, reason = False, f"unexpected {type(e).__name__}: {e}"
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f" ({reason})"))
    return ok


def test_examples_pass_real_validator():
    import solver as solver_mod
    from rule_check import Rule, parse_reply

    for example in solver_mod.EXAMPLES:
        response = example["response"]
        if "no_safe_rule" in response:
            continue
        obj, err = parse_reply(json.dumps(response))
        if obj is None or not isinstance(obj, Rule):
            return False, f"example {response.get('rule_id')!r} failed rule_check.parse_reply: {err}"
    return True, ""


def test_system_prompt_nonempty_and_mentions_no_safe_rule():
    import solver as solver_mod

    if not solver_mod.SYSTEM_PROMPT.strip():
        return False, "system prompt file is empty"
    if "no_safe_rule" not in solver_mod.SYSTEM_PROMPT:
        return False, "system prompt does not mention no_safe_rule"
    return True, ""


def test_system_prompt_limits_match_rules_limits():
    import solver as solver_mod
    from rules.limits import (
        CARRY_ACCEL_MAX_MS2,
        CARRY_ACCEL_MIN_MS2,
        CRUSH_LIMIT_N,
        GRIP_FORCE_MAX_N,
        GRIP_FORCE_MIN_N,
    )

    text = solver_mod.SYSTEM_PROMPT
    for value in (GRIP_FORCE_MIN_N, GRIP_FORCE_MAX_N, CARRY_ACCEL_MIN_MS2, CARRY_ACCEL_MAX_MS2):
        if str(value) not in text:
            return False, f"limit {value} from rules.limits not found in the loaded prompt text"
    for object_class, limit in CRUSH_LIMIT_N.items():
        if object_class not in text or str(limit) not in text:
            return False, f"crush limit for {object_class!r} ({limit}) not found in the loaded prompt text"
    return True, ""


def test_examples_exclude_glass_jar():
    import solver as solver_mod

    # glass_jar is the real demo's hidden-friction failure class (CONTEXT.md).
    # A few-shot example using it risks anchoring the model on that exact
    # answer instead of reasoning from the evidence it's actually given, so
    # no example (report or response) may use object_class "glass_jar".
    for example in solver_mod.EXAMPLES:
        for report in example["failure_reports"]:
            if report.get("object_class") == "glass_jar":
                return False, f"example report {report.get('report_id')!r} uses object_class=glass_jar"
        if example["response"].get("object_class") == "glass_jar":
            return False, "example response uses object_class=glass_jar"
    return True, ""


UNIT_TESTS = [
    ("examples.json Rules pass rule_check.parse_reply", test_examples_pass_real_validator),
    ("system prompt non-empty and mentions no_safe_rule", test_system_prompt_nonempty_and_mentions_no_safe_rule),
    ("system prompt limits match rules/limits.py", test_system_prompt_limits_match_rules_limits),
    ("no example uses object_class=glass_jar (anchoring risk)", test_examples_exclude_glass_jar),
]


if __name__ == "__main__":
    if port_is_open(PORT):
        print(f"ABORT: port {PORT} is already in use before tests started. Stop whatever is using it and retry.")
        sys.exit(2)

    results = [run_case(*case) for case in CASES]
    results += [run_unit(name, fn) for name, fn in UNIT_TESTS]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
