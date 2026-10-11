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
]


if __name__ == "__main__":
    if port_is_open(PORT):
        print(f"ABORT: port {PORT} is already in use before tests started. Stop whatever is using it and retry.")
        sys.exit(2)

    results = [run_case(*case) for case in CASES]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
