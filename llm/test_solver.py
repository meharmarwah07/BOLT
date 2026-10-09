"""Regression tests for solver.py against fake_server.py.

Each case starts a fresh fake_server.py subprocess in a given failure mode,
runs solver.py as a subprocess against it, and checks the result. Run with:
    python test_solver.py
"""
import os
import socket
import subprocess
import sys
import time

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


def run_case(name, start_server, server_env, solver_env, check):
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
        env = {**os.environ, **solver_env}
        result = subprocess.run(
            [sys.executable, "solver.py"], cwd=HERE, env=env,
            capture_output=True, text=True, timeout=30,
        )
        ok, reason = check(result)
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


CASES = [
    ("good reply (no retry)", True, {"FAKE_ALWAYS_GOOD": "1"}, {}, accepted),
    ("bad-then-good (one retry)", True, {}, {}, accepted),
    ("all 3 attempts bad (exhausted retries)", True, {"FAKE_ALWAYS_BAD": "1"}, {}, rejected("Extra inputs are not permitted")),
    ("server down (connection refused)", False, {}, {}, rejected("request failed")),
    ("HTTP 500", True, {"FAKE_HTTP_STATUS": "500"}, {}, rejected("500")),
    ("empty/missing content", True, {"FAKE_EMPTY_CONTENT": "1"}, {}, rejected("empty or missing message content")),
]


if __name__ == "__main__":
    if port_is_open(PORT):
        print(f"ABORT: port {PORT} is already in use before tests started. Stop whatever is using it and retry.")
        sys.exit(2)

    results = [run_case(*case) for case in CASES]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
