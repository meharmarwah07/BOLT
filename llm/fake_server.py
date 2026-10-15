"""Stub OpenAI-compatible server for testing solver.py without a GPU.

Env vars to force failure modes for testing:
  FAKE_ALWAYS_GOOD=1    every reply is the good one (no retry needed)
  FAKE_ALWAYS_BAD=1     every reply is the bad (extra-field) one
  FAKE_HTTP_STATUS=500  respond with this HTTP status instead of 200
  FAKE_EMPTY_CONTENT=1  message.content is JSON null instead of real text
  FAKE_SLEEP_S=10       sleep this many seconds before replying
  FAKE_SCRIPT=<json>    a JSON list of per-request step dicts (see below);
                        request N (1-based) uses step min(N, len(script)) - 1,
                        i.e. the last step repeats forever once exhausted.
                        Takes priority over all the flags above when set.

FAKE_SCRIPT step dict keys (all optional):
  status          HTTP status to send (default 200)
  retry_after     if status == 429, send this value as the Retry-After header
  finish_reason   "stop" or "length" (default "stop")
  content         "good" | "strong" | "bad" | "truncated" | "unknown_source" | "mismatched_class"
                  (default "good"). "good" is a well-formed but physically
                  insufficient glass_jar rule (14N/2.5 m/s^2 -- measured
                  insufficient for most glass_jar seeds, sim/STATUS.md).
                  "strong" is a well-formed one that is measured sufficient
                  to clear Check 1 v1 (20N/1.0 m/s^2 -- see
                  pipeline/check1.py's check1(..., version=1); it does NOT
                  clear v2's tighter absolute-drop-rate gate, since no
                  single (grip, accel) pair for glass_jar does without also
                  crushing some fraction of seeds under
                  sim/cell_model.py's CRUSH_LIMIT_JITTER -- see
                  pipeline/dev_check.py). A script like
                  [{"content":"good"}, {"content":"strong"}] lets a test
                  simulate round 1 proposing something too weak and round 2
                  (the request after feedback) proposing something that
                  actually passes.
  no_choices      true -> response body has "choices": [] (overrides content/finish_reason)

Default (no env vars): request 1 is bad, request 2+ is good (tests retry).

Every reply's rule_id is unique per request (f"r_fake_{request_count:04d}")
-- a real model can also repeat an id across samples/rounds, but this stub
should not do so by construction. source_report_ids/object_class are
echoed from the real request even when the caller appended extra plain
text after the JSON array (e.g. solver.solve's `feedback` argument): this
server parses only the LEADING JSON value of the last user message
(json.JSONDecoder.raw_decode), not the whole string, so trailing text
after the array no longer breaks the echo.

Every request (successful or not) is recorded; GET /__log__ returns
{"count": N, "max_tokens": [...], "messages": [...], "reasoning_effort": [...]}
-- the max_tokens, full "messages" list, and reasoning_effort (or None if
absent) sent on each request, in order -- so tests can assert exactly how
many requests were made, with what max_tokens, what message content, and
whether reasoning_effort was sent, without the test process needing to
share memory with this subprocess.
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

TRUNCATED_REPLY = '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_mi'
UNKNOWN_SOURCE_REPLY = '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_min_n":14,"carry_accel_max_ms2":2.5,"rationale":"slip before drop","source_report_ids":["fr_9999"]}'


def _good_reply(source_ids, rule_id):
    # Well-formed but measured INSUFFICIENT for most glass_jar seeds
    # (sim/STATUS.md: mean min holding grip ~17-20N). "good" historically
    # meant "parses and validates", not "actually fixes the drop".
    return json.dumps({
        "rule_id": rule_id, "object_class": "glass_jar", "grip_force_min_n": 14,
        "carry_accel_max_ms2": 2.5, "rationale": "slip before drop", "source_report_ids": source_ids,
    })


def _strong_reply(source_ids, rule_id):
    # Well-formed and measured sufficient to clear Check 1 v1 (fixes most
    # glass_jar seeds, Wilson 95% CI lower bound of fixed-rate >= 0.5).
    # Previously 24.0N; lowered to 20.0N after sim/cell_model.py's
    # CRUSH_LIMIT_JITTER was added (glass_jar's crush_limit_n is nominally
    # 25.0N but can jitter as low as 25.0*0.85=21.25N -- 24.0N sometimes
    # crushed, 20.0N never does). Does not clear v2 (see module docstring).
    return json.dumps({
        "rule_id": rule_id, "object_class": "glass_jar", "grip_force_min_n": 20.0,
        "carry_accel_max_ms2": 1.0, "rationale": "stronger grip, slower carry", "source_report_ids": source_ids,
    })


def _bad_reply(source_ids, rule_id):
    return json.dumps({
        "rule_id": rule_id, "object_class": "glass_jar", "grip_force_min_n": 14,
        "carry_accel_max_ms2": 2.5, "rationale": "slip before drop", "source_report_ids": source_ids,
        "confidence": 0.9,
    })


def _parse_leading_json(text):
    """Parse only the LEADING JSON value of `text`, ignoring anything after
    it. solver.solve's optional `feedback` argument appends plain text
    after the JSON-encoded failure reports in the same user message, so a
    plain json.loads() on the whole string would fail and silently lose
    the real report ids/object_class."""
    try:
        value, _end = json.JSONDecoder().raw_decode(text)
        return value
    except (json.JSONDecodeError, TypeError):
        return None


def _last_user_message_content(req_body):
    for msg in reversed(req_body.get("messages") or []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def _extract_report_ids(req_body):
    """Pull report_id values out of the most recent user message, so the
    stub's "good"/"bad" replies echo back a source_report_ids the caller
    will actually recognize (real callers don't all use "fr_0001")."""
    content = _last_user_message_content(req_body)
    if content is None:
        return None
    payload = _parse_leading_json(content)
    if isinstance(payload, list):
        ids = [item["report_id"] for item in payload if isinstance(item, dict) and "report_id" in item]
        return ids or None
    return None


def _extract_object_class(req_body):
    """Pull the object_class of the first report in the most recent user
    message, so "mismatched_class" can echo back something different."""
    content = _last_user_message_content(req_body)
    if content is None:
        return None
    payload = _parse_leading_json(content)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0].get("object_class")
    return None


def _mismatched_class_reply(source_ids, real_class, rule_id):
    wrong_class = "steel" if real_class != "steel" else "cardboard"
    return json.dumps({
        "rule_id": rule_id, "object_class": wrong_class, "grip_force_min_n": 14,
        "carry_accel_max_ms2": 2.5, "rationale": "slip before drop", "source_report_ids": source_ids,
    })


# Hygiene: tests bind port 0 (FAKE_SERVER_PORT=0) so an OS-assigned, always-
# free ephemeral port is used instead of a fixed one -- a fixed port
# collides with another concurrent test run (this suite or any other
# process) or a leftover orphaned server. Default 8000 preserves this
# script's own documented standalone behavior when run by hand. The actual
# bound port is written to FAKE_SERVER_PORT_FILE (if set) once bound, so a
# test harness started with port 0 can discover which port it actually got.
FAKE_SERVER_PORT = int(os.environ.get("FAKE_SERVER_PORT", "8000"))
FAKE_SERVER_PORT_FILE = os.environ.get("FAKE_SERVER_PORT_FILE")

ALWAYS_GOOD = os.environ.get("FAKE_ALWAYS_GOOD") == "1"
ALWAYS_BAD = os.environ.get("FAKE_ALWAYS_BAD") == "1"
HTTP_STATUS = int(os.environ.get("FAKE_HTTP_STATUS", "200"))
EMPTY_CONTENT = os.environ.get("FAKE_EMPTY_CONTENT") == "1"
SLEEP_S = float(os.environ.get("FAKE_SLEEP_S", "0"))
SCRIPT = json.loads(os.environ["FAKE_SCRIPT"]) if os.environ.get("FAKE_SCRIPT") else None

request_count = 0
request_log = []  # list of {"max_tokens": ...} per request, in order


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global request_count
        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length)
        request_count += 1

        try:
            req_body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            req_body = {}
        request_log.append({
            "max_tokens": req_body.get("max_tokens"),
            "messages": req_body.get("messages"),
            "reasoning_effort": req_body.get("reasoning_effort"),
        })
        source_ids = _extract_report_ids(req_body) or ["fr_0001"]
        real_class = _extract_object_class(req_body) or "glass_jar"
        rule_id = f"r_fake_{request_count:04d}"

        if SLEEP_S > 0:
            time.sleep(SLEEP_S)

        if SCRIPT is not None:
            step = SCRIPT[min(request_count - 1, len(SCRIPT) - 1)]
            self._respond_scripted(step, source_ids, real_class, rule_id)
            return

        if HTTP_STATUS != 200:
            headers = {"Retry-After": "60"} if HTTP_STATUS == 429 else None
            self._send_json(HTTP_STATUS, {"error": "simulated server error"}, extra_headers=headers)
            return

        if EMPTY_CONTENT:
            content = None
        elif ALWAYS_GOOD:
            content = _good_reply(source_ids, rule_id)
        elif ALWAYS_BAD:
            content = _bad_reply(source_ids, rule_id)
        else:
            content = _bad_reply(source_ids, rule_id) if request_count == 1 else _good_reply(source_ids, rule_id)

        self._send_json(200, self._completion_body(content, "stop"))

    def do_GET(self):
        if self.path == "/__log__":
            self._send_json(200, {
                "count": request_count,
                "max_tokens": [r["max_tokens"] for r in request_log],
                "messages": [r["messages"] for r in request_log],
                "reasoning_effort": [r["reasoning_effort"] for r in request_log],
            })
            return
        self.send_response(404)
        self.end_headers()

    def _respond_scripted(self, step, source_ids, real_class, rule_id):
        status = step.get("status", 200)
        if status != 200:
            headers = None
            if status == 429 and "retry_after" in step:
                headers = {"Retry-After": str(step["retry_after"])}
            self._send_json(status, {"error": "simulated server error"}, extra_headers=headers)
            return
        if step.get("no_choices"):
            self._send_json(200, {"id": "chatcmpl-fake", "object": "chat.completion", "choices": []})
            return
        name = step.get("content", "good")
        if name == "good":
            content = _good_reply(source_ids, rule_id)
        elif name == "strong":
            content = _strong_reply(source_ids, rule_id)
        elif name == "bad":
            content = _bad_reply(source_ids, rule_id)
        elif name == "truncated":
            content = TRUNCATED_REPLY
        elif name == "unknown_source":
            content = UNKNOWN_SOURCE_REPLY
        elif name == "mismatched_class":
            content = _mismatched_class_reply(source_ids, real_class, rule_id)
        else:
            raise ValueError(f"unknown FAKE_SCRIPT content name: {name!r}")
        finish_reason = step.get("finish_reason", "stop")
        self._send_json(200, self._completion_body(content, finish_reason))

    @staticmethod
    def _completion_body(content, finish_reason):
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
            ],
        }

    def _send_json(self, status, obj, extra_headers=None):
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[fake_server] request #{request_count}: {fmt % args}")


if __name__ == "__main__":
    print(f"fake_server modes: ALWAYS_GOOD={ALWAYS_GOOD} ALWAYS_BAD={ALWAYS_BAD} HTTP_STATUS={HTTP_STATUS} EMPTY_CONTENT={EMPTY_CONTENT} SLEEP_S={SLEEP_S} SCRIPT={SCRIPT}")
    server = HTTPServer(("localhost", FAKE_SERVER_PORT), Handler)
    actual_port = server.server_address[1]
    if FAKE_SERVER_PORT_FILE:
        with open(FAKE_SERVER_PORT_FILE, "w") as f:
            f.write(str(actual_port))
    print(f"fake_server listening on http://localhost:{actual_port}/v1/chat/completions")
    server.serve_forever()
