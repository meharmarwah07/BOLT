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
  content         "good" | "bad" | "truncated" | "unknown_source" (default "good")
  no_choices      true -> response body has "choices": [] (overrides content/finish_reason)

Default (no env vars): request 1 is bad, request 2+ is good (tests retry).

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


def _good_reply(source_ids):
    return json.dumps({
        "rule_id": "r_0001", "object_class": "glass_jar", "grip_force_min_n": 14,
        "carry_accel_max_ms2": 2.5, "rationale": "slip before drop", "source_report_ids": source_ids,
    })


def _bad_reply(source_ids):
    return json.dumps({
        "rule_id": "r_0001", "object_class": "glass_jar", "grip_force_min_n": 14,
        "carry_accel_max_ms2": 2.5, "rationale": "slip before drop", "source_report_ids": source_ids,
        "confidence": 0.9,
    })


def _extract_report_ids(req_body):
    """Pull report_id values out of the most recent user message, so the
    stub's "good"/"bad" replies echo back a source_report_ids the caller
    will actually recognize (real callers don't all use "fr_0001")."""
    for msg in reversed(req_body.get("messages") or []):
        if msg.get("role") != "user":
            continue
        try:
            payload = json.loads(msg.get("content", ""))
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(payload, list):
            ids = [item["report_id"] for item in payload if isinstance(item, dict) and "report_id" in item]
            return ids or None
        return None
    return None


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

        if SLEEP_S > 0:
            time.sleep(SLEEP_S)

        if SCRIPT is not None:
            step = SCRIPT[min(request_count - 1, len(SCRIPT) - 1)]
            self._respond_scripted(step, source_ids)
            return

        if HTTP_STATUS != 200:
            headers = {"Retry-After": "60"} if HTTP_STATUS == 429 else None
            self._send_json(HTTP_STATUS, {"error": "simulated server error"}, extra_headers=headers)
            return

        if EMPTY_CONTENT:
            content = None
        elif ALWAYS_GOOD:
            content = _good_reply(source_ids)
        elif ALWAYS_BAD:
            content = _bad_reply(source_ids)
        else:
            content = _bad_reply(source_ids) if request_count == 1 else _good_reply(source_ids)

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

    def _respond_scripted(self, step, source_ids):
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
            content = _good_reply(source_ids)
        elif name == "bad":
            content = _bad_reply(source_ids)
        elif name == "truncated":
            content = TRUNCATED_REPLY
        elif name == "unknown_source":
            content = UNKNOWN_SOURCE_REPLY
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
    server = HTTPServer(("localhost", 8000), Handler)
    print("fake_server listening on http://localhost:8000/v1/chat/completions")
    server.serve_forever()
