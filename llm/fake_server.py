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
  finish_reason   "stop" or "length" (default "stop")
  content         "good" | "bad" | "truncated" | "unknown_source" (default "good")
  no_choices      true -> response body has "choices": [] (overrides content/finish_reason)

Default (no env vars): request 1 is bad, request 2+ is good (tests retry).

Every request (successful or not) is recorded; GET /__log__ returns
{"count": N, "max_tokens": [...], "messages": [...]}  -- the max_tokens and
full "messages" list sent on each request, in order -- so tests can assert
exactly how many requests were made, with what max_tokens, and with what
message content, without the test process needing to share memory with
this subprocess.
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

BAD_REPLY = '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_min_n":14,"carry_accel_max_ms2":2.5,"rationale":"slip before drop","source_report_ids":["fr_0001"],"confidence":0.9}'
GOOD_REPLY = '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_min_n":14,"carry_accel_max_ms2":2.5,"rationale":"slip before drop","source_report_ids":["fr_0001"]}'
TRUNCATED_REPLY = '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_mi'
UNKNOWN_SOURCE_REPLY = '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_min_n":14,"carry_accel_max_ms2":2.5,"rationale":"slip before drop","source_report_ids":["fr_9999"]}'
REPLIES = [BAD_REPLY, GOOD_REPLY]
CONTENT_BY_NAME = {"good": GOOD_REPLY, "bad": BAD_REPLY, "truncated": TRUNCATED_REPLY, "unknown_source": UNKNOWN_SOURCE_REPLY}

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
        request_log.append({"max_tokens": req_body.get("max_tokens"), "messages": req_body.get("messages")})

        if SLEEP_S > 0:
            time.sleep(SLEEP_S)

        if SCRIPT is not None:
            step = SCRIPT[min(request_count - 1, len(SCRIPT) - 1)]
            self._respond_scripted(step)
            return

        if HTTP_STATUS != 200:
            self._send_json(HTTP_STATUS, {"error": "simulated server error"})
            return

        if EMPTY_CONTENT:
            content = None
        elif ALWAYS_GOOD:
            content = GOOD_REPLY
        elif ALWAYS_BAD:
            content = BAD_REPLY
        else:
            content = REPLIES[min(request_count - 1, len(REPLIES) - 1)]

        self._send_json(200, self._completion_body(content, "stop"))

    def do_GET(self):
        if self.path == "/__log__":
            self._send_json(200, {
                "count": request_count,
                "max_tokens": [r["max_tokens"] for r in request_log],
                "messages": [r["messages"] for r in request_log],
            })
            return
        self.send_response(404)
        self.end_headers()

    def _respond_scripted(self, step):
        status = step.get("status", 200)
        if status != 200:
            self._send_json(status, {"error": "simulated server error"})
            return
        if step.get("no_choices"):
            self._send_json(200, {"id": "chatcmpl-fake", "object": "chat.completion", "choices": []})
            return
        content = CONTENT_BY_NAME[step.get("content", "good")]
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

    def _send_json(self, status, obj):
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[fake_server] request #{request_count}: {fmt % args}")


if __name__ == "__main__":
    print(f"fake_server modes: ALWAYS_GOOD={ALWAYS_GOOD} ALWAYS_BAD={ALWAYS_BAD} HTTP_STATUS={HTTP_STATUS} EMPTY_CONTENT={EMPTY_CONTENT} SLEEP_S={SLEEP_S} SCRIPT={SCRIPT}")
    server = HTTPServer(("localhost", 8000), Handler)
    print("fake_server listening on http://localhost:8000/v1/chat/completions")
    server.serve_forever()
