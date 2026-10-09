"""Stub OpenAI-compatible server for testing solver.py without a GPU.

Env vars to force failure modes for testing:
  FAKE_ALWAYS_BAD=1     every reply is the bad (extra-field) one
  FAKE_HTTP_STATUS=500  respond with this HTTP status instead of 200
  FAKE_EMPTY_CONTENT=1  message.content is JSON null instead of real text
  FAKE_SLEEP_S=10       sleep this many seconds before replying
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

BAD_REPLY = '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_min_n":14,"carry_accel_max_ms2":2.5,"rationale":"slip before drop","source_report_ids":["fr_0001"],"confidence":0.9}'
GOOD_REPLY = '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_min_n":14,"carry_accel_max_ms2":2.5,"rationale":"slip before drop","source_report_ids":["fr_0001"]}'
REPLIES = [BAD_REPLY, GOOD_REPLY]

ALWAYS_BAD = os.environ.get("FAKE_ALWAYS_BAD") == "1"
HTTP_STATUS = int(os.environ.get("FAKE_HTTP_STATUS", "200"))
EMPTY_CONTENT = os.environ.get("FAKE_EMPTY_CONTENT") == "1"
SLEEP_S = float(os.environ.get("FAKE_SLEEP_S", "0"))

request_count = 0


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global request_count
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # request body unused by the stub
        request_count += 1

        if SLEEP_S > 0:
            time.sleep(SLEEP_S)

        if HTTP_STATUS != 200:
            payload = json.dumps({"error": "simulated server error"}).encode()
            self.send_response(HTTP_STATUS)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if EMPTY_CONTENT:
            content = None
        elif ALWAYS_BAD:
            content = BAD_REPLY
        else:
            content = REPLIES[min(request_count - 1, len(REPLIES) - 1)]

        body = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
        }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[fake_server] request #{request_count}: {fmt % args}")


if __name__ == "__main__":
    print(f"fake_server modes: ALWAYS_BAD={ALWAYS_BAD} HTTP_STATUS={HTTP_STATUS} EMPTY_CONTENT={EMPTY_CONTENT} SLEEP_S={SLEEP_S}")
    server = HTTPServer(("localhost", 8000), Handler)
    print("fake_server listening on http://localhost:8000/v1/chat/completions")
    server.serve_forever()
