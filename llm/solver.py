"""Solver: failure reports -> Rule or NoSafeRule, via an OpenAI-compatible chat endpoint."""
import json
import os
import sys

from openai import OpenAI
from rule_check import parse_reply

BASE_URL = os.environ.get("FLEET_LLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("FLEET_LLM_MODEL", "stub-model")
MAX_RETRIES = 2  # plus the first attempt = 3 tries total
REQUEST_TIMEOUT_S = float(os.environ.get("FLEET_LLM_TIMEOUT_S", "60"))

SYSTEM_PROMPT = (
    "You output ONLY a single JSON object. No prose, no markdown fences. "
    "Given failure reports, propose a Rule with fields: rule_id, object_class, "
    "grip_force_min_n, carry_accel_max_ms2, rationale, source_report_ids. "
    "If no safe rule exists, output {\"no_safe_rule\": true, \"reason\": \"...\"} instead."
)


def solve(failure_reports):
    try:
        client = OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=REQUEST_TIMEOUT_S, max_retries=0)
    except Exception as e:
        raise RuntimeError(f"failed to construct OpenAI client: {e}") from e

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(failure_reports)},
    ]

    last_err = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = client.chat.completions.create(model=MODEL, messages=messages)
        except Exception as e:
            last_err = f"request failed: {e}"
            continue

        reply_text = response.choices[0].message.content or ""
        if not reply_text.strip():
            obj, err = None, "empty or missing message content"
        else:
            obj, err = parse_reply(reply_text)

        if obj is not None:
            return obj

        last_err = err
        messages.append({"role": "assistant", "content": reply_text})
        messages.append({"role": "user", "content": f"That reply was rejected: {err}. Reply again with ONLY the corrected JSON."})

    raise RuntimeError(f"solve() failed after {1 + MAX_RETRIES} attempts; last error: {last_err}")


if __name__ == "__main__":
    reports = [{
        "report_id": "fr_0001", "cell_id": "A", "owner_id": "owner1",
        "object_class": "glass_jar",
        "controller": {"grip_force_n": 8.0, "carry_accel_ms2": 3.0},
        "outcome": "drop", "drop_phase": "carry",
        "slip_mm_before_drop": 12.4, "est_mass_kg": 0.35, "seed": 42,
    }]
    try:
        result = solve(reports)
        print(type(result).__name__, result)
    except Exception as e:
        print(f"solve() failed: {e}", file=sys.stderr)
        sys.exit(1)
