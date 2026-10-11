"""Solver: failure reports -> Rule or NoSafeRule, via an OpenAI-compatible chat endpoint."""
import json
import os
import sys
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from rule_check import Rule, parse_reply

BASE_URL = os.environ.get("FLEET_LLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("FLEET_LLM_MODEL", "stub-model")
MAX_RETRIES = 2  # plus the first attempt = 3 tries total
REQUEST_TIMEOUT_S = float(os.environ.get("FLEET_LLM_TIMEOUT_S", "60"))
MAX_TOKENS = int(os.environ.get("FLEET_LLM_MAX_TOKENS", "512"))
MAX_TOKENS_CEILING = 2048
RETRY_SLEEP_S = float(os.environ.get("FLEET_LLM_RETRY_SLEEP_S", "1"))

SYSTEM_PROMPT = (
    "You output ONLY a single JSON object. No prose, no markdown fences. "
    "Given failure reports, propose a Rule with fields: rule_id, object_class, "
    "grip_force_min_n, carry_accel_max_ms2, rationale, source_report_ids. "
    "If no safe rule exists, output {\"no_safe_rule\": true, \"reason\": \"...\"} instead."
)


class _TransientRequestError(Exception):
    """A request-level failure worth retrying: timeout, connection error, 5xx, 429."""


def _call(client, messages, max_tokens):
    try:
        return client.chat.completions.create(model=MODEL, messages=messages, max_tokens=max_tokens)
    except (APIConnectionError, RateLimitError, InternalServerError) as e:
        raise _TransientRequestError(str(e)) from e
    except APIStatusError as e:
        body = e.response.text[:200] if e.response is not None else ""
        raise RuntimeError(f"solve() failed: HTTP {e.status_code} (not retried): {body}") from e


def solve(failure_reports):
    try:
        client = OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=REQUEST_TIMEOUT_S, max_retries=0)
    except Exception as e:
        raise RuntimeError(f"failed to construct OpenAI client: {e}") from e

    valid_report_ids = {r["report_id"] for r in failure_reports if "report_id" in r}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(failure_reports)},
    ]

    max_tokens = MAX_TOKENS
    length_retry_used = False
    last_err = None

    for attempt in range(1 + MAX_RETRIES):
        try:
            response = _call(client, messages, max_tokens)
        except _TransientRequestError as e:
            last_err = f"request failed: {e}"
            if attempt < MAX_RETRIES and RETRY_SLEEP_S > 0:
                time.sleep(RETRY_SLEEP_S)
            continue

        if not response.choices or response.choices[0].message is None:
            last_err = "malformed response: no choices"
            continue

        choice = response.choices[0]
        if choice.finish_reason == "length":
            if length_retry_used:
                raise RuntimeError(
                    f"solve() failed: finish_reason=length twice; max_tokens limit used={max_tokens}"
                )
            length_retry_used = True
            max_tokens = min(max_tokens * 2, MAX_TOKENS_CEILING)
            last_err = f"finish_reason=length (truncated; retrying with max_tokens={max_tokens})"
            continue  # do not feed the truncated text back

        reply_text = choice.message.content or ""
        if not reply_text.strip():
            obj, err = None, "empty or missing message content"
        else:
            obj, err = parse_reply(reply_text)

        if obj is not None and isinstance(obj, Rule):
            unknown = sorted(set(obj.source_report_ids) - valid_report_ids)
            if unknown:
                obj = None
                err = (
                    f"unknown source_report_ids: {unknown}; "
                    f"valid ids are: {sorted(valid_report_ids)}"
                )

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
