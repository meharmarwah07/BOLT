"""Solver: failure reports -> Rule or NoSafeRule, via an OpenAI-compatible chat endpoint."""
import json
import os
import random
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
from rules.limits import (
    CARRY_ACCEL_MAX_MS2,
    CARRY_ACCEL_MIN_MS2,
    CRUSH_LIMIT_N,
    GRIP_FORCE_MAX_N,
    GRIP_FORCE_MIN_N,
    OBJECT_CLASSES,
)

BASE_URL = os.environ.get("FLEET_LLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("FLEET_LLM_MODEL", "stub-model")
API_KEY = os.environ.get("FLEET_LLM_API_KEY") or os.environ.get("RADEON_API_KEY") or "not-needed"
MAX_RETRIES = 2  # plus the first attempt = 3 tries total
REQUEST_TIMEOUT_S = float(os.environ.get("FLEET_LLM_TIMEOUT_S", "60"))
MAX_TOKENS = int(os.environ.get("FLEET_LLM_MAX_TOKENS", "512"))
MAX_TOKENS_CEILING = 2048
RETRY_SLEEP_S = float(os.environ.get("FLEET_LLM_RETRY_SLEEP_S", "1"))
REASONING_EFFORT = os.environ.get("FLEET_LLM_REASONING_EFFORT") or None

# 429 retry policy: sleep the server's Retry-After (capped), plus jitter, then retry.
RETRY_AFTER_DEFAULT_S = 60.0
RETRY_AFTER_CAP_S = 90.0
RETRY_AFTER_JITTER_MAX_S = 0.5

# Request-count / outcome counters for the current process, read by smoke_real.py.
_METRICS = {"http_requests": 0, "http_429s": 0, "finish_reasons": {}}


def reset_metrics():
    _METRICS["http_requests"] = 0
    _METRICS["http_429s"] = 0
    _METRICS["finish_reasons"] = {}


def get_metrics():
    return {
        "http_requests": _METRICS["http_requests"],
        "http_429s": _METRICS["http_429s"],
        "finish_reasons": dict(_METRICS["finish_reasons"]),
    }


# Per-attempt bookkeeping for the most recent solve() call, read by smoke_real.py.
# attempt_errors[i] is None if attempt i+1 succeeded, else its failure reason.
_LAST_CALL = {"attempts": 0, "attempt_errors": []}


def get_last_call_info():
    return {"attempts": _LAST_CALL["attempts"], "attempt_errors": list(_LAST_CALL["attempt_errors"])}


PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_SYSTEM_PROMPT_PATH = os.path.join(PROMPTS_DIR, "solver_system.txt")
_EXAMPLES_PATH = os.path.join(PROMPTS_DIR, "examples.json")


def _load_system_prompt():
    try:
        with open(_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            template = f.read()
    except OSError as e:
        raise RuntimeError(f"missing system prompt file: {_SYSTEM_PROMPT_PATH}") from e

    crush_limits = "\n".join(f"  {cls}: {limit} N" for cls, limit in CRUSH_LIMIT_N.items())
    return template.format(
        object_classes=", ".join(OBJECT_CLASSES),
        grip_min=GRIP_FORCE_MIN_N,
        grip_max=GRIP_FORCE_MAX_N,
        accel_min=CARRY_ACCEL_MIN_MS2,
        accel_max=CARRY_ACCEL_MAX_MS2,
        crush_limits=crush_limits,
    )


def _load_examples():
    try:
        with open(_EXAMPLES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        raise RuntimeError(f"missing examples file: {_EXAMPLES_PATH}") from e
    return data["examples"]


SYSTEM_PROMPT = _load_system_prompt()
EXAMPLES = _load_examples()


class _TransientRequestError(Exception):
    """A request-level failure worth retrying: timeout, connection error, 5xx, 429."""


class _RateLimitTransientError(_TransientRequestError):
    """A 429, carrying the server's requested backoff (seconds, already capped)."""

    def __init__(self, message, retry_after_s):
        super().__init__(message)
        self.retry_after_s = retry_after_s


def _retry_after_seconds(e):
    headers = e.response.headers if e.response is not None else None
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return RETRY_AFTER_DEFAULT_S
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return RETRY_AFTER_DEFAULT_S
    return min(max(seconds, 0.0), RETRY_AFTER_CAP_S)


def _call(client, messages, max_tokens):
    kwargs = {}
    if REASONING_EFFORT:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    _METRICS["http_requests"] += 1
    try:
        return client.chat.completions.create(model=MODEL, messages=messages, max_tokens=max_tokens, **kwargs)
    except RateLimitError as e:
        _METRICS["http_429s"] += 1
        raise _RateLimitTransientError(str(e), _retry_after_seconds(e)) from e
    except (APIConnectionError, InternalServerError) as e:
        raise _TransientRequestError(str(e)) from e
    except APIStatusError as e:
        body = e.response.text[:200] if e.response is not None else ""
        raise RuntimeError(f"solve() failed: HTTP {e.status_code} (not retried): {body}") from e


def solve(failure_reports):
    _LAST_CALL["attempts"] = 0
    _LAST_CALL["attempt_errors"] = []

    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT_S, max_retries=0)
    except Exception as e:
        raise RuntimeError(f"failed to construct OpenAI client: {e}") from e

    valid_report_ids = {r["report_id"] for r in failure_reports if "report_id" in r}
    expected_object_classes = {r["object_class"] for r in failure_reports if "object_class" in r}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in EXAMPLES:
        messages.append({"role": "user", "content": json.dumps(example["failure_reports"])})
        messages.append({"role": "assistant", "content": json.dumps(example["response"])})
    messages.append({"role": "user", "content": json.dumps(failure_reports)})

    max_tokens = MAX_TOKENS
    length_retry_used = False
    last_err = None

    for attempt in range(1 + MAX_RETRIES):
        _LAST_CALL["attempts"] = attempt + 1
        try:
            response = _call(client, messages, max_tokens)
        except _RateLimitTransientError as e:
            last_err = f"request failed: {e}"
            _LAST_CALL["attempt_errors"].append(last_err)
            if attempt < MAX_RETRIES:
                sleep_s = e.retry_after_s + random.uniform(0, RETRY_AFTER_JITTER_MAX_S)
                time.sleep(sleep_s)
            continue
        except _TransientRequestError as e:
            last_err = f"request failed: {e}"
            _LAST_CALL["attempt_errors"].append(last_err)
            if attempt < MAX_RETRIES and RETRY_SLEEP_S > 0:
                time.sleep(RETRY_SLEEP_S)
            continue
        except RuntimeError as e:
            _LAST_CALL["attempt_errors"].append(str(e))
            raise

        if not response.choices or response.choices[0].message is None:
            last_err = "malformed response: no choices"
            _LAST_CALL["attempt_errors"].append(last_err)
            continue

        choice = response.choices[0]
        _METRICS["finish_reasons"][choice.finish_reason] = _METRICS["finish_reasons"].get(choice.finish_reason, 0) + 1
        if choice.finish_reason == "length":
            if length_retry_used:
                msg = f"finish_reason=length twice; max_tokens limit used={max_tokens}"
                _LAST_CALL["attempt_errors"].append(msg)
                raise RuntimeError(f"solve() failed: {msg}")
            length_retry_used = True
            max_tokens = min(max_tokens * 2, MAX_TOKENS_CEILING)
            last_err = f"finish_reason=length (truncated; retrying with max_tokens={max_tokens})"
            _LAST_CALL["attempt_errors"].append(last_err)
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
            elif expected_object_classes and obj.object_class not in expected_object_classes:
                err = (
                    f"object_class mismatch: rule proposes {obj.object_class!r} but the "
                    f"source report(s) are {sorted(expected_object_classes)}"
                )
                obj = None

        if obj is not None:
            _LAST_CALL["attempt_errors"].append(None)
            return obj

        last_err = err
        _LAST_CALL["attempt_errors"].append(err)
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
