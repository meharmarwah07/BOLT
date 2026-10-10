"""DEPRECATED local copy of the Rule contract. Part C (fleet/rules) now owns
the real schema, limits, and validator -- this module is a thin shim that
delegates to it, kept only so solver.py / test_solver.py don't need to
change. See fleet/rules/INTEGRATION.md for every behavior difference and
what this shim papers over.
"""
import datetime as dt
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from rules.schema import Rule
from rules.validate import validate as rules_validate

# Part C's canonical Rule has fields (proposed_by, created_at, schema_version)
# that the Solver's prompt doesn't ask the model to emit yet. The shim fills
# them in with placeholders before validating, so Part B's existing prompt
# and replies keep working unchanged. See INTEGRATION.md.
_DEFAULT_PROPOSED_BY = "llm-solver"


class NoSafeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    no_safe_rule: Literal[True]
    reason: str


def parse_reply(text):
    """Turn raw LLM text into a Rule or NoSafeRule. Returns (obj, error)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"not valid JSON ({e.msg})"
    if not isinstance(data, dict):
        return None, "JSON is not an object"

    if "no_safe_rule" in data:
        try:
            return NoSafeRule.model_validate(data), None
        except ValidationError as e:
            first = e.errors()[0]
            return None, f"{first['loc']}: {first['msg']}"

    filled = dict(data)
    filled.setdefault("proposed_by", _DEFAULT_PROPOSED_BY)
    filled.setdefault("created_at", dt.datetime.now(dt.timezone.utc).isoformat())

    result = rules_validate(filled)
    if not result.ok:
        return None, result.errors[0]  # preserve the old first-error-only contract
    return Rule.model_validate(filled), None
