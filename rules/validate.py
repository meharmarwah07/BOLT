"""validate(rule) -> ValidationResult.

Structural checks (missing/extra fields, wrong types, unknown object_class,
bad schema_version shape) come from pydantic via schema.Rule. Semantic checks
(numeric ranges, NaN/inf, crush limit, unsupported schema_version value) are
done by hand here so every error message can name the field, the bad value,
the allowed range, and a concrete fix -- the kind of message Part B's solver
pastes straight into the next LLM prompt.

All errors across both passes are collected before returning; validate()
never stops at the first problem.
"""
import math
from dataclasses import dataclass, field

from pydantic import ValidationError

from rules.limits import (
    CARRY_ACCEL_MAX_MS2,
    CARRY_ACCEL_MIN_MS2,
    CRUSH_LIMIT_N,
    GRIP_FORCE_MAX_N,
    GRIP_FORCE_MIN_N,
    OBJECT_CLASSES,
)
from rules.schema import SCHEMA_VERSION, Rule

# Fields pydantic validates structurally; semantic checks below only run on
# a field if it survived structural validation (so we don't, e.g., compare
# a missing or non-numeric grip_force_min_n to a range).
_NUMERIC_FIELDS = ("grip_force_min_n", "carry_accel_max_ms2")


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    @property
    def error_text(self) -> str:
        return "; ".join(self.errors)


def _structural_errors(data: dict) -> tuple[list[str], dict]:
    """Run schema.Rule over data. Returns (friendly_error_messages, raw_data).

    raw_data is the original dict, used for semantic checks below so they can
    see fields pydantic may have rejected or coerced.
    """
    try:
        Rule.model_validate(data)
        return [], data
    except ValidationError as e:
        messages = []
        for err in e.errors():
            messages.append(_friendly_structural_message(err, data))
        return messages, data


def _friendly_structural_message(err: dict, data: dict) -> str:
    loc = ".".join(str(p) for p in err["loc"]) if err["loc"] else "<root>"
    etype = err["type"]

    if etype == "extra_forbidden":
        return f"{loc}: Extra inputs are not permitted; a Rule must only contain the fields defined in rules.schema.Rule"
    if etype == "missing":
        return f"{loc}: missing required field"
    if etype in ("string_too_long",):
        value = data.get(err["loc"][0]) if err["loc"] else None
        return f"{loc}: value too long (max 300 chars); got {len(value) if isinstance(value, str) else '?'} chars"
    if etype in ("too_short",) and err["loc"] and err["loc"][0] == "source_report_ids":
        return f"{loc}: must contain at least 1 report id, got an empty list"
    if etype == "enum" and err["loc"] and err["loc"][0] == "object_class":
        bad = data.get("object_class")
        return (
            f"object_class: unknown value {bad!r}; must be one of "
            f"{list(OBJECT_CLASSES)} -- choose one of these"
        )
    if "float_parsing" in etype or "int_parsing" in etype or etype == "float_type":
        bad = _dig(data, err["loc"])
        return f"{loc}: expected a number, got {bad!r} ({type(bad).__name__})"
    if "datetime" in etype:
        bad = _dig(data, err["loc"])
        return f"{loc}: expected an ISO-8601 datetime, got {bad!r}"
    if etype in ("string_type",):
        bad = _dig(data, err["loc"])
        return f"{loc}: expected a string, got {bad!r} ({type(bad).__name__})"
    if etype in ("list_type",):
        bad = _dig(data, err["loc"])
        return f"{loc}: expected a list, got {bad!r} ({type(bad).__name__})"

    # Fallback: keep pydantic's own message, still prefixed with the field.
    return f"{loc}: {err['msg']}"


def _dig(data: dict, loc: tuple):
    cur = data
    for key in loc:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def _semantic_errors(data: dict) -> list[str]:
    errors = []

    schema_version = data.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        errors.append(
            f"schema_version: unsupported value {schema_version!r}; "
            f"this system only supports schema_version={SCHEMA_VERSION!r} -- "
            f"set schema_version to {SCHEMA_VERSION!r}"
        )

    object_class = data.get("object_class")
    class_known = object_class in OBJECT_CLASSES

    for field_name in _NUMERIC_FIELDS:
        if field_name not in data:
            continue  # already reported as a missing field structurally
        value = data[field_name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue  # already reported as a type error structurally
        if math.isnan(value) or math.isinf(value):
            errors.append(f"{field_name}: value is {value} (NaN/inf is not allowed); must be a finite number")
            continue

        if field_name == "grip_force_min_n":
            lo, hi = GRIP_FORCE_MIN_N, GRIP_FORCE_MAX_N
        else:
            lo, hi = CARRY_ACCEL_MIN_MS2, CARRY_ACCEL_MAX_MS2

        if value < lo or value > hi:
            errors.append(
                f"{field_name}={value} is outside the allowed range [{lo}, {hi}]; "
                f"choose a value between {lo} and {hi}"
            )
        elif field_name == "grip_force_min_n" and class_known:
            crush_limit = CRUSH_LIMIT_N[object_class]
            if value > crush_limit:
                errors.append(
                    f"grip_force_min_n={value} exceeds crush limit {crush_limit} N for "
                    f"object_class {object_class!r}; choose a value between {lo} and {crush_limit}"
                )

    return errors


def validate(rule) -> ValidationResult:
    """Validate a Rule or a dict that is supposed to look like one.

    Accepts a dict (e.g. raw JSON the LLM produced) or an already-built
    schema.Rule instance. Returns a ValidationResult collecting every error
    found, not just the first.
    """
    if isinstance(rule, Rule):
        data = rule.model_dump(mode="json")
    elif isinstance(rule, dict):
        data = rule
    else:
        return ValidationResult(ok=False, errors=[f"<root>: expected a dict or rules.schema.Rule, got {type(rule).__name__}"])

    structural, data = _structural_errors(data)
    semantic = _semantic_errors(data) if isinstance(data, dict) else []

    errors = structural + semantic
    return ValidationResult(ok=not errors, errors=errors)
