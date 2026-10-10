import math

import pytest

from rules.validate import validate


def good(**overrides):
    base = dict(
        schema_version="1",
        rule_id="r_0001",
        object_class="glass_jar",
        grip_force_min_n=14.0,
        carry_accel_max_ms2=2.5,
        proposed_by="solver",
        created_at="2026-10-11T00:00:00Z",
        rationale="slip before drop",
        source_report_ids=["fr_0001"],
    )
    base.update(overrides)
    return base


# (name, data, expect_ok, substrings_expected_in_error_text)
CASES = [
    ("all fields good", good(), True, []),
    ("omits schema_version (defaults to current)", {k: v for k, v in good().items() if k != "schema_version"}, True, []),
    ("missing rule_id", {k: v for k, v in good().items() if k != "rule_id"}, False, ["rule_id", "missing"]),
    ("missing object_class", {k: v for k, v in good().items() if k != "object_class"}, False, ["object_class", "missing"]),
    ("missing grip_force_min_n", {k: v for k, v in good().items() if k != "grip_force_min_n"}, False, ["grip_force_min_n", "missing"]),
    ("missing source_report_ids", {k: v for k, v in good().items() if k != "source_report_ids"}, False, ["source_report_ids", "missing"]),
    ("extra field", good(confidence=0.9), False, ["confidence", "not permitted"]),
    ("wrong type grip_force_min_n (string)", good(grip_force_min_n="fourteen"), False, ["grip_force_min_n", "number"]),
    ("wrong type source_report_ids (string not list)", good(source_report_ids="fr_0001"), False, ["source_report_ids"]),
    ("NaN grip_force_min_n", good(grip_force_min_n=float("nan")), False, ["grip_force_min_n", "NaN"]),
    ("inf carry_accel_max_ms2", good(carry_accel_max_ms2=float("inf")), False, ["carry_accel_max_ms2", "NaN/inf"]),
    ("unknown object_class", good(object_class="bottle"), False, ["object_class", "bottle", "steel", "cardboard", "glass_jar"]),
    ("grip_force_min_n below absolute min", good(grip_force_min_n=1.0), False, ["grip_force_min_n=1.0", "range", "[2.0, 40.0]"]),
    ("grip_force_min_n above absolute max", good(grip_force_min_n=50.0), False, ["grip_force_min_n=50.0", "range"]),
    ("carry_accel_max_ms2 below absolute min", good(carry_accel_max_ms2=0.1), False, ["carry_accel_max_ms2=0.1", "range"]),
    ("carry_accel_max_ms2 above absolute max", good(carry_accel_max_ms2=10.0), False, ["carry_accel_max_ms2=10.0", "range"]),
    ("grip_force_min_n above crush limit for glass_jar", good(grip_force_min_n=35.0), False, ["crush limit", "25", "glass_jar"]),
    ("grip_force_min_n above crush limit for cardboard", good(object_class="cardboard", grip_force_min_n=13.0), False, ["crush limit", "12", "cardboard"]),
    ("grip_force_min_n at crush limit boundary is ok", good(object_class="cardboard", grip_force_min_n=12.0), True, []),
    ("empty source_report_ids", good(source_report_ids=[]), False, ["source_report_ids", "at least 1"]),
    ("rationale too long", good(rationale="x" * 301), False, ["rationale", "too long", "300"]),
    ("bad schema_version", good(schema_version="2"), False, ["schema_version", "2", "unsupported"]),
    ("multiple errors at once", good(object_class="bottle", grip_force_min_n=99.0, source_report_ids=[]), False, ["object_class", "grip_force_min_n", "source_report_ids"]),
    ("not a dict or Rule", "just a string", False, ["dict or rules.schema.Rule"]),
]


@pytest.mark.parametrize("name,data,expect_ok,substrings", CASES, ids=[c[0] for c in CASES])
def test_validate_table(name, data, expect_ok, substrings):
    result = validate(data)
    assert result.ok is expect_ok, f"{name}: expected ok={expect_ok}, errors={result.errors}"
    for s in substrings:
        assert s in result.error_text, f"{name}: expected {s!r} in error_text {result.error_text!r}"


def test_validate_collects_all_errors_not_just_first():
    data = good(object_class="bottle", grip_force_min_n=99.0, source_report_ids=[])
    result = validate(data)
    assert len(result.errors) >= 3


def test_validate_accepts_rule_instance(rule_factory):
    rule = rule_factory()
    result = validate(rule)
    assert result.ok


def test_validate_error_text_is_joined_string():
    result = validate(good(object_class="bottle", rule_id=None))
    assert isinstance(result.error_text, str)
    assert ";" in result.error_text or len(result.errors) == 1
