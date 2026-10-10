import datetime as dt

import pytest

from rules.schema import Rule

FIXED_TIME = dt.datetime(2026, 10, 11, 0, 0, 0, tzinfo=dt.timezone.utc)


def make_rule(**overrides) -> Rule:
    fields = dict(
        rule_id="r_0001",
        object_class="glass_jar",
        grip_force_min_n=14.0,
        carry_accel_max_ms2=2.5,
        proposed_by="solver",
        created_at=FIXED_TIME,
        rationale="slip before drop",
        source_report_ids=["fr_0001"],
    )
    fields.update(overrides)
    return Rule(**fields)


def make_rule_dict(**overrides) -> dict:
    fields = dict(
        rule_id="r_0001",
        object_class="glass_jar",
        grip_force_min_n=14.0,
        carry_accel_max_ms2=2.5,
        proposed_by="solver",
        created_at=FIXED_TIME.isoformat(),
        rationale="slip before drop",
        source_report_ids=["fr_0001"],
    )
    fields.update(overrides)
    return fields


@pytest.fixture
def rule_factory():
    return make_rule


@pytest.fixture
def rule_dict_factory():
    return make_rule_dict


@pytest.fixture
def audit_log_path(tmp_path):
    return tmp_path / "audit.jsonl"
