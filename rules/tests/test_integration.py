"""Integration test with Part B (fleet/llm).

Builds a rule the way fleet/llm/fake_server.py's GOOD_REPLY and
fleet/llm/solver.py actually produce one (through the rule_check.py shim),
then checks it flows through rules.validate -> PROPOSED -> Check 1 ->
APPROVED -> shared to a same-owner cell -> ACTIVE.
"""
import sys
from pathlib import Path

import pytest

LLM_DIR = Path(__file__).resolve().parents[2] / "llm"


@pytest.fixture
def llm_rule_check():
    sys.path.insert(0, str(LLM_DIR))
    try:
        import rule_check

        yield rule_check
    finally:
        sys.path.remove(str(LLM_DIR))
        sys.modules.pop("rule_check", None)


def test_fake_server_good_reply_flows_through_shim_and_lifecycle(llm_rule_check, tmp_path):
    from rules.apply import CellState, add_rule, effective_settings
    from rules.lifecycle import Check1Record, State, current_cell_state, current_state, record_check1, transition, transition_cell
    from rules.policy import Owner, can_share
    from rules.validate import validate

    # The exact GOOD_REPLY text fleet/llm/fake_server.py serves.
    good_reply_text = (
        '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_min_n":14,'
        '"carry_accel_max_ms2":2.5,"rationale":"slip before drop","source_report_ids":["fr_0001"]}'
    )

    rule, err = llm_rule_check.parse_reply(good_reply_text)
    assert err is None
    assert rule.rule_id == "r_0001"

    # It must independently pass rules.validate (not just the shim's call to it).
    result = validate(rule)
    assert result.ok, result.error_text

    log = tmp_path / "audit.jsonl"
    transition(log, rule.rule_id, State.PROPOSED, "solver")
    assert current_state(log, rule.rule_id) == State.PROPOSED

    record_check1(log, rule.rule_id, "sim", True, Check1Record("glass_jar", 5, 0, True, 0, rule.source_report_ids))
    assert current_state(log, rule.rule_id) == State.CHECK1_PASSED

    transition(log, rule.rule_id, State.APPROVED, "alice", {"note": "matches replay evidence"})
    assert current_state(log, rule.rule_id) == State.APPROVED

    cell_a = CellState("A", "owner1", {"grip_force_n": 8.0, "carry_accel_ms2": 3.0})
    cell_b = CellState("B", "owner1", {"grip_force_n": 8.0, "carry_accel_ms2": 3.0})
    decision = can_share(log, rule, cell_a, cell_b, owners={"owner1": Owner("owner1")})
    assert decision.allowed
    assert current_cell_state(log, rule.rule_id, "B") == State.SHARED

    transition_cell(log, rule.rule_id, "B", State.ACTIVE, "owner1")
    cell_b = add_rule(cell_b, rule)
    settings = effective_settings(cell_b, rule.object_class)
    assert settings["grip_force_n"] == 14.0
    assert settings["carry_accel_ms2"] == 2.5


def test_fake_server_bad_reply_is_rejected_by_both_shim_and_validate(llm_rule_check):
    from rules.validate import validate

    bad_reply_text = (
        '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_min_n":14,'
        '"carry_accel_max_ms2":2.5,"rationale":"slip before drop",'
        '"source_report_ids":["fr_0001"],"confidence":0.9}'
    )
    obj, err = llm_rule_check.parse_reply(bad_reply_text)
    assert obj is None
    assert "not permitted" in err

    import json

    data = json.loads(bad_reply_text)
    data.setdefault("proposed_by", "llm-solver")
    data.setdefault("created_at", "2026-10-11T00:00:00Z")
    assert not validate(data).ok
