import pytest

import rules.policy as policy
from rules.apply import CellState
from rules.lifecycle import Check1Record, State, current_cell_state, record_check1, transition
from rules.policy import Decision, Owner, can_share, export_for_sharing, receive_shared_rule, revoke


@pytest.fixture(autouse=True)
def restore_flags():
    before_opt_in = policy.REQUIRE_RECEIVER_OPT_IN
    before_revoke = policy.ALLOW_OWNER_REVOKE
    yield
    policy.REQUIRE_RECEIVER_OPT_IN = before_opt_in
    policy.ALLOW_OWNER_REVOKE = before_revoke


def _approved_rule(audit_log_path, rule, rule_id="r1"):
    transition(audit_log_path, rule_id, State.PROPOSED, "solver")
    record_check1(audit_log_path, rule_id, "sim", True, Check1Record("glass_jar", 5, 0, True, 0, ["fr_1"]))
    transition(audit_log_path, rule_id, State.APPROVED, "alice", {"note": "ok"})


def _cells():
    base = {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}
    a = CellState("A", "owner1", base)
    b = CellState("B", "owner1", base)
    c = CellState("C", "owner2", base)
    return a, b, c


# --- permission matrix: one row per case, opt-in flag both ways ---

PERMISSION_MATRIX = [
    # name, from_cell, to_cell, source_sharing_enabled, receiver_opt_in, require_opt_in_flag, rule_approved, expect_allowed
    ("not approved yet, same owner", "A", "B", False, False, False, False, False),
    ("not approved yet, different owner, sharing on", "A", "C", True, True, False, False, False),
    ("same owner always allowed regardless of flag", "A", "B", False, False, False, True, True),
    ("same owner always allowed, flag on", "A", "B", True, False, False, True, True),
    ("different owner, sharing off, opt-in not required", "A", "C", False, False, False, True, False),
    ("different owner, sharing on, opt-in not required", "A", "C", True, False, False, True, True),
    ("different owner, sharing off, opt-in required", "A", "C", False, False, True, True, False),
    ("different owner, sharing on, opt-in required but receiver has not opted in", "A", "C", True, False, True, True, False),
    ("different owner, sharing on, opt-in required and receiver opted in", "A", "C", True, True, True, True, True),
]


@pytest.mark.parametrize(
    "name,from_id,to_id,sharing_enabled,receiver_opt_in,require_opt_in,approved,expect_allowed",
    PERMISSION_MATRIX,
    ids=[row[0] for row in PERMISSION_MATRIX],
)
def test_permission_matrix(audit_log_path, name, from_id, to_id, sharing_enabled, receiver_opt_in, require_opt_in, approved, expect_allowed, rule_factory):
    policy.REQUIRE_RECEIVER_OPT_IN = require_opt_in
    a, b, c = _cells()
    cells = {"A": a, "B": b, "C": c}
    from_cell, to_cell = cells[from_id], cells[to_id]

    rule = rule_factory(rule_id="r1")
    if approved:
        _approved_rule(audit_log_path, rule, "r1")
    else:
        transition(audit_log_path, "r1", State.PROPOSED, "solver")

    owners = {
        "owner1": Owner("owner1", sharing_enabled=sharing_enabled, receiving_opt_in=receiver_opt_in),
        "owner2": Owner("owner2", sharing_enabled=sharing_enabled, receiving_opt_in=receiver_opt_in),
    }

    decision = can_share(audit_log_path, rule, from_cell, to_cell, owners)
    assert decision.allowed is expect_allowed, f"{name}: {decision}"


def test_denied_decisions_are_logged(audit_log_path, rule_factory):
    a, b, c = _cells()
    rule = rule_factory(rule_id="r1")
    _approved_rule(audit_log_path, rule, "r1")
    owners = {"owner1": Owner("owner1", sharing_enabled=False), "owner2": Owner("owner2")}

    decision = can_share(audit_log_path, rule, a, c, owners)
    assert decision.allowed is False
    assert current_cell_state(audit_log_path, "r1", "C") == State.BLOCKED


def test_allowed_decisions_are_logged(audit_log_path, rule_factory):
    a, b, c = _cells()
    rule = rule_factory(rule_id="r1")
    _approved_rule(audit_log_path, rule, "r1")
    owners = {"owner1": Owner("owner1", sharing_enabled=True), "owner2": Owner("owner2")}

    decision = can_share(audit_log_path, rule, a, c, owners)
    assert decision.allowed is True
    assert current_cell_state(audit_log_path, "r1", "C") == State.SHARED


def test_default_deny_with_unknown_source_owner(audit_log_path, rule_factory):
    a, _, c = _cells()
    a_unknown_owner = CellState("A", "owner_nonexistent", a.base_params)
    rule = rule_factory(rule_id="r1")
    _approved_rule(audit_log_path, rule, "r1")
    decision = can_share(audit_log_path, rule, a_unknown_owner, c, owners={"owner2": Owner("owner2")})
    assert decision.allowed is False


def test_export_strips_to_rule_fields_only(rule_factory):
    rule = rule_factory(rule_id="r1")
    exported = export_for_sharing(rule)
    assert set(exported.keys()) == set(rule.model_dump(mode="json").keys())
    assert "raw_failure_reports" not in exported


def test_receive_shared_rule_revalidates(rule_factory):
    rule = rule_factory(rule_id="r1")
    exported = export_for_sharing(rule)
    received = receive_shared_rule(exported)
    assert received.rule_id == "r1"


def test_receive_shared_rule_rejects_invalid_untrusted_rule():
    bad = {
        "rule_id": "r_evil",
        "object_class": "glass_jar",
        "grip_force_min_n": 999.0,  # way above crush limit
        "carry_accel_max_ms2": 2.5,
        "proposed_by": "owner2",
        "created_at": "2026-10-11T00:00:00Z",
        "rationale": "x",
        "source_report_ids": ["fr_1"],
    }
    with pytest.raises(ValueError):
        receive_shared_rule(bad)


def test_revoke_active_rule(audit_log_path, rule_factory):
    a, b, _ = _cells()
    rule = rule_factory(rule_id="r1")
    _approved_rule(audit_log_path, rule, "r1")
    can_share(audit_log_path, rule, a, b, owners={"owner1": Owner("owner1")})

    from rules.lifecycle import transition_cell
    transition_cell(audit_log_path, "r1", "B", State.ACTIVE, "owner1")

    revoke(audit_log_path, rule, b, "owner1", "no longer needed")
    assert current_cell_state(audit_log_path, "r1", "B") == State.REVOKED


def test_revoke_disabled_by_flag(audit_log_path, rule_factory):
    policy.ALLOW_OWNER_REVOKE = False
    a, b, _ = _cells()
    rule = rule_factory(rule_id="r1")
    _approved_rule(audit_log_path, rule, "r1")
    with pytest.raises(RuntimeError):
        revoke(audit_log_path, rule, b, "owner1", "no longer needed")
