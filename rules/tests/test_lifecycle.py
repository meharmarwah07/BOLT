import pytest

from rules.lifecycle import (
    Check1Record,
    LifecycleError,
    State,
    current_cell_state,
    current_state,
    record_check1,
    transition,
    transition_cell,
)


def _approve_rule(log_path, rule_id="r1"):
    transition(log_path, rule_id, State.PROPOSED, "solver")
    record_check1(log_path, rule_id, "sim", True, Check1Record("glass_jar", 5, 0, True, 0, ["fr_1"]))
    transition(log_path, rule_id, State.APPROVED, "alice", {"note": "ok"})


def test_initial_state_is_none(audit_log_path):
    assert current_state(audit_log_path, "unknown_rule") is None


def test_legal_chain_proposed_to_approved(audit_log_path):
    transition(audit_log_path, "r1", State.PROPOSED, "solver")
    assert current_state(audit_log_path, "r1") == State.PROPOSED

    record_check1(audit_log_path, "r1", "sim", True, Check1Record("glass_jar", 5, 0, True, 0, ["fr_1"]))
    assert current_state(audit_log_path, "r1") == State.CHECK1_PASSED

    transition(audit_log_path, "r1", State.APPROVED, "alice", {"note": "ok"})
    assert current_state(audit_log_path, "r1") == State.APPROVED


def test_legal_chain_check1_failed_is_terminal(audit_log_path):
    transition(audit_log_path, "r1", State.PROPOSED, "solver")
    record_check1(audit_log_path, "r1", "sim", False, Check1Record("glass_jar", 5, 5, True, 0, ["fr_1"]))
    assert current_state(audit_log_path, "r1") == State.CHECK1_FAILED
    with pytest.raises(LifecycleError):
        transition(audit_log_path, "r1", State.APPROVED, "alice")


def test_legal_chain_rejected_is_terminal(audit_log_path):
    transition(audit_log_path, "r1", State.PROPOSED, "solver")
    record_check1(audit_log_path, "r1", "sim", True, Check1Record("glass_jar", 5, 0, True, 0, ["fr_1"]))
    transition(audit_log_path, "r1", State.REJECTED, "alice", {"note": "no"})
    assert current_state(audit_log_path, "r1") == State.REJECTED
    with pytest.raises(LifecycleError):
        transition(audit_log_path, "r1", State.APPROVED, "alice")


@pytest.mark.parametrize(
    "from_state,event",
    [
        (None, State.CHECK1_PASSED),
        (None, State.APPROVED),
        (None, State.REJECTED),
        (State.PROPOSED, State.APPROVED),
        (State.PROPOSED, State.REJECTED),
    ],
)
def test_illegal_rule_level_transitions(audit_log_path, from_state, event):
    if from_state == State.PROPOSED:
        transition(audit_log_path, "r1", State.PROPOSED, "solver")
    with pytest.raises(LifecycleError):
        transition(audit_log_path, "r1", event, "someone")


def test_illegal_transition_is_not_logged(audit_log_path):
    transition(audit_log_path, "r1", State.PROPOSED, "solver")
    with pytest.raises(LifecycleError):
        transition(audit_log_path, "r1", State.APPROVED, "alice")
    assert current_state(audit_log_path, "r1") == State.PROPOSED


def test_cell_transitions_require_rule_approved_first(audit_log_path):
    transition(audit_log_path, "r1", State.PROPOSED, "solver")
    with pytest.raises(LifecycleError):
        transition_cell(audit_log_path, "r1", "B", State.SHARED, "policy")


def test_cell_lifecycle_shared_to_active_to_revoked(audit_log_path):
    _approve_rule(audit_log_path, "r1")
    transition_cell(audit_log_path, "r1", "B", State.SHARED, "policy")
    assert current_cell_state(audit_log_path, "r1", "B") == State.SHARED

    transition_cell(audit_log_path, "r1", "B", State.ACTIVE, "owner1")
    assert current_cell_state(audit_log_path, "r1", "B") == State.ACTIVE

    transition_cell(audit_log_path, "r1", "B", State.REVOKED, "owner1")
    assert current_cell_state(audit_log_path, "r1", "B") == State.REVOKED

    with pytest.raises(LifecycleError):
        transition_cell(audit_log_path, "r1", "B", State.ACTIVE, "owner1")


def test_cell_lifecycle_blocked_cannot_activate(audit_log_path):
    _approve_rule(audit_log_path, "r1")
    transition_cell(audit_log_path, "r1", "C", State.BLOCKED, "policy")
    assert current_cell_state(audit_log_path, "r1", "C") == State.BLOCKED
    with pytest.raises(LifecycleError):
        transition_cell(audit_log_path, "r1", "C", State.ACTIVE, "owner2")


def test_cell_lifecycle_decision_can_flip_before_activation(audit_log_path):
    _approve_rule(audit_log_path, "r1")
    transition_cell(audit_log_path, "r1", "C", State.BLOCKED, "policy")
    transition_cell(audit_log_path, "r1", "C", State.SHARED, "policy")
    assert current_cell_state(audit_log_path, "r1", "C") == State.SHARED


def test_cell_states_are_independent_per_cell(audit_log_path):
    _approve_rule(audit_log_path, "r1")
    transition_cell(audit_log_path, "r1", "B", State.SHARED, "policy")
    transition_cell(audit_log_path, "r1", "C", State.BLOCKED, "policy")
    assert current_cell_state(audit_log_path, "r1", "B") == State.SHARED
    assert current_cell_state(audit_log_path, "r1", "C") == State.BLOCKED


def test_cell_events_do_not_change_rule_level_state(audit_log_path):
    _approve_rule(audit_log_path, "r1")
    transition_cell(audit_log_path, "r1", "B", State.SHARED, "policy")
    transition_cell(audit_log_path, "r1", "B", State.ACTIVE, "owner1")
    assert current_state(audit_log_path, "r1") == State.APPROVED


def test_state_rebuilt_from_log_after_reload(audit_log_path):
    _approve_rule(audit_log_path, "r1")
    transition_cell(audit_log_path, "r1", "B", State.SHARED, "policy")
    transition_cell(audit_log_path, "r1", "B", State.ACTIVE, "owner1")

    # No separate state file -- re-derive purely from re-reading the log path.
    assert current_state(audit_log_path, "r1") == State.APPROVED
    assert current_cell_state(audit_log_path, "r1", "B") == State.ACTIVE


def test_multiple_rules_tracked_independently(audit_log_path):
    transition(audit_log_path, "r1", State.PROPOSED, "solver")
    transition(audit_log_path, "r2", State.PROPOSED, "solver")
    record_check1(audit_log_path, "r2", "sim", False, Check1Record("steel", 2, 2, True, 1, ["fr_2"]))

    assert current_state(audit_log_path, "r1") == State.PROPOSED
    assert current_state(audit_log_path, "r2") == State.CHECK1_FAILED
