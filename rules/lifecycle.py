"""Rule lifecycle state machine.

PROPOSED -> CHECK1_PASSED | CHECK1_FAILED -> APPROVED | REJECTED
         -> (per receiving cell) SHARED | BLOCKED -> ACTIVE -> REVOKED

There is no separate state file. Current state is derived by replaying a
rule's events from the audit log (event-sourced) via current_state().
Illegal transitions raise LifecycleError and are never written to the log.
"""
from dataclasses import asdict, dataclass, field
from enum import Enum

from rules.audit import append_event, read_events


class State(str, Enum):
    PROPOSED = "PROPOSED"
    CHECK1_PASSED = "CHECK1_PASSED"
    CHECK1_FAILED = "CHECK1_FAILED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SHARED = "SHARED"
    BLOCKED = "BLOCKED"
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"


class LifecycleError(Exception):
    """Raised on an illegal state transition. Nothing is logged when this is raised."""


# Legal transitions for the rule's own lifecycle (not per-cell share/activate
# events, which are scoped separately -- see _PER_CELL_TRANSITIONS).
_TRANSITIONS = {
    None: {State.PROPOSED},
    State.PROPOSED: {State.CHECK1_PASSED, State.CHECK1_FAILED},
    State.CHECK1_PASSED: {State.APPROVED, State.REJECTED},
    State.CHECK1_FAILED: set(),  # terminal: a failed replay needs a new rule
    State.APPROVED: set(),  # sharing/activation is tracked per cell, not here
    State.REJECTED: set(),  # terminal
}

# Per-(rule, cell) transitions, layered on top of the rule-level state once
# it reaches APPROVED. can_share() re-evaluates every time it is called (a
# sharing flag can change later), so SHARED/BLOCKED may be re-entered as the
# decision flips, right up until the rule is actually activated on that cell.
_PER_CELL_TRANSITIONS = {
    None: {State.SHARED, State.BLOCKED},
    State.SHARED: {State.SHARED, State.BLOCKED, State.ACTIVE},
    State.BLOCKED: {State.SHARED, State.BLOCKED},
    State.ACTIVE: {State.REVOKED},
    State.REVOKED: set(),  # terminal
}

_EVENT_TO_STATE = {s.value: s for s in State}


@dataclass(frozen=True)
class Check1Record:
    """Simulation replay record for Check 1, for the rule's target object_class."""

    target_class: str
    drops_before: int
    drops_after: int
    other_classes_unchanged: bool
    crush_count: int
    cited_report_ids: list = field(default_factory=list)
    # Added for Check 1 v2 (pipeline/check1.py): |after - nominal| / nominal
    # for grip and accel. Optional/defaulted so existing (pre-v2) callers
    # and already-logged audit entries are unaffected.
    intervention_cost_grip: "float | None" = None
    intervention_cost_accel: "float | None" = None


def current_state(path, rule_id: str) -> State | None:
    """Replay the audit log and return the rule's current lifecycle state.

    Returns None if the rule has no PROPOSED event yet.
    """
    state = None
    for ev in read_events(path):
        if ev.rule_id != rule_id or ev.event not in _EVENT_TO_STATE:
            continue
        if "cell_id" in ev.details:
            continue  # per-cell event, not a rule-level transition
        state = _EVENT_TO_STATE[ev.event]
    return state


def current_cell_state(path, rule_id: str, cell_id: str) -> State | None:
    """Replay the audit log and return the rule's lifecycle state for one receiving cell."""
    state = None
    for ev in read_events(path):
        if ev.rule_id != rule_id or ev.event not in _EVENT_TO_STATE:
            continue
        if ev.details.get("cell_id") != cell_id:
            continue
        state = _EVENT_TO_STATE[ev.event]
    return state


def record_check1(path, rule_id: str, actor: str, passed: bool, record: Check1Record) -> None:
    """Convenience wrapper: log CHECK1_PASSED/CHECK1_FAILED with the replay numbers attached."""
    transition(path, rule_id, State.CHECK1_PASSED if passed else State.CHECK1_FAILED, actor, asdict(record))


def transition(path, rule_id: str, event: State, actor: str, details: dict | None = None) -> None:
    """Attempt a rule-level transition. Raises LifecycleError if illegal, else appends to the log."""
    state = current_state(path, rule_id)
    allowed = _TRANSITIONS.get(state, set())
    if event not in allowed:
        raise LifecycleError(
            f"illegal transition for rule {rule_id}: {state.value if state else 'None'} -> {event.value}"
        )
    append_event(path, event.value, rule_id, actor, details or {})


def transition_cell(path, rule_id: str, cell_id: str, event: State, actor: str, details: dict | None = None) -> None:
    """Attempt a per-cell transition (SHARED/BLOCKED/ACTIVE/REVOKED). Requires rule-level APPROVED first."""
    rule_state = current_state(path, rule_id)
    if rule_state != State.APPROVED:
        raise LifecycleError(
            f"illegal transition for rule {rule_id} on cell {cell_id}: rule must be APPROVED first (is {rule_state.value if rule_state else 'None'})"
        )

    cell_state = current_cell_state(path, rule_id, cell_id)
    allowed = _PER_CELL_TRANSITIONS.get(cell_state, set())
    if event not in allowed:
        raise LifecycleError(
            f"illegal transition for rule {rule_id} on cell {cell_id}: {cell_state.value if cell_state else 'None'} -> {event.value}"
        )
    details = dict(details or {})
    details["cell_id"] = cell_id
    append_event(path, event.value, rule_id, actor, details)
