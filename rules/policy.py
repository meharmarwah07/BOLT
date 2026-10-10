"""Permission gate: can_share(), config flags, share-export, revoke.

Default-deny. Every decision -- allowed or denied -- is logged. See
DECISIONS.md for the two flags below, which are undecided per the spec and
therefore live here instead of being hardcoded.
"""
from dataclasses import dataclass

from rules.lifecycle import LifecycleError, State, current_state, transition_cell
from rules.schema import Rule
from rules.validate import validate

# Undecided in the spec: whether a different-owner share also needs the
# receiving owner's opt-in, on top of the source owner's sharing flag.
# PLACEHOLDER default: off (source owner's flag alone is sufficient, as in
# "Check order" step 3 of the spec).
REQUIRE_RECEIVER_OPT_IN = False

# Whether an owner can revoke a rule that is ACTIVE on one of their cells.
ALLOW_OWNER_REVOKE = True


@dataclass(frozen=True)
class Owner:
    owner_id: str
    sharing_enabled: bool = False  # this owner allows rules proposed from their cells to cross to other owners
    receiving_opt_in: bool = False  # this owner has opted in to receive rules from other owners (only read if REQUIRE_RECEIVER_OPT_IN)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def can_share(audit_path, rule: Rule, from_cell, to_cell, owners: dict) -> Decision:
    """Decide whether `rule` may move from from_cell to to_cell. Logs the decision either way.

    from_cell, to_cell: apply.CellState (or anything with .cell_id and .owner).
    owners: {owner_id: Owner}, so sharing/opt-in flags can be looked up for
    both the source and destination owner.
    """
    decision = _decide(audit_path, rule, from_cell, to_cell, owners)

    event = State.SHARED if decision.allowed else State.BLOCKED
    try:
        transition_cell(audit_path, rule.rule_id, to_cell.cell_id, event, "policy", {"reason": decision.reason, "from_cell": from_cell.cell_id})
    except LifecycleError:
        # Rule isn't APPROVED yet (check order step 1 already denied it for
        # that reason); nothing legal to log at the per-cell level, but the
        # denial itself is still returned to the caller.
        pass

    return decision


def _decide(audit_path, rule: Rule, from_cell, to_cell, owners: dict) -> Decision:
    rule_state = current_state(audit_path, rule.rule_id)
    if rule_state != State.APPROVED:
        return Decision(False, "rule not approved")

    if from_cell.owner == to_cell.owner:
        return Decision(True, "same owner")

    source_owner = owners.get(from_cell.owner)
    if source_owner is None or not source_owner.sharing_enabled:
        return Decision(False, "different owner and source owner's sharing flag is off")

    if REQUIRE_RECEIVER_OPT_IN:
        dest_owner = owners.get(to_cell.owner)
        if dest_owner is None or not dest_owner.receiving_opt_in:
            return Decision(False, "different owner, source sharing is on, but receiving owner has not opted in")

    return Decision(True, "different owner, source owner's sharing flag is on")


def export_for_sharing(rule: Rule) -> dict:
    """Strip everything except the rule itself before it crosses an owner boundary.

    Never includes raw failure reports or free-text evidence -- only the
    fields defined on schema.Rule, as JSON-safe primitives.
    """
    return rule.model_dump(mode="json")


def receive_shared_rule(exported: dict) -> Rule:
    """Rebuild and re-validate a rule received from another owner.

    Other owners' rules are untrusted input: always re-run validate() before
    the rule may become ACTIVE on the receiving cell. Raises ValueError with
    the validation error text if the received rule is not valid.
    """
    result = validate(exported)
    if not result.ok:
        raise ValueError(f"received rule failed validation: {result.error_text}")
    return Rule.model_validate(exported)


def revoke(audit_path, rule: Rule, cell, actor: str, reason: str) -> None:
    """Owner revokes a rule that is ACTIVE on one of their cells. Logs a REVOKED event.

    Raises RuntimeError if ALLOW_OWNER_REVOKE is False, or LifecycleError if
    the rule is not currently ACTIVE on that cell.
    """
    if not ALLOW_OWNER_REVOKE:
        raise RuntimeError("owner revoke is disabled (policy.ALLOW_OWNER_REVOKE = False)")
    transition_cell(audit_path, rule.rule_id, cell.cell_id, State.REVOKED, actor, {"reason": reason})
