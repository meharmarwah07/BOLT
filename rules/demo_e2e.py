"""End-to-end demo of the full rule lifecycle, also runnable as a pytest.

Cell A and Cell B belong to owner1. Cell C belongs to owner2, whose sharing
flag starts OFF. A rule proposed from Cell A's failures is validated, logged
PROPOSED, given a (fabricated) Check 1 replay record, approved through the
review CLI's own functions, applied to Cell B with its effective settings
changed, and shown NOT applied to Cell C (denied, reason logged). Finishes
by verifying the audit chain and printing a log summary.

Run directly:  python -m rules.demo_e2e
Run as a test: pytest rules/demo_e2e.py
"""
import datetime as dt
import tempfile
from pathlib import Path

from rules.apply import CellState, add_rule, effective_settings
from rules.audit import read_events, verify_chain
from rules.lifecycle import Check1Record, State, current_cell_state, current_state, record_check1, transition
from rules.policy import Owner, can_share
from rules.review_cli import cmd_approve, cmd_reject, build_parser, store_rule
from rules.schema import Rule
from rules.validate import validate


def run_demo(workdir: Path) -> dict:
    log = workdir / "audit.jsonl"
    store = workdir / "store"

    rule = Rule(
        rule_id="r_demo_0001",
        object_class="glass_jar",
        grip_force_min_n=14.0,
        carry_accel_max_ms2=2.5,
        proposed_by="solver",
        created_at=dt.datetime.now(dt.timezone.utc),
        rationale="Cell A dropped glass_jar 5/10 held-out trials at 8 N grip; raising to 14 N and "
        "capping carry accel at 2.5 m/s^2 eliminated drops in replay.",
        source_report_ids=["fr_0001", "fr_0002"],
    )

    result = validate(rule)
    assert result.ok, f"demo rule failed validate(): {result.error_text}"

    store_rule(store, rule)
    transition(log, rule.rule_id, State.PROPOSED, "solver")
    assert current_state(log, rule.rule_id) == State.PROPOSED

    record_check1(
        log,
        rule.rule_id,
        "sim",
        True,
        Check1Record(
            target_class="glass_jar",
            drops_before=5,
            drops_after=0,
            other_classes_unchanged=True,
            crush_count=0,
            cited_report_ids=rule.source_report_ids,
        ),
    )
    assert current_state(log, rule.rule_id) == State.CHECK1_PASSED

    parser = build_parser()
    approve_args = parser.parse_args(["--log", str(log), "--store", str(store), "approve", rule.rule_id, "--reviewer", "alice", "--note", "replay clean, approve"])
    assert cmd_approve(approve_args) == 0
    assert current_state(log, rule.rule_id) == State.APPROVED

    base_params = {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}
    cell_a = CellState("A", "owner1", base_params)
    cell_b = CellState("B", "owner1", base_params)
    cell_c = CellState("C", "owner2", base_params)

    owners = {
        "owner1": Owner("owner1", sharing_enabled=False),  # owner1 hasn't opted to share cross-owner
        "owner2": Owner("owner2", sharing_enabled=False),
    }

    decision_b = can_share(log, rule, cell_a, cell_b, owners)
    assert decision_b.allowed, decision_b.reason
    assert current_cell_state(log, rule.rule_id, "B") == State.SHARED

    cell_b = add_rule(cell_b, rule)
    settings_b = effective_settings(cell_b, rule.object_class)
    assert settings_b == {"grip_force_n": 14.0, "carry_accel_ms2": 2.5}
    assert settings_b != base_params

    decision_c = can_share(log, rule, cell_a, cell_c, owners)
    assert not decision_c.allowed
    assert current_cell_state(log, rule.rule_id, "C") == State.BLOCKED
    settings_c = effective_settings(cell_c, rule.object_class)
    assert settings_c == base_params  # unchanged: the rule never reached Cell C

    chain_ok = verify_chain(log)
    assert chain_ok is True

    events = read_events(log)
    return {
        "rule_id": rule.rule_id,
        "settings_b": settings_b,
        "settings_c": settings_c,
        "decision_b": decision_b,
        "decision_c": decision_c,
        "chain_ok": chain_ok,
        "n_events": len(events),
        "events": events,
    }


def print_summary(summary: dict) -> None:
    print(f"rule: {summary['rule_id']}")
    print(f"Cell B (owner1, same owner as A) effective settings: {summary['settings_b']}")
    print(f"Cell C (owner2, sharing OFF) effective settings:     {summary['settings_c']} (unchanged)")
    print(f"share decision B: allowed={summary['decision_b'].allowed} ({summary['decision_b'].reason})")
    print(f"share decision C: allowed={summary['decision_c'].allowed} ({summary['decision_c'].reason})")
    print(f"audit chain verifies: {summary['chain_ok']}")
    print(f"audit log: {summary['n_events']} events")
    for ev in summary["events"]:
        detail_keys = ", ".join(sorted(ev.details.keys())) if ev.details else ""
        print(f"  {ev.ts}  {ev.event:14} rule={ev.rule_id} actor={ev.actor}" + (f" details=[{detail_keys}]" if detail_keys else ""))


def test_demo_e2e(tmp_path):
    summary = run_demo(tmp_path)
    assert summary["settings_b"] == {"grip_force_n": 14.0, "carry_accel_ms2": 2.5}
    assert summary["settings_c"] == {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}
    assert summary["decision_b"].allowed
    assert not summary["decision_c"].allowed
    assert summary["chain_ok"] is True


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        summary = run_demo(Path(tmp))
        print_summary(summary)
