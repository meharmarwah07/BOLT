"""Human review CLI. No web UI -- this is it.

    python -m rules.review_cli --log <path> --store <dir> list
    python -m rules.review_cli --log <path> --store <dir> show <rule_id>
    python -m rules.review_cli --log <path> --store <dir> approve <rule_id> --reviewer NAME --note TEXT
    python -m rules.review_cli --log <path> --store <dir> reject <rule_id> --reviewer NAME --note TEXT

`--store <dir>` holds one JSON file per rule (<dir>/<rule_id>.json), written
when a rule is proposed (see demo_e2e.py / the Part B integration shim) --
the audit log records events, not the rule payload itself, so the CLI needs
somewhere to read the rule's fields back from for `show`.
"""
import argparse
import json
import sys
from pathlib import Path

from rules.audit import read_events
from rules.explain import explain
from rules.lifecycle import Check1Record, LifecycleError, State, current_state, transition
from rules.schema import Rule


def store_rule(store_dir, rule: Rule) -> None:
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / f"{rule.rule_id}.json").write_text(json.dumps(rule.model_dump(mode="json")), encoding="utf-8")


def load_rule(store_dir, rule_id: str) -> Rule:
    path = Path(store_dir) / f"{rule_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no stored rule for {rule_id!r} in {store_dir}")
    return Rule.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _latest_check1(log_path, rule_id: str) -> dict | None:
    details = None
    for ev in read_events(log_path):
        if ev.rule_id == rule_id and ev.event in (State.CHECK1_PASSED.value, State.CHECK1_FAILED.value):
            details = ev.details
    return details


def cmd_list(args) -> int:
    all_rule_ids = {ev.rule_id for ev in read_events(args.log)}
    pending = [rid for rid in sorted(all_rule_ids) if current_state(args.log, rid) == State.CHECK1_PASSED]
    if not pending:
        print("no rules pending review")
        return 0
    for rid in pending:
        print(rid)
    return 0


def cmd_show(args) -> int:
    rule = load_rule(args.store, args.rule_id)
    print(explain(rule))

    check1 = _latest_check1(args.log, args.rule_id)
    if check1 is None:
        print("(no Check 1 record found)")
        return 0

    print(f"Check 1: target_class={check1.get('target_class')}")
    print(f"  drops_before={check1.get('drops_before')} drops_after={check1.get('drops_after')}")
    print(f"  other_classes_unchanged={check1.get('other_classes_unchanged')}")
    print(f"  crush_count={check1.get('crush_count')}")
    print(f"  cited_report_ids={check1.get('cited_report_ids')}")
    return 0


def cmd_approve(args) -> int:
    try:
        transition(args.log, args.rule_id, State.APPROVED, args.reviewer, {"note": args.note})
    except LifecycleError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{args.rule_id}: APPROVED by {args.reviewer}")
    return 0


def cmd_reject(args) -> int:
    try:
        transition(args.log, args.rule_id, State.REJECTED, args.reviewer, {"note": args.note})
    except LifecycleError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{args.rule_id}: REJECTED by {args.reviewer}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rules.review_cli")
    parser.add_argument("--log", required=True, help="path to the audit log (JSON Lines)")
    parser.add_argument("--store", required=True, help="directory holding one JSON file per rule")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list rules pending review (Check 1 done, not yet approved/rejected)")

    p_show = sub.add_parser("show", help="show a rule in plain language plus its Check 1 evidence")
    p_show.add_argument("rule_id")

    p_approve = sub.add_parser("approve", help="approve a rule")
    p_approve.add_argument("rule_id")
    p_approve.add_argument("--reviewer", required=True)
    p_approve.add_argument("--note", required=True)

    p_reject = sub.add_parser("reject", help="reject a rule")
    p_reject.add_argument("rule_id")
    p_reject.add_argument("--reviewer", required=True)
    p_reject.add_argument("--note", required=True)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"list": cmd_list, "show": cmd_show, "approve": cmd_approve, "reject": cmd_reject}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
