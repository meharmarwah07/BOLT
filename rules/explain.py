"""Deterministic plain-language description of a Rule.

A fixed string template over the rule's own fields -- never an LLM call, and
never reads rule.rationale (that field is free text from the proposer and
must never drive logic or review-facing wording; the reviewer judges the
numbers and the Check 1 evidence, not the LLM's prose).
"""
from rules.schema import Rule


def explain(rule: Rule) -> str:
    object_class = rule.object_class.value if hasattr(rule.object_class, "value") else rule.object_class
    report_ids = ", ".join(rule.source_report_ids)
    return (
        f"Rule {rule.rule_id} proposes that for object_class '{object_class}', "
        f"the controller require a grip force of at least {rule.grip_force_min_n} N "
        f"and a carry acceleration of at most {rule.carry_accel_max_ms2} m/s^2, "
        f"based on report(s): {report_ids}."
    )
