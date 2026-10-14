"""apply(): combine a Rule with current controller params for one object class.

Also CellState and the multi-rule helpers (effective_settings, add_rule,
remove_rule) for applying several active rules to one cell at once.
"""
import copy
from dataclasses import dataclass, field, replace

from rules.limits import (
    CARRY_ACCEL_MAX_MS2,
    CARRY_ACCEL_MIN_MS2,
    CRUSH_LIMIT_N,
    GRIP_FORCE_MAX_N,
    GRIP_FORCE_MIN_N,
)
from rules.schema import Rule


class RuleRejected(Exception):
    """Raised when applying a rule would push params outside a hard limit.

    Never silently clamped -- the caller must see this and decide.
    """


def _object_class_value(object_class) -> str:
    return object_class.value if hasattr(object_class, "value") else object_class


def _check_accel_within_limits(accel: float, context: str) -> None:
    if accel < CARRY_ACCEL_MIN_MS2 or accel > CARRY_ACCEL_MAX_MS2:
        raise RuleRejected(
            f"{context}: carry_accel_ms2={accel} would be outside datasheet limits "
            f"[{CARRY_ACCEL_MIN_MS2}, {CARRY_ACCEL_MAX_MS2}]"
        )


def _check_within_limits(params: dict, object_class: str, context: str) -> None:
    """Fixed-rule params only (params has "grip_force_n"). A conditional
    rule's realized grip depends on a per-trial probe reading, not a
    single number apply() can check deterministically -- see apply()'s own
    conditional branch, which only checks accel via _check_accel_within_limits."""
    grip = params["grip_force_n"]

    if grip < GRIP_FORCE_MIN_N or grip > GRIP_FORCE_MAX_N:
        raise RuleRejected(
            f"{context}: grip_force_n={grip} would be outside datasheet limits "
            f"[{GRIP_FORCE_MIN_N}, {GRIP_FORCE_MAX_N}]"
        )
    crush_limit = CRUSH_LIMIT_N.get(object_class)
    if crush_limit is not None and grip > crush_limit:
        raise RuleRejected(
            f"{context}: grip_force_n={grip} would exceed crush limit {crush_limit} N "
            f"for object_class {object_class!r}"
        )
    _check_accel_within_limits(params["carry_accel_ms2"], context)


def apply(rule: Rule, params: dict, object_class) -> dict:
    """Pure function. Returns a NEW params dict; never mutates `params`.

    For a "fixed" rule (rule.rule_type, default): params is
    {"grip_force_n": float, "carry_accel_ms2": float}. If rule.object_class
    != object_class, returns an unchanged copy. Otherwise grip_force_n
    becomes max(current, rule.grip_force_min_n), carry_accel_ms2 becomes
    min(current, rule.carry_accel_max_ms2). If that combination would land
    outside a datasheet or crush limit, raises RuleRejected -- never
    silently clamps. Idempotent: apply(r, apply(r, p), c) == apply(r, p, c).

    For a "conditional" rule: returns {"grip_fraction": rule.grip_fraction,
    "carry_accel_ms2": min(current, rule.carry_accel_max_ms2)} instead --
    the realized grip (grip_fraction * a per-trial probe reading,
    sim/cell_model.py's probe_crush_limit_reading) isn't a single number
    apply() can check against a crush limit at this point, so only accel
    is checked here; sim/run_trial.py resolves the actual grip per trial.
    Also idempotent (re-applying only re-derives the same two fields).
    """
    new_params = copy.deepcopy(params)
    target_class = _object_class_value(object_class)
    rule_class = _object_class_value(rule.object_class)

    if rule_class != target_class:
        return new_params

    if rule.rule_type == "conditional":
        new_accel = min(new_params["carry_accel_ms2"], rule.carry_accel_max_ms2)
        _check_accel_within_limits(new_accel, f"applying rule {rule.rule_id}")
        return {"grip_fraction": rule.grip_fraction, "carry_accel_ms2": new_accel}

    if "grip_force_n" not in new_params:
        raise RuleRejected(
            f"applying fixed rule {rule.rule_id}: current params has no grip_force_n "
            f"(got {sorted(new_params)}) -- can't combine a fixed-grip rule with a "
            "conditional (grip_fraction-shaped) upstream params dict"
        )
    new_params["grip_force_n"] = max(new_params["grip_force_n"], rule.grip_force_min_n)
    new_params["carry_accel_ms2"] = min(new_params["carry_accel_ms2"], rule.carry_accel_max_ms2)

    _check_within_limits(
        {"grip_force_n": new_params["grip_force_n"], "carry_accel_ms2": new_params["carry_accel_ms2"]},
        target_class,
        f"applying rule {rule.rule_id}",
    )
    return new_params


@dataclass(frozen=True)
class CellState:
    cell_id: str
    owner: str
    base_params: dict  # {"grip_force_n": float, "carry_accel_ms2": float}
    active_rules: tuple = field(default_factory=tuple)  # tuple[Rule, ...]


def effective_settings(cell: CellState, object_class) -> dict:
    """Combine base_params with every active rule that targets object_class.

    Largest min force wins, smallest max accel wins, for FIXED rules
    (unchanged from before this function knew about conditional rules).
    A single (or several) active CONDITIONAL rule for this class returns a
    grip_fraction-shaped result instead, combined the same max-wins way
    (a higher grip_fraction is the more aggressive/restrictive choice).
    Mixing a conditional rule with a fixed-grip rule on the SAME class is
    NOT supported -- raises RuleRejected -- there is no principled way to
    merge "a fixed newton value" with "a fraction of a stochastic probe
    reading" into one number, and nothing in this project needs it yet
    (Check 1 and dev_check.py only ever evaluate one candidate rule at a
    time against the plain baseline, never a mixed active set).
    """
    target_class = _object_class_value(object_class)
    accel = cell.base_params["carry_accel_ms2"]

    fixed_rules = []
    conditional_rules = []
    for rule in cell.active_rules:
        if _object_class_value(rule.object_class) != target_class:
            continue
        accel = min(accel, rule.carry_accel_max_ms2)
        if rule.rule_type == "conditional":
            conditional_rules.append(rule)
        else:
            fixed_rules.append(rule)

    if conditional_rules and fixed_rules:
        raise RuleRejected(
            f"effective_settings for cell {cell.cell_id}: mixing a fixed-grip rule and a "
            "conditional (grip_fraction) rule on the same object_class is not supported"
        )

    if conditional_rules:
        grip_fraction = max(rule.grip_fraction for rule in conditional_rules)
        _check_accel_within_limits(accel, f"effective_settings for cell {cell.cell_id}")
        return {"grip_fraction": grip_fraction, "carry_accel_ms2": accel}

    grip = cell.base_params["grip_force_n"]
    for rule in fixed_rules:
        grip = max(grip, rule.grip_force_min_n)
    result = {"grip_force_n": grip, "carry_accel_ms2": accel}
    _check_within_limits(result, target_class, f"effective_settings for cell {cell.cell_id}")
    return result


def add_rule(cell: CellState, rule: Rule) -> CellState:
    """Return a NEW CellState with rule added, for every object_class it could affect.

    Rejects the NEWER rule (raises RuleRejected, leaves `cell` untouched) if
    combining it with the cell's existing active rules would push the
    effective settings for its object_class outside any limit.
    """
    effective_settings(
        replace(cell, active_rules=cell.active_rules + (rule,)),
        rule.object_class,
    )
    return replace(cell, active_rules=cell.active_rules + (rule,))


def remove_rule(cell: CellState, rule_id: str) -> CellState:
    """Return a NEW CellState with the rule removed (simple list removal / rollback)."""
    return replace(cell, active_rules=tuple(r for r in cell.active_rules if r.rule_id != rule_id))
