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


def _check_within_limits(params: dict, object_class: str, context: str) -> None:
    grip = params["grip_force_n"]
    accel = params["carry_accel_ms2"]

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
    if accel < CARRY_ACCEL_MIN_MS2 or accel > CARRY_ACCEL_MAX_MS2:
        raise RuleRejected(
            f"{context}: carry_accel_ms2={accel} would be outside datasheet limits "
            f"[{CARRY_ACCEL_MIN_MS2}, {CARRY_ACCEL_MAX_MS2}]"
        )


def apply(rule: Rule, params: dict, object_class) -> dict:
    """Pure function. Returns a NEW params dict; never mutates `params`.

    params: {"grip_force_n": float, "carry_accel_ms2": float}

    If rule.object_class != object_class, returns an unchanged copy.
    Otherwise grip_force_n becomes max(current, rule.min), carry_accel_ms2
    becomes min(current, rule.max). If that combination would land outside
    a datasheet or crush limit, raises RuleRejected -- never silently clamps.
    Idempotent: apply(r, apply(r, p), c) == apply(r, p, c).
    """
    new_params = copy.deepcopy(params)
    target_class = _object_class_value(object_class)
    rule_class = _object_class_value(rule.object_class)

    if rule_class != target_class:
        return new_params

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

    Largest min force wins, smallest max accel wins. Rule order never
    changes the result (max/min are commutative and associative).
    """
    target_class = _object_class_value(object_class)
    grip = cell.base_params["grip_force_n"]
    accel = cell.base_params["carry_accel_ms2"]

    for rule in cell.active_rules:
        if _object_class_value(rule.object_class) != target_class:
            continue
        grip = max(grip, rule.grip_force_min_n)
        accel = min(accel, rule.carry_accel_max_ms2)

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
