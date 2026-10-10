import copy
import random

import pytest

from rules.apply import CellState, RuleRejected, add_rule, apply, effective_settings, remove_rule
from rules.limits import CARRY_ACCEL_MAX_MS2, CARRY_ACCEL_MIN_MS2, CRUSH_LIMIT_N, GRIP_FORCE_MAX_N, GRIP_FORCE_MIN_N


# (name, rule_kwargs, params, object_class, expect)
# expect is either a dict of the resulting params, or "reject"
TABLE = [
    ("raises min force when current is lower", dict(grip_force_min_n=14.0, carry_accel_max_ms2=2.5), {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}, "glass_jar", {"grip_force_n": 14.0, "carry_accel_ms2": 2.5}),
    ("keeps current force when already higher than rule min", dict(grip_force_min_n=10.0, carry_accel_max_ms2=2.5), {"grip_force_n": 20.0, "carry_accel_ms2": 3.0}, "glass_jar", {"grip_force_n": 20.0, "carry_accel_ms2": 2.5}),
    ("lowers accel when current is higher", dict(grip_force_min_n=10.0, carry_accel_max_ms2=1.0), {"grip_force_n": 15.0, "carry_accel_ms2": 3.0}, "glass_jar", {"grip_force_n": 15.0, "carry_accel_ms2": 1.0}),
    ("keeps current accel when already lower than rule max", dict(grip_force_min_n=10.0, carry_accel_max_ms2=5.0), {"grip_force_n": 15.0, "carry_accel_ms2": 2.0}, "glass_jar", {"grip_force_n": 15.0, "carry_accel_ms2": 2.0}),
    ("class mismatch is a no-op", dict(grip_force_min_n=14.0, carry_accel_max_ms2=1.0), {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}, "steel", {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}),
    ("rejects when forced above crush limit", dict(object_class="cardboard", grip_force_min_n=13.0, carry_accel_max_ms2=2.5), {"grip_force_n": 2.0, "carry_accel_ms2": 2.0}, "cardboard", "reject"),
    ("forcing grip up to the exact datasheet max is accepted", dict(grip_force_min_n=GRIP_FORCE_MAX_N, object_class="steel", carry_accel_max_ms2=2.5), {"grip_force_n": 5.0, "carry_accel_ms2": 2.0}, "steel", {"grip_force_n": GRIP_FORCE_MAX_N, "carry_accel_ms2": 2.0}),
    ("at crush limit boundary is accepted", dict(object_class="cardboard", grip_force_min_n=12.0, carry_accel_max_ms2=2.5), {"grip_force_n": 2.0, "carry_accel_ms2": 2.0}, "cardboard", {"grip_force_n": 12.0, "carry_accel_ms2": 2.0}),
    ("rejects when forced below datasheet accel min", dict(grip_force_min_n=5.0, carry_accel_max_ms2=CARRY_ACCEL_MIN_MS2 - 0.1), {"grip_force_n": 5.0, "carry_accel_ms2": 3.0}, "glass_jar", "reject"),
    ("no-op leaves params exactly equal, not just close", dict(grip_force_min_n=1.0, carry_accel_max_ms2=10.0), {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}, "steel", {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}),
]


@pytest.mark.parametrize("name,rule_kwargs,params,object_class,expect", TABLE, ids=[t[0] for t in TABLE])
def test_apply_table(name, rule_kwargs, params, object_class, expect, rule_factory):
    overrides = dict(rule_kwargs)
    overrides.setdefault("object_class", "glass_jar")
    rule = rule_factory(**overrides)

    if expect == "reject":
        with pytest.raises(RuleRejected):
            apply(rule, params, object_class)
    else:
        result = apply(rule, params, object_class)
        assert result == expect, name


def test_apply_does_not_mutate_input_params(rule_factory):
    rule = rule_factory(grip_force_min_n=14.0, carry_accel_max_ms2=2.5)
    params = {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}
    before = copy.deepcopy(params)
    apply(rule, params, "glass_jar")
    assert params == before


def test_apply_returns_new_dict_not_same_object(rule_factory):
    rule = rule_factory(grip_force_min_n=14.0, carry_accel_max_ms2=2.5)
    params = {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}
    result = apply(rule, params, "glass_jar")
    assert result is not params


@pytest.mark.parametrize("seed", range(20))
def test_apply_is_idempotent_fixed_seeds(seed, rule_factory):
    rnd = random.Random(seed)
    grip_min = rnd.uniform(GRIP_FORCE_MIN_N, CRUSH_LIMIT_N["glass_jar"])
    accel_max = rnd.uniform(CARRY_ACCEL_MIN_MS2, CARRY_ACCEL_MAX_MS2)
    rule = rule_factory(object_class="glass_jar", grip_force_min_n=grip_min, carry_accel_max_ms2=accel_max)
    params = {
        "grip_force_n": rnd.uniform(GRIP_FORCE_MIN_N, CRUSH_LIMIT_N["glass_jar"]),
        "carry_accel_ms2": rnd.uniform(CARRY_ACCEL_MIN_MS2, CARRY_ACCEL_MAX_MS2),
    }
    once = apply(rule, params, "glass_jar")
    twice = apply(rule, once, "glass_jar")
    assert once == twice


@pytest.mark.parametrize("seed", range(20))
def test_apply_output_never_leaves_absolute_limits(seed, rule_factory):
    rnd = random.Random(seed)
    grip_min = rnd.uniform(GRIP_FORCE_MIN_N, GRIP_FORCE_MAX_N)
    accel_max = rnd.uniform(CARRY_ACCEL_MIN_MS2, CARRY_ACCEL_MAX_MS2)
    rule = rule_factory(object_class="steel", grip_force_min_n=grip_min, carry_accel_max_ms2=accel_max)
    params = {
        "grip_force_n": rnd.uniform(GRIP_FORCE_MIN_N, GRIP_FORCE_MAX_N),
        "carry_accel_ms2": rnd.uniform(CARRY_ACCEL_MIN_MS2, CARRY_ACCEL_MAX_MS2),
    }
    result = apply(rule, params, "steel")
    assert GRIP_FORCE_MIN_N <= result["grip_force_n"] <= GRIP_FORCE_MAX_N
    assert CARRY_ACCEL_MIN_MS2 <= result["carry_accel_ms2"] <= CARRY_ACCEL_MAX_MS2


# --- multi-rule (CellState) ---


def test_effective_settings_order_independent(rule_factory):
    base = {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}
    r1 = rule_factory(rule_id="r1", object_class="glass_jar", grip_force_min_n=14.0, carry_accel_max_ms2=2.5)
    r2 = rule_factory(rule_id="r2", object_class="glass_jar", grip_force_min_n=16.0, carry_accel_max_ms2=2.0)

    cell_a = CellState("A", "owner1", base, (r1, r2))
    cell_b = CellState("A", "owner1", base, (r2, r1))

    assert effective_settings(cell_a, "glass_jar") == effective_settings(cell_b, "glass_jar")
    assert effective_settings(cell_a, "glass_jar") == {"grip_force_n": 16.0, "carry_accel_ms2": 2.0}


def test_effective_settings_ignores_other_classes(rule_factory):
    base = {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}
    r1 = rule_factory(rule_id="r1", object_class="steel", grip_force_min_n=30.0, carry_accel_max_ms2=1.0)
    cell = CellState("A", "owner1", base, (r1,))
    assert effective_settings(cell, "glass_jar") == base


def test_add_rule_returns_new_cellstate(rule_factory):
    base = {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}
    cell = CellState("A", "owner1", base)
    r1 = rule_factory(rule_id="r1", object_class="glass_jar", grip_force_min_n=14.0, carry_accel_max_ms2=2.5)
    new_cell = add_rule(cell, r1)
    assert cell.active_rules == ()
    assert new_cell.active_rules == (r1,)


def test_add_rule_rejects_newer_conflicting_rule(rule_factory):
    base = {"grip_force_n": 2.0, "carry_accel_ms2": 2.0}
    cell = CellState("A", "owner1", base)
    r_lo = rule_factory(rule_id="r_lo", object_class="cardboard", grip_force_min_n=11.0, carry_accel_max_ms2=2.5)
    cell = add_rule(cell, r_lo)

    r_hi = rule_factory(rule_id="r_hi", object_class="cardboard", grip_force_min_n=13.0, carry_accel_max_ms2=2.5)
    with pytest.raises(RuleRejected):
        add_rule(cell, r_hi)

    # rejected rule never got added
    assert [r.rule_id for r in cell.active_rules] == ["r_lo"]


def test_remove_rule_restores_base_behavior(rule_factory):
    base = {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}
    cell = CellState("A", "owner1", base)
    r1 = rule_factory(rule_id="r1", object_class="glass_jar", grip_force_min_n=14.0, carry_accel_max_ms2=2.5)
    cell = add_rule(cell, r1)
    assert effective_settings(cell, "glass_jar") != base

    cell = remove_rule(cell, "r1")
    assert cell.active_rules == ()
    assert effective_settings(cell, "glass_jar") == base


def test_remove_rule_is_simple_list_removal_for_unknown_id(rule_factory):
    base = {"grip_force_n": 8.0, "carry_accel_ms2": 3.0}
    r1 = rule_factory(rule_id="r1", object_class="glass_jar", grip_force_min_n=14.0, carry_accel_max_ms2=2.5)
    cell = CellState("A", "owner1", base, (r1,))
    unchanged = remove_rule(cell, "does_not_exist")
    assert unchanged.active_rules == (r1,)
