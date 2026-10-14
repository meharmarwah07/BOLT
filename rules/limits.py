"""All numeric limits for the gripper rule system, in one place.

Every number here is a PLACEHOLDER pending real datasheet/calibration figures
from Part A (the MuJoCo simulation). Nothing outside this file should hardcode
a limit value — import from here so Part A's eventual numbers only need to
change in one place.
"""

# Datasheet hard limits for the gripper controller. Values outside these
# ranges are physically/mechanically invalid regardless of object class.
# PLACEHOLDER: pending Part A calibration.
GRIP_FORCE_MIN_N = 2.0
GRIP_FORCE_MAX_N = 40.0

CARRY_ACCEL_MIN_MS2 = 0.5
CARRY_ACCEL_MAX_MS2 = 6.0

# Per-object-class crush limit: the grip force above which the object is
# damaged. Each value must be <= GRIP_FORCE_MAX_N. Reuses Part B's object
# classes and placeholder values (fleet/llm/rule_check.py CRUSH_LIMIT_N),
# which already match CONTEXT.md section 5/7.
# PLACEHOLDER: pending Part A calibration.
CRUSH_LIMIT_N = {
    "steel": 40.0,
    "cardboard": 12.0,
    "glass_jar": 25.0,
}

OBJECT_CLASSES = tuple(CRUSH_LIMIT_N.keys())

# Phase B: the conditional rule type's grip_fraction (grip = grip_fraction
# * a noisy pre-grasp probe reading of the part's true crush limit --
# sim/cell_model.py's probe_crush_limit_reading). Unlike grip_force_min_n,
# this is a dimensionless ratio, not a force, so it needs its own range --
# reused GRIP_FORCE_MIN_N/MAX_N would be the wrong units entirely.
# PLACEHOLDER: a design choice, not a measured/agreed tolerance.
# GRIP_FRACTION_MIN=0.1 is just a sanity floor against a near-zero grip;
# GRIP_FRACTION_MAX=1.5 allows some deliberate overshoot past the raw
# reading (in case a probe systematically under-reads) while still ruling
# out an obviously-absurd request regardless of what the probe said.
GRIP_FRACTION_MIN = 0.1
GRIP_FRACTION_MAX = 1.5
