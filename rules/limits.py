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
