"""The Rule model: the single official shape for a gripper controller rule.

Field names and the grip-force/carry-accel units match Part B's existing
local copy (fleet/llm/rule_check.py) and CONTEXT.md section 7, so the LLM
solver's prompt and retry loop don't need to change. See INTEGRATION.md for
every difference from Part B's copy.

Only structural checks (types, required/extra fields, enum membership, string
length, list length) live here as pydantic constraints, so pydantic's own
error aggregation gives us "all structural errors at once" for free. Semantic
checks that need custom, LLM-actionable error text (numeric ranges, crush
limit, schema_version) are NOT pydantic constraints — they live in
validate.py so every error in a single validate() call can get a tailored
message.
"""
import datetime as dt
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from rules.limits import OBJECT_CLASSES

SCHEMA_VERSION = "1"

# Built from limits.OBJECT_CLASSES so the two never drift apart.
ObjectClass = Enum("ObjectClass", {name.upper(): name for name in OBJECT_CLASSES}, type=str)

# Phase B: a second rule_type alongside the original "fixed" grip. A
# "conditional" rule's realized grip is grip_fraction * a noisy pre-grasp
# probe reading of the part's true crush limit (sim/cell_model.py's
# probe_crush_limit_reading), decided per trial, not a single fixed
# number -- so it uses grip_fraction instead of grip_force_min_n.
# carry_accel_max_ms2 means the same thing (an accel cap) for both types.
RuleType = Literal["fixed", "conditional"]


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    rule_id: str
    object_class: ObjectClass
    rule_type: RuleType = "fixed"
    # Exactly one of these two is set, enforced semantically in
    # validate.py (not structurally here, since pydantic has no clean
    # "exactly one of" constraint and the per-field error messages validate.py
    # already gives are more LLM-actionable than pydantic's own). Both
    # Optional, defaulting to None, so every EXISTING "fixed" rule
    # (rule_type omitted, grip_force_min_n always set) is unaffected.
    grip_force_min_n: Optional[float] = None  # "fixed" rules only
    grip_fraction: Optional[float] = None  # "conditional" rules only
    carry_accel_max_ms2: float
    proposed_by: str
    created_at: dt.datetime
    # Display only. Never read by validate(), apply(), or any other logic.
    rationale: str = Field(max_length=300)
    source_report_ids: list[str] = Field(min_length=1)
