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

from pydantic import BaseModel, ConfigDict, Field

from rules.limits import OBJECT_CLASSES

SCHEMA_VERSION = "1"

# Built from limits.OBJECT_CLASSES so the two never drift apart.
ObjectClass = Enum("ObjectClass", {name.upper(): name for name in OBJECT_CLASSES}, type=str)


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    rule_id: str
    object_class: ObjectClass
    grip_force_min_n: float
    carry_accel_max_ms2: float
    proposed_by: str
    created_at: dt.datetime
    # Display only. Never read by validate(), apply(), or any other logic.
    rationale: str = Field(max_length=300)
    source_report_ids: list[str] = Field(min_length=1)
