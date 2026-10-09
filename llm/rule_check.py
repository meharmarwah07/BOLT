import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# PLACEHOLDER limits (contract v0). Part C owns the real ones.
CRUSH_LIMIT_N = {"steel": 40.0, "cardboard": 12.0, "glass_jar": 25.0}


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")      # unknown fields = reject
    rule_id: str
    object_class: Literal["steel", "cardboard", "glass_jar"]
    grip_force_min_n: float = Field(ge=2, le=40)
    carry_accel_max_ms2: float = Field(ge=0.5, le=6)
    rationale: str = Field(max_length=300)
    source_report_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def not_above_crush_limit(self):
        limit = CRUSH_LIMIT_N[self.object_class]
        if self.grip_force_min_n > limit:
            raise ValueError(f"grip_force_min_n {self.grip_force_min_n} exceeds "
                             f"crush limit {limit} for {self.object_class}")
        return self


class NoSafeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    no_safe_rule: Literal[True]
    reason: str


def parse_reply(text):
    """Turn raw LLM text into a Rule or NoSafeRule. Returns (obj, error)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"not valid JSON ({e.msg})"
    if not isinstance(data, dict):
        return None, "JSON is not an object"
    model = NoSafeRule if "no_safe_rule" in data else Rule
    try:
        return model.model_validate(data), None
    except ValidationError as e:
        first = e.errors()[0]
        return None, f"{first['loc']}: {first['msg']}"
