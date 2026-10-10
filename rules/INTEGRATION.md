# Integration with Part B (fleet/llm)

`fleet/llm/rule_check.py` is now a thin shim over `rules.validate` /
`rules.schema.Rule`. `solver.py`, `fake_server.py`, `try_replies.py`, and
`test_solver.py` were **not edited**. Baseline (`python test_solver.py`,
run from `fleet/llm/` with cwd there) is unchanged: `6/6 passed`, before and
after the shim.

## Format differences found (Part B's local copy vs. `rules.schema.Rule`)

| | Part B's old `rule_check.Rule` | `rules.schema.Rule` |
|---|---|---|
| New required fields | -- | `schema_version`, `proposed_by`, `created_at` |
| `object_class` | `Literal["steel","cardboard","glass_jar"]` | `Enum` built from `limits.OBJECT_CLASSES` (same 3 values, same wire format: a plain string) |
| Range/crush-limit checks | pydantic `Field(ge=.., le=..)` + a `model_validator` on the model itself | moved out of the model into `validate.py`, as plain semantic checks, so every error (not just the first) can be collected and given field-name + bad-value + range + fix-text |
| Error reporting | `parse_reply` returns only `e.errors()[0]`, pydantic's default wording | `rules.validate()` returns *every* error with custom LLM-actionable wording; the shim still only surfaces the first one to preserve `parse_reply`'s old contract |
| Everything else (field names, units, `rationale`, `source_report_ids`, `NoSafeRule`) | identical | identical |

Everything else matches exactly: `rule_id`, `grip_force_min_n`,
`carry_accel_max_ms2`, `rationale` (max 300 chars), `source_report_ids`
(min 1), and the 3 object classes with the same crush limits.

## What the shim maps

`fleet/llm/rule_check.py`:

1. Keeps its own local `NoSafeRule` model unchanged -- Part C doesn't own
   that shape, it's purely a Solver/Part-B concept.
2. For anything that isn't `no_safe_rule`, before validating, fills in
   `proposed_by` (default `"llm-solver"`) and `created_at` (now, UTC) **if
   absent**, since Part B's `SYSTEM_PROMPT` in `solver.py` doesn't ask the
   model to emit these new Part C fields. `schema_version` is not filled in
   explicitly -- it defaults inside `schema.Rule` itself (see
   `DECISIONS.md`).
3. Calls `rules.validate()` and returns only `result.errors[0]` as the
   error string, and a `rules.schema.Rule` instance (not the old local
   class) on success -- preserving `parse_reply(text) -> (obj, error)`'s
   exact signature and the "first error only" behavior `solver.py`'s retry
   loop and `test_solver.py`'s assertions depend on.
4. The one pydantic default-message dependency in `test_solver.py` --
   asserting `"Extra inputs are not permitted"` appears in stderr for the
   all-attempts-bad case -- is preserved deliberately:
   `validate.py`'s `extra_forbidden` message is worded
   `"<field>: Extra inputs are not permitted; a Rule must only contain..."`,
   so it carries that exact legacy substring while still being a complete,
   field-naming, LLM-actionable message in its own right.

## What Part B's owner needs to change (not urgent, not done here)

- `solver.py`'s `SYSTEM_PROMPT` doesn't ask the model for `schema_version`,
  `proposed_by`, or `created_at`. The shim papers over this by defaulting
  them, but a real vLLM-served model will never see these fields in its
  instructions either. If/when Part C's new fields need to come from the
  model itself (e.g. for a real `proposed_by` identifying which model/run
  proposed it), update `SYSTEM_PROMPT` and this default-filling shim
  behavior should be removed at the same time.
- Nothing else. `rule_check.py`'s public surface (`parse_reply`, the
  `NoSafeRule` shape) is unchanged.

## Integration test

`fleet/rules/tests/test_integration.py` builds a rule the way
`fake_server.py`'s `GOOD_REPLY` and `solver.py` actually produce one (through
the shim's `parse_reply`), confirms it independently passes
`rules.validate`, and walks it through PROPOSED -> Check 1 -> APPROVED ->
shared to a same-owner cell -> ACTIVE, with `effective_settings` reflecting
the change. A second test confirms `fake_server.py`'s `BAD_REPLY` (the
extra-field case) is rejected by both the shim and `rules.validate`
directly.
