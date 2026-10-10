# Decisions and open questions

Everything below was undecided or unspecified at build time. Each entry
states the assumption made, why, and the flag/constant that changes it.

## Auto-approve: NOT implemented

A human must approve every rule via `review_cli.py approve` (or the
equivalent call to `lifecycle.transition(..., State.APPROVED, ...)`). There
is no code path that marks a rule APPROVED without an explicit reviewer and
note. This matches CONTEXT.md section 4 step 5 (Check 2 is human review) and
was not treated as a flag -- auto-approve is out of scope entirely for v1,
not just defaulted off.

## Receiver opt-in: `policy.REQUIRE_RECEIVER_OPT_IN`, default `False`

CONTEXT.md section 7 ("Whether the receiving owner must also opt in is
undecided") and the spec both leave this open. Default is `False`: the
source owner's `sharing_enabled` flag alone is sufficient for a cross-owner
share, matching the spec's "Check order" step 3 ("different owner and
source owner's sharing flag ON -> allowed (v0)"). Set
`policy.REQUIRE_RECEIVER_OPT_IN = True` to also require
`owners[to_cell.owner].receiving_opt_in` before allowing a cross-owner
share. Both settings are covered by `tests/test_policy.py::test_permission_matrix`.

## Owner revoke: `policy.ALLOW_OWNER_REVOKE`, default `True`

Spec says this is a flag but doesn't give a default. Defaulted to `True`
(owners can revoke an active shared rule on their own cells) since the
alternative -- no way to undo a bad share once it's ACTIVE -- seemed like
the worse failure mode for a safety-relevant system. Set to `False` to
disable `policy.revoke()` entirely (it raises `RuntimeError`).

## What crosses the owner boundary: `policy.export_for_sharing()`

Only the fields on `schema.Rule` itself ever cross an owner boundary --
never raw failure reports, replay trial data, or free-text evidence. This
was explicit in the spec ("strips everything except the rule itself"), not
really a judgment call, but recorded here because it's a hard boundary a
later change might be tempted to weaken (e.g. to help a receiving owner's
reviewer understand "why"). If that's ever wanted, it should be a separate,
clearly-labeled "evidence summary" export, not a loosening of this one.

## Revalidate on receipt: `policy.receive_shared_rule()`

Every rule received from another owner is re-run through `validate()`
before it can become ACTIVE on the receiving cell -- other owners' rules are
untrusted input, same as an LLM's raw output. Not a flag; this one is not
optional. `receive_shared_rule()` raises `ValueError` on failure.

## `schema_version` defaulting

`schema.Rule.schema_version` defaults to the current `SCHEMA_VERSION`
("1") when absent from input, rather than treating a missing
`schema_version` as a validation error. Rationale: Part B's existing
Solver prompt doesn't ask the model to emit `schema_version` (see
`INTEGRATION.md`), and there's currently only one version in existence, so
"absent" and "current" mean the same thing in practice. An *explicit*
mismatched value (e.g. `"2"`) is still rejected, naming the supported
version, per spec. If a second schema version is ever introduced, this
default should be revisited -- it would then silently treat an old,
unversioned payload as the newest version.

## Placeholder numbers (pending Part A)

Every number in `limits.py` -- `GRIP_FORCE_MIN_N`/`MAX_N`,
`CARRY_ACCEL_MIN_MS2`/`MAX_MS2`, and the three `CRUSH_LIMIT_N` entries -- is
a placeholder pending real calibration from Part A's MuJoCo simulation.
Values were carried over unchanged from Part B's existing local copy
(`fleet/llm/rule_check.py`, pre-shim) and CONTEXT.md section 5, which
already agreed on them, rather than inventing new ones. `OBJECT_CLASSES`
(`steel`, `cardboard`, `glass_jar`) likewise matches Part B/CONTEXT.md
exactly rather than the example classes (egg/can/bottle/box) from the
original task description, since Part B's code and tests already depend on
these three names.

## Field naming: kept Part B's names

`grip_force_min_n`, `carry_accel_max_ms2`, `object_class`, `rule_id`,
`rationale`, `source_report_ids` are kept as-is from Part B's local copy,
rather than renamed to match the original task description's
`min_grip_force_n`/`max_carry_accel` wording. `source_report_ids` is the
only name for cited report IDs -- no `cited_report_ids` alias was added.
New fields not present in Part B's copy: `schema_version`, `proposed_by`,
`created_at`. See `INTEGRATION.md` for how the shim bridges this gap for
Part B's existing (unmodified) prompt and replies.
