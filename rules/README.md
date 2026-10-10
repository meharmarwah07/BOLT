# Part C — Rules engine (`fleet/rules`, Python package `rules`)

Owns the official `Rule` schema, limits, validation, applying rules to
controller settings, the rule lifecycle, the cross-owner permission gate,
and a tamper-evident audit log. Replaces `fleet/llm/rule_check.py` as the
source of truth; that file is now a thin shim (see `INTEGRATION.md`).

## Setup (one-time, or after a fresh clone)

```bash
cd fleet/rules
../../.venv/Scripts/python.exe -m pip install -r requirements.txt
../../.venv/Scripts/python.exe -m pip install -e .
```

The editable install makes `import rules` work from any working directory
(in particular from `fleet/llm/`, where Part B's scripts are run from) --
without it, `rule_check.py`'s `from rules... import ...` will fail.

## Running the tests

```bash
cd fleet/rules
../../.venv/Scripts/python.exe -m pytest -q
```

Or, from anywhere once the editable install is done: `pytest -q fleet/rules`.

## Running the demo

```bash
python -m rules.demo_e2e
```

Walks a rule through PROPOSED -> Check 1 -> APPROVED -> shared to a
same-owner cell (ACTIVE, settings change) -> denied to a different-owner
cell (BLOCKED, settings unchanged) -> verifies the audit chain.

## Running the review CLI

```bash
python -m rules.review_cli --log <path-to-audit.jsonl> --store <rule-store-dir> list
python -m rules.review_cli --log <path> --store <dir> show <rule_id>
python -m rules.review_cli --log <path> --store <dir> approve <rule_id> --reviewer NAME --note TEXT
python -m rules.review_cli --log <path> --store <dir> reject <rule_id> --reviewer NAME --note TEXT
```

`--store` is a directory holding one JSON file per rule (written by whatever
proposes the rule, e.g. `review_cli.store_rule()`); the audit log only
records events, not the rule payload.

## Checking Part B still works

```bash
cd fleet/llm
../../.venv/Scripts/python.exe test_solver.py
```

Should print `6/6 passed`, unchanged from before Part C existed.

## Module map

| Module | Owns |
|---|---|
| `limits.py` | every numeric limit (all PLACEHOLDER, pending Part A) |
| `schema.py` | the `Rule` model, `SCHEMA_VERSION`, `ObjectClass` |
| `validate.py` | `validate(rule) -> ValidationResult`, all-errors-at-once |
| `apply.py` | `apply()`, `CellState`, `effective_settings`, `add_rule`/`remove_rule` |
| `audit.py` | append-only hash-chained JSONL log, `verify_chain()` |
| `lifecycle.py` | the rule/cell state machine, event-sourced from the log |
| `policy.py` | `can_share()`, owner flags, export/receive, `revoke()` |
| `explain.py` | deterministic plain-language rule description |
| `review_cli.py` | `python -m rules.review_cli ...` |

See `DECISIONS.md` for every placeholder/assumption and the flag that
changes it, and `INTEGRATION.md` for exactly what differs from Part B's
original local copy and what the shim papers over.
