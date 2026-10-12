# Part A Status — Simulation (MuJoCo)

All 10 Part A deliverables from CONTEXT.md section 6 are done and verified
by actually running the code, not just written.

## What's built (sim/)

- `cell_model.py` — shared model: floor, two-finger force-controlled
  gripper, 3 object classes, smooth half-sine lift/carry motion profile.
- `pick_test.py` — interactive viewer tool for watching a single trial.
- `run_trial.py` — `run_trial(object_class, params, seed) -> TrialResult`,
  the real deliverable API. Headless, ~44ms/trial.
- `batch.py` — seeded batch runner. `TRAIN_SEEDS = range(0, 200)`,
  `HELD_OUT_SEEDS = range(1000, 1200)` per CONTEXT.md section 5. Reports
  drop rate with a 95% Wilson confidence interval. `--parallel` verified
  to give bit-identical results to serial, ~3x real speedup on 8 cores.
- `monitor.py` — FailureReport generator. Only fires on actual failures
  (`outcome != "ok"`). Output checked field-for-field against CONTEXT.md
  section 7's schema. True friction is never included.

## Object classes (our simulation design choices, not measurements)

| class | mass_kg | friction | crush_limit_n |
|---|---|---|---|
| steel | 0.5 | 0.8 | 40 (matches rules/limits.py) |
| cardboard | 0.3 | 0.9 | 12 (matches rules/limits.py) |
| glass_jar | 0.4 | **0.15 (hidden)** | 25 (matches rules/limits.py) |

`crush_limit_n` is unchanged from the already-agreed `rules/limits.py`
placeholders. `mass_kg` and `friction` are new — nothing upstream defined
them, so these are design choices picked to produce a meaningful
demonstration, not calibrated real-world data.

## Real simulation results (headless runs — measured, not assumed)

- glass_jar: drops below ~2-3N grip, holds 3-25N, crushes ≥30N — the only
  class with a real "Goldilocks window," matching the project's premise.
- steel: drops ≤3N, holds 5-40N, never crushes in the tested range.
- cardboard: holds even at 2N, crushes ≥15N (fails by crushing, not
  dropping — never observed a cardboard drop).
- grip=2.5N, glass_jar, 200 train seeds: drop rate 32.5%, 95% CI
  [26.4%, 39.3%].
- grip=15N, glass_jar: 0% drop rate on both train seeds AND the held-out
  1000-1199 range — consistent across two independent seed sets.

## Known gaps / needs team input

1. Monitor emits `outcome: "crush"` in addition to `"drop"`.
   `llm/solver.py` currently only reads `report_id` off a FailureReport,
   not `outcome`/`drop_phase`, and every existing fixture only uses
   `"drop"`. Crush reports won't be meaningfully handled downstream until
   `solver.py` is updated. Flagging per CONTEXT.md rule 3 ("don't change a
   shared contract without telling the team") — `outcome` has no enum in
   the schema, so this isn't a break, just new.
2. Never run end-to-end through the real `llm/solver.py` or
   `rules/validate.py` — only verified our output matches their expected
   shapes structurally. Full three-part pipeline integration is untested.
3. Parallel speedup (~3x) is real but measured on this dev machine only —
   untested on the AMD cloud instance.
4. A handful of bugs were found and fixed along the way that are worth
   knowing about if anyone else touches sim/: finger pads initially
   overlapped the table geom, a bang-bang motion profile had an
   acceleration discontinuity that broke grip, directly overwriting
   qpos/qvel to script the gripper's motion bypassed MuJoCo's own
   constraint solver and caused slow unrecoverable slip, and
   `noslip_iterations` (off by default in MuJoCo) allowed persistent
   numerical creep even on a grip well within the friction cone.

All numbers above are simulation results from this machine, not
extrapolated or assumed.
