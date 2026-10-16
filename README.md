# BOLT safety-gate demo

A single-page Streamlit app that walks through BOLT's safety gate end to end: a real simulated
failure, real LLM proposals from recorded runs (across two registered revisions), a live run of
the real automated check at a seed count you pick, a real human-review/audit-log walkthrough on a
real accepted rule, a cross-owner permission gate, and the hash-chained audit log that records
every decision.

**SIMULATION.** This is a demo of a research pipeline that runs entirely inside a MuJoCo physics
simulator. Nothing here controls a real robot, and no number on the page describes a real
warehouse, incident, or deployment.

## How to run

```bash
pip install -r requirements.txt
streamlit run demo/app.py
```

No API key is read, set, or required anywhere in this app. It does not call an LLM at any point.

## What's cached vs. what's live

| Section | Source |
|---|---|
| 1. Failure (Cell A) | Cached: a real `FailureReport` JSON generated once by `sim/run_trial.py` + `sim/monitor.py` (`demo_data/failure_report_seed0.json`, see `demo_data/gen_failure_report.py`), plus pre-rendered MP4s copied from `demo_assets/` |
| 2. Proposals (recorded) | Cached, **both revisions**: revision 1 (registered, 100-seed gate, 0/10 — `demo_data/loop_records/`, `demo_data/REVISED_RUNS_RESULT.md`) and revision 2 (post hoc, 300-seed gate, 10/10 — `demo_data/loop_records/revised2/`, `demo_data/revision2_rules/`, `demo_data/REVISION2_RESULT.md`, `REVISION2_FINAL_REPORT.md`, `REVISION2_FINAL_SET.md`, `REVISION2.md`). The held-out result table and a "why revision 1 passed nothing and revision 2 did" panel are both here |
| 3. Try it yourself | **Precomputed for 5 documented (grip, accel) inputs at 300 seeds** (`demo_data/check1_300_precomputed.json`, generated once by a real `check1()` call — see `demo_data/gen_check1_300_precomputed.py`), **live for everything else**, gated behind an explicit button whose label states an estimated time range and that timing depends on the host — measured on one machine: ~11s at 16 CPU cores, ~38-43s at 2 cores, ~87-89s at 1 core (300 seeds), ~30s at 1 core (100 seeds) |
| 4. Human review (Check 2) | **Live, on a real rule**: `r_a8e42ed5_2`, the first (not best) of revision 2's 10 gate-passing rules, with its real recorded audit trail (`demo_data/loop_records/revised2/loop_20261016T171221Z_audit.jsonl`) copied into a session-local working copy and replayed through the real `rules/lifecycle.py` state machine — no fabricated data |
| 5. Permission gate | **Live**: the real `rules.policy.can_share()` on the same real rule from section 4, logging to the same session-local audit copy |
| 6. Audit log | **Live**: the real `rules.audit.read_events()` / `verify_chain()` on that session's working copy (seeded from real history), plus a hand-tampered copy to show the chain break |
| 7. Registered results | Cached: `demo_data/registered_results.json`, the template registered under the *original* (revision-1-only) protocol, which never had a gate-passing rule to evaluate. Revision 2's real held-out result is in section 2, not here |

## Honest limitations

- Revision 2 (section 2) is **post hoc**: the decision to widen Check 1's seed count to 300 was
  made only after seeing revision 1 end 0/10 at 100 seeds. The registered **primary** result stays
  revision 1's 0/10 (`docs/REVISED_RUNS_RESULT.md`); revision 2's 10/10 is reported alongside it,
  not in place of it, per `docs/REVISION2.md`.
- The app states a **hypothesis** for why revision 2 passed where revision 1 didn't (a tighter
  Wilson-interval at 300 seeds vs. 100) but does not re-derive it — nobody re-ran revision 1's exact
  failing candidates at 300 seeds to confirm this specific mechanism, as opposed to, e.g., the model
  simply proposing different grip values the second time. Section 3 lets you probe this yourself.
- Section 5's permission gate has no "receiving owner grants access" control in the current
  `rules/policy.py` (`REQUIRE_RECEIVER_OPT_IN` defaults off, per `rules/DECISIONS.md`) — the app
  shows the control that actually exists (the source owner's `sharing_enabled` flag) rather than
  inventing one.
- Section 4's rule was really auto-approved by `pipeline/evaluate_heldout.py` for its non-interactive
  held-out measurement, not by an actual human reviewer — the app says so plainly rather than
  implying a person signed off on it.
- The audit log and Check 1 run in sections 3-6 are real, but scoped to a temp file / this process
  for the duration of your browser session — nothing is written back into `pipeline/runs/` or
  `llm/runs/`.
- Section 3's live path never auto-runs: a single-core measurement of a 300-seed call took ~87-89s
  (well over what a page should trigger without asking), so any input not in the 5-entry precomputed
  table requires an explicit click on a button that states an estimated time range up front. The
  precomputed table itself only covers 300 seeds at the 5 documented (grip, accel) pairs — any other
  input, or any 100-seed run, goes through the live, explicitly-confirmed path.

## License

MIT, same as the rest of this repository — see [LICENSE](LICENSE).

---

BOLT by Pink Panthers -- simulation only
