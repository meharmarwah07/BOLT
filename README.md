# BOLT

**Simulated factory robot cells learn from each other's failures, with a safety gate and a human in the loop.** Team Pink Panthers, AMD Developer Hackathon: ACT III.

**SIMULATION ONLY.** Everything runs in a MuJoCo simulator. Nothing controls a real robot, and no number here describes a real deployment.

Demo: https://boltamddemo.streamlit.app  |  Flowchart: demo/assets/bolt_flowchart.png

## What it does
Cell A drops a glass jar with hidden low friction. A Monitor writes a FailureReport. An open LLM served through AMD Token Factory (DeepSeek-V4-Flash and Gemma-4-31B-IT) proposes a rule (grip force and carry acceleration) or says no_safe_rule. The rule must pass Check 1 (simulator replay on 300 seeds; Wilson 95% upper bound on failure <= 0.4 and zero crushes), then Check 2 (human review), then the owner permission gate. Only then are Cells B and C measured with and without the rule. Every decision goes to an append-only, hash-chained audit log.

## How the loop works
pipeline/loop.py runs a bounded search: up to MAX_ROUNDS=3 rounds of CANDIDATES_PER_ROUND=3 candidates each, re-proposing with feedback after each round. As soon as a round produces at least one Check-1-passing candidate, the loop picks the one with the lowest Wilson 95% upper bound of combined (drop-or-crush) failure, tie-broken by lower crush count, then larger crush margin, then smaller slowdown.

## Results (simulation, held-out seeds, n=100 per cell)
- Without a rule: 100% of attempts drop the jar.
- With a gate-passing rule: 18-30% drop, 0 crushes (10 of 10 revision-2 rules; 3 distinct policies; all exact McNemar p < 1e-20 vs baseline).
- With sharing denied the gate blocks the rule, so Cell C stays at baseline by construction (the baseline batch is reused).
- A naive 22.5 N grip heuristic matched one policy on success but crushed the jar 2/100 in Cell B and 4/100 in Cell C; task success comparable to the gate-passing policies (McNemar p 0.08 to 1.0). BOLT's measured advantage is safety, not a higher success rate.
- Policies are not ranked: at n=100 the uncertainty is about +/-9 points.

## Limitations (read these)
- **Revision 2 is post hoc.** The pre-registered result (revision 1, Check 1 at 100 seeds) was 0/10 gate-passing rules. Widening Check 1 to 300 seeds was decided after seeing that. Revision 2's 10/10 is reported alongside the registered 0/10, never instead of it. Why 300 seeds passes where 100 did not is a hypothesis (tighter interval), not tested.
- **Check 2 was automated in the evaluation.** rules/review_cli.py is the real human-review CLI and rules/explain.py produces a deterministic plain-language description for the reviewer; the held-out evaluation auto-approved Check 2 instead (non-interactive, logged as APPROVED with a note).
- pipeline/dev_check.py is a dev-only probe that never touches train or held-out seeds.
- One object class (glass jar), one simulator, 18-30% residual failure. Not evidence about real robots.
- Held-out seed ranges are single-use and spent. Protocol and records are in docs/.

## AMD usage
All revision-2 proposals came from models served via AMD Token Factory (OpenAI-compatible endpoint). Request counts and wall times are on the demo's "Where AMD is used" tab, computed from the recorded loop files.

## Provenance
Committed before the Oct 12 build window (Oct 10-11): initial simulator files, the rule checker, the solver with bounded retries and prompts, a Wilson-interval helper, and the rule schema, lifecycle, permission gate and audit log. Built during the window (Oct 12-18): simulator completion and fixes, Token Factory integration, Check 1 and its v2 gate, the feedback loop, the held-out evaluator, the pre-registered runs and both revisions, and the demo. See git history for exact dates.

## Run the demo
pip install -r requirements.txt
streamlit run demo/app.py
The demo needs no API key and makes no LLM calls. API keys for the pipeline are read from environment variables (FLEET_LLM_*) and are never committed.

## Docs
docs/REVISION2_RESULT.md (results), docs/REVISED_RUNS_RESULT.md (registered 0/10), docs/REVISION2.md (post hoc note), docs/SUCCESS_CRITERION.md and docs/FINAL_PROTOCOL.md (pre-registration). Demo details: demo/README.md.

MIT License. See LICENSE.
