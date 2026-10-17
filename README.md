# BOLT

**Simulated factory robot cells learn from each other's failures, with a safety gate and a human in the loop.** Team Pink Panthers, AMD Developer Hackathon: ACT III.

**SIMULATION ONLY.** Everything runs in a MuJoCo simulator. Nothing controls a real robot, and no number here describes a real deployment.

Demo: https://boltamddemo.streamlit.app  |  Flowchart: demo/assets/bolt_flowchart.png

## What it does
Cell A drops a glass jar with hidden low friction. A Monitor writes a FailureReport. An open LLM served through AMD Token Factory (DeepSeek-V4-Flash and Gemma-4-31B-IT) proposes a rule (grip force and carry acceleration) or says no_safe_rule. The rule must pass Check 1 (simulator replay on 300 seeds; Wilson 95% upper bound on failure <= 0.4 and zero crushes), then Check 2 (human review), then the owner permission gate. Only then are Cells B and C measured with and without the rule. Every decision goes to an append-only, hash-chained audit log.

## Results (simulation, held-out seeds, n=100 per cell)
- Without a rule: 100% of attempts drop the jar.
- With a gate-passing rule: 18-30% drop, 0 crushes (10 of 10 revision-2 rules; 3 distinct policies; all exact McNemar p < 1e-20 vs baseline).
- Cell C with sharing denied: unchanged.
- A naive 22.5 N grip heuristic matched one policy on success but crushed the jar in 4 of 100 trials. BOLT's measured advantage is safety, not a higher success rate.
- Policies are not ranked: at n=100 the uncertainty is about +/-9 points.

## Limitations (read these)
- **Revision 2 is post hoc.** The pre-registered result (revision 1, Check 1 at 100 seeds) was 0/10 gate-passing rules. Widening Check 1 to 300 seeds was decided after seeing that. Revision 2's 10/10 is reported alongside the registered 0/10, never instead of it. Why 300 seeds passes where 100 did not is a hypothesis (tighter interval), not tested.
- **Check 2 was automated in the evaluation.** The review step is implemented; in the held-out run, approval was automatic (non-interactive), not a person.
- One object class (glass jar), one simulator, 18-30% residual failure. Not evidence about real robots.
- Held-out seed ranges are single-use and spent. Protocol and records are in docs/.

## AMD usage
All revision-2 proposals came from models served via AMD Token Factory (OpenAI-compatible endpoint). Request counts and wall times are on the demo's "Where AMD is used" tab, computed from the recorded loop files.

## Provenance
Some components (simulator, rule schema, solver, checker) were committed before the Oct 12 build window opened. The pipeline, pre-registered evaluation, revisions, audit log and demo were built during the window. See git history.

## Run the demo
pip install -r requirements.txt
streamlit run demo/app.py
The demo needs no API key and makes no LLM calls. API keys for the pipeline are read from environment variables (FLEET_LLM_*) and are never committed.

## Docs
docs/REVISION2_RESULT.md (results), docs/REVISED_RUNS_RESULT.md (registered 0/10), docs/REVISION2.md (post hoc note), docs/SUCCESS_CRITERION.md and docs/FINAL_PROTOCOL.md (pre-registration). Demo details: demo/README.md.

MIT License. See LICENSE.
