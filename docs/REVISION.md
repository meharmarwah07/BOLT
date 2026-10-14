# Prompt/feedback revision (registered before any revised run)

Why: 0/5 real-model loop runs produced a gate-passing rule (docs/loop_baseline_prerevision_summary.txt).
Observed: 13/45 attempts declined (no_safe_rule), concentrated in rounds 2-3; declines cite an assumed friction (~4.1 N holding force) and conclude "not a grip problem"; the feedback echoes the model's own decline reasons into the next round; solver_system.txt permits a decline when "the drop was not caused by grip force or acceleration"; examples.json contains a worked decline that uses the quasi-static formula.
Hypothesis (untested): the model anchors on a textbook friction estimate and the loop reinforces it.

Changes (prompt and feedback only):
1. solver_system.txt: friction is not observable and varies per part; reports at the nominal setting saturate, so they carry no information about how far grip is from sufficient; replay results override any physics estimate; decline only if tried grips show failure on both sides of every untested value.
2. examples.json: replace the cardboard decline example with a worked example (made-up numbers, not any sim class's true values) where a formula says grip is sufficient but replay shows otherwise and the model proposes a higher grip.
3. loop.py::_build_feedback: do not echo the model's own decline reasons; add a table of all tried grips (sorted) with drop rate, crush count and combined failure.
4. Optional, arm B only: strip at most one clean ```json code fence before strict schema validation (opt-in flag).

Not changed: Check 1 gate and thresholds, selection key, MAX_ROUNDS=3, candidates per round, sim, seeds, throttling.
Not disclosed to the model: the friction range, the crush-jitter distribution.

Arms (same protocol, 5 runs each on train seeds 0-4 reports / Check 1 train seeds): A = DeepSeek-V4-Flash; B = Gemma via Token Factory if available. All runs reported, including no-rule runs and infrastructure errors.
Availability check (15 Oct, before any revised run): the endpoint's /models lists Gemma-4-31B-IT, so arm B uses that exact name.
