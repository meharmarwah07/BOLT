# FLEET — Project Context

Version v0.1 · 2026-10-08 · Owner: Mehar · Status: Day 1 (environment + simulation basics)

Paste this whole file into your LLM (or keep it as `CONTEXT.md` in the repo root and tell the LLM to read it first).

---

## 0. Instructions to the LLM reading this

You are helping a small student team build FLEET for a hackathon. Follow these rules.

1. **Teach while building.** The team wants to learn side by side. Give one small step at a time. Explain what it does and why in 2–4 sentences. Wait for the user's output before the next step. Never write a whole module in one go.
2. **Say where each thing goes.** Distinguish clearly between "run this in the terminal" and "put this in a file". Give files as `cat > name.py << 'EOF' ... EOF` blocks so users cannot paste Python into bash by mistake (this has already happened).
3. **Stay inside your assigned part** (section 6). Do not change a shared contract (section 7) without telling the whole team.
4. **Never fabricate results.** Everything is simulation. Label every number as a simulation result. If something was not measured, say "not measured".
5. **Be skeptical and say "untested"** for anything not actually run on our machines. Flag weak assumptions instead of agreeing.
6. **Keep answers short.** The users prefer terse, direct, evidence-based answers, no flattery.
7. **Out of scope for now:** UI/dashboard, real robots, blockchain, anything not in section 4.

---

## 1. One-paragraph summary

FLEET is a cloud system in which simulated factory robot cells learn from each other's failures. When one cell fails (drops a part), a monitor writes a structured failure report. An open LLM served on AMD GPUs proposes a machine-checkable rule (for example "grip glass jars with at least 14 N and carry acceleration at most 2.5 m/s²"). The rule must pass two checks: automated parallel simulation replays, then a human reviewer. An owner-permission gate then decides which other cells may receive it. We measure whether other cells' failure rates drop. The whole thing runs in the cloud and is a **simulation** project, not a real-robot one.

## 2. Hackathon constraints

- Event: AMD Developer Hackathon: ACT III, online, **Oct 12–18, 2026**. Track: **Intelligent Industry**.
- Everything must run in the cloud, with a meaningful workload on AMD hardware shown in the working product.
- AMD Developer Cloud credits (about $100) come via the AMD AI Developer Program. Mehar is registered there. Hackathon portal registration (lablab.ai) for the team is still to do.
- **Not yet verified:** exact submission deadline and time, judging criteria, the Rule Book's position on work built before Oct 12, which GPU tier the credits give. Check these before relying on them.

## 3. Problem and why it matters

Robots in different cells repeat the same failures independently. When a failure is understood in one cell, that knowledge rarely reaches the others in a form that is safe to apply. Our claim: failures can be converted into **precise, structured, verified rules** and pushed to other robots **only if** they pass verification and the owners' permissions allow it.

Honest positioning (research done earlier): this is a **gap, not a moat**. Related work exists (AutoRT robot constitutions, REFLECT failure explanation, ExpeL rule extraction, RoboGuard, Fleet-Merge, Viam staged rollouts) but none combine failure → structured rule → dual verification → permission-gated cross-owner propagation. Candidate long-term moats: a cross-owner shared rule library (network effect, cold start), trust/certification, a history of which rules held up. Do not build for the long-term vision now. The hackathon deliverable is a measurable simulation pipeline.

## 4. The product flow (v1)

1. **Cell A** runs pick-and-place. A hidden low-friction object class (`glass_jar`) makes it drop parts.
2. **Monitor** (plain rule-based code, not an LLM) turns each failure into a structured **FailureReport** (section 7).
3. **Solver** (open LLM on AMD, served with vLLM on ROCm) reads reports and outputs either a candidate **Rule** in strict JSON, or `{"no_safe_rule": true, "reason": "..."}`.
4. **Check 1: simulation replays.** Run the candidate in parallel on many seeds. The rule must (a) reduce drops for its target class, (b) not regress any other class or metric, (c) respect the gripper datasheet hard limits.
5. **Check 2: human review.** A reviewer sees the rule in plain language plus the replay evidence, and approves or rejects. The decision is logged.
6. **Permission gate.** Cell B (same owner as A) receives the rule directly. Cell C (different owner) receives it only if sharing is allowed.
7. **Measure** B's and C's drop rate on the affected class, with and without the rule.
8. **No-solution path:** if the Solver finds no safe rule, the event is escalated to a human as "failure seen, no safe rule".

```
Cell A fails -> Monitor -> FailureReport -> Solver (LLM on AMD)
   -> Candidate Rule -> Check 1 (replays) -> Check 2 (human)
   -> Permission gate -> Cell B (same owner) / Cell C (other owner, if allowed)
   -> Measure drop rate with vs without rule
```

## 5. The simulated world (the "cell")

- Physics: **MuJoCo** (Python, `pip install mujoco`, version 3.15.0), run headless from Python. Rendering, when needed, uses EGL (`MUJOCO_GL=egl`), which worked in one test environment and is untested on AMD cloud.
- Cell model (planned): gantry-style pick-and-place with a two-finger gripper driven by force-controlled finger motors.
- Object classes with different mass, friction and crush limits: `steel`, `cardboard`, `glass_jar`. Pad friction is set low. MuJoCo contact friction is the max of the two geoms' coefficients.
- Controller parameters we vary: `grip_force_n`, `carry_accel_ms2`.
- Every trial randomizes initial conditions with a seed. **Train seeds** are used for developing and for Check 1. **Held-out seeds (e.g. 1000–1199)** are used only for the final measurement.
- All numbers below are placeholders to be calibrated in simulation: gripper hard limits `grip_force_n` 2–40, `carry_accel_ms2` 0.5–6.0; crush limits `glass_jar` 25 N, `cardboard` 12 N, `steel` 40 N.

## 6. Parts and ownership

Owners are not assigned yet. Each part is learned and built by one person, with the others reviewing.

**Part A: Simulation (MuJoCo).**
Build the cell: floor, object, gripper, controller, trial runner, drop/crush detection, seeded randomization. Output: a function `run_trial(object_class, params, seed) -> TrialResult`, plus a FailureReport generator (the Monitor). Steps: box falls (done) → gripper → pick and carry → drop failure on `glass_jar` → seeded batch → report.

**Part B: LLM on AMD.**
Check the GPU and ROCm, serve an open model with vLLM, call it from Python, force schema-valid JSON output. Output: `solve(failure_reports) -> Rule | NoSafeRule`. Includes sampling multiple candidates. First step: confirm the GPU is visible (`rocm-smi`) and vLLM starts.

**Part C: Rules engine.**
Rule schema, validator (type checks, datasheet limits), applying a rule to controller settings, permission gate, append-only log, reviewer decision record. Output: `validate(rule)`, `apply(rule, controller_params, object_class)`, `can_share(rule, from_cell, to_cell)`, with unit tests.

**Shared step 0 (everyone):** agree on the contracts in section 7 before anyone writes real code.

## 7. Shared contracts (proposed v0, change only by team agreement)

**FailureReport** (true friction is hidden and deliberately not included; the Solver must infer from evidence):
```json
{
  "report_id": "fr_0001",
  "cell_id": "A", "owner_id": "owner1",
  "object_class": "glass_jar",
  "controller": {"grip_force_n": 8.0, "carry_accel_ms2": 3.0},
  "outcome": "drop",
  "drop_phase": "carry",
  "slip_mm_before_drop": 12.4,
  "est_mass_kg": 0.35,
  "seed": 42
}
```

**Rule** (what the Solver outputs; strict schema):
```json
{
  "rule_id": "r_0001",
  "object_class": "glass_jar",
  "grip_force_min_n": 14.0,
  "carry_accel_max_ms2": 2.5,
  "rationale": "short text",
  "source_report_ids": ["fr_0001"]
}
```
No-solution output: `{"no_safe_rule": true, "reason": "..."}`.

**ReplayResult** (output of Check 1):
```json
{
  "rule_id": "r_0001", "n_trials": 200,
  "per_class": {
    "glass_jar": {"drops_before": 0, "drops_after": 0, "crushes_before": 0, "crushes_after": 0,
                  "mean_cycle_s_before": 0.0, "mean_cycle_s_after": 0.0}
  },
  "passed": false, "reasons": []
}
```

**ReviewDecision:** `{rule_id, reviewer, decision: "approve"|"reject", time, note}`.
**ShareRecord:** `{rule_id, from_cell, to_cell, allowed, reason}`.

**Permission model (v0):** each cell has an `owner_id`. A and B belong to `owner1`; C belongs to `owner2`. A rule moves to a cell of a different owner only if the source owner's sharing flag is true. Whether the receiving owner must also opt in is **undecided**.

## 8. What we measure

1. **Drop rate on the affected class (`glass_jar`) for Cell B and Cell C, before vs after the rule.** At least 200 held-out-seed trials per condition, report with confidence intervals.
2. **Number of harmful candidate rules blocked by Check 1.** Define "harmful" up front (crushes the part, exceeds datasheet limits, regresses another class or cycle time). The candidate pool is the LLM's sampled candidates; if we also inject deliberately bad rules to test the gate, label them as injected.
3. **Replays per minute, serial vs parallel, on the AMD instance.**

Caveats to state in any write-up: it is a simulation we designed, with a hidden-friction failure we created, so the improvement is real within the sim but is not evidence about real factories. Do not claim otherwise.

## 9. Scope and non-goals

In scope: sections 4–8. Out of scope for v1: UI/dashboard (deferred, "later when required"), real robots, blockchain (the audit trail is a plain append-only log), many object classes beyond three, learning-based controllers, multi-gripper physics realism.

## 10. Environment and status

- Machines: Windows laptops with **WSL2 Ubuntu** (one 22.04 / Python 3.10, one 26.04 / Python 3.14.4); one teammate on native Linux Mint. Editor: Antigravity IDE with a Remote-WSL extension, opening `~/fleet`.
- Always use a virtual environment: `cd ~/fleet && source .venv/bin/activate`. The prompt must show `(.venv)`. (Plain `pip install` fails on newer Ubuntu with "externally-managed-environment".)
- Two prompts exist: `PS C:\...>` is Windows PowerShell, `name@host:~/fleet$` is Ubuntu. Run Python and MuJoCo only in the Ubuntu one.
- **Done:** Step 1 (`step1_box.py`, a box falling onto a floor). Verified: final height 0.0299 m, and the t=0.20 s height matches the analytical free-fall value for MuJoCo's integrator (0.2979 m).
- **Not yet done:** viewer or GIF rendering on WSL, gripper (Step 2), anything on the AMD cloud, role assignment.

Proposed repo layout:
```
fleet/
  CONTEXT.md            this file
  sim/                  Part A: cell.py, objects.py, run_trials.py
  llm/                  Part B: serve.sh, solver.py, prompts/
  rules/                Part C: schema.py, validator.py, apply.py, tests/
  data/                 failure_reports/, rules/, logs/
```

## 11. Proposed schedule (rough, adjust to reality)

- Day 1: environment, MuJoCo basics, agree contracts, GPU/ROCm check on AMD cloud.
- Day 2: gripper and pick-and-carry; vLLM serving; rule schema and validator.
- Day 3: reproducible `glass_jar` drops; FailureReport generator; Solver returns schema-valid JSON.
- Day 4: Check 1 replays (serial, then parallel); `apply` rule to controller.
- Day 5: human review step (command-line or file-based is enough), permission gate, logging.
- Day 6: held-out measurements for B and C; the three numbers.
- Day 7: demo script, submission write-up, fixes.

## 12. Known risks and unknowns

- **"Meaningful AMD workload" may be only the LLM.** MuJoCo physics runs on CPU. Parallel replays on the AMD instance would use CPU cores unless we adopt MJX (GPU physics via JAX), which is untested on ROCm. Decide early what we claim for the replay speedup.
- AMD GPU tier, ROCm and vLLM setup, and EGL rendering on the cloud instance are all untested.
- The LLM may output rules that look plausible but fail replays. That is expected and is exactly what Check 1 measures.
- Hackathon Rule Book may restrict pre-built work. Unverified.
- Grip/crush thresholds are placeholders, so the whole "harmful rule" story depends on calibrating them honestly.
- Timeline is tight (about a week). If behind, cut: Cell C and the permission gate demo come before anything cosmetic; cut the no-solution path last.

## 13. Glossary

- **Cell:** one simulated robot workstation with its own owner.
- **Rule:** a structured, machine-checkable setting change for a specific object class.
- **Check 1 / Check 2:** automated simulation replay / human approval.
- **Held-out seeds:** random seeds never used while developing, used only for the final measurement.
- **Replay:** re-running trials with and without a candidate rule.
- **vLLM / ROCm:** LLM serving software / AMD's GPU compute stack.
- **MuJoCo:** physics simulator. **MJX:** its JAX-based GPU version (not used yet).

---

## 14. Part C — Rules engine (built)

Lives in `fleet/rules/` (Python package `rules`), on git branch `rules`.
Owns the official `Rule` schema and limits, `validate()`, `apply()` +
multi-rule `CellState`, the rule lifecycle (event-sourced from a
hash-chained audit log), the owner permission gate (`can_share()`), and a
review CLI (`python -m rules.review_cli`). `fleet/llm/rule_check.py` is now
a thin shim that delegates to it -- Part B's `solver.py`, `fake_server.py`,
and `test_solver.py` are unchanged and still pass (`6/6`). See
`fleet/rules/README.md` for setup/run instructions,
`fleet/rules/DECISIONS.md` for every placeholder/flag and why, and
`fleet/rules/INTEGRATION.md` for exactly what differs from Part B's original
local copy.

Run its tests: `cd fleet/rules && ../../.venv/Scripts/python.exe -m pytest -q`.
