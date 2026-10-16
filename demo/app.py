"""BOLT safety-gate demo (Streamlit, single page, cached-only).

SIMULATION. No part of this app needs or reads an API key, and it never
calls an LLM: proposals shown in section 2 are cached JSON copied out of
finished pipeline/runs/loop_*.json files (both revision 1 and the post
hoc revision 2), and section 3's "Try it yourself" button runs the
project's real, registered Check 1 (pipeline/check1.py) on a seed count
you pick (100 or 300) -- plain simulation replay, not a model call.

Everything numeric shown here is either (a) loaded from a file in
demo_data/ with its source file named next to it, or (b) computed live,
in this process, by pipeline/check1.py's check1() on the seed count
chosen in section 3. Section 4's "human review" walkthrough uses a real
rule (r_a8e42ed5_2) and its real recorded audit trail, not a fabricated
example -- there is no synthetic/hand-built data left in this app.
"""
import datetime as dt
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import streamlit as st

# ---- Make the real project packages importable, the same way pipeline/loop.py does ----
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
PIPELINE_DIR = os.path.join(ROOT, "pipeline")
SIM_DIR = os.path.join(ROOT, "sim")
LLM_DIR = os.path.join(ROOT, "llm")
for _p in (ROOT, PIPELINE_DIR, SIM_DIR, LLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check1 as check1_mod  # pipeline/check1.py -- the registered Check 1

from rules import audit, policy
from rules.apply import CellState
from rules.lifecycle import State, current_state, transition
from rules.schema import Rule

DEMO_DATA = Path(APP_DIR).parent / "demo_data"
ASSETS = Path(APP_DIR) / "assets"

st.set_page_config(page_title="BOLT safety gate (demo)", layout="wide")

# ============================================================================
# Banner + intro
# ============================================================================
st.warning("SIMULATION -- every number on this page is cached data or a live MuJoCo replay on this machine. Nothing here is a robot, a factory, or a real incident.", icon="⚠️")
st.title("BOLT: a safety gate for a robot that proposes its own fixes")
st.markdown(
    """
A warehouse pick-and-place cell fails at something (drops or crushes an object). An LLM proposes a
fix -- a tweak to grip force and carry acceleration -- for that failure. Before that fix can ever touch
a real cell, it has to survive two gates:

- **Check 1 (automated):** the proposed rule is replayed hundreds of times through the real physics
  simulator, on seeds the model never saw at proposal time. If it doesn't measurably reduce drops
  without crushing the object or breaking something that used to work, it's rejected -- automatically,
  before any human looks at it.
- **Check 2 (human):** only after Check 1 passes does a person approve or reject the rule. Approval is
  what lets the rule cross from one robot cell to another -- and even then, only if the cell's owner has
  agreed to receive rules from other owners.

Every one of those decisions is written to an append-only, hash-chained audit log. This page walks
through all of it end to end, using real code from this repository -- not a mockup.
"""
)

# ============================================================================
# 1. Failure (Cell A)
# ============================================================================
st.header("1. A real failure (Cell A)")
st.markdown(
    "The JSON below is a real `FailureReport`, generated once by `sim/run_trial.py` + `sim/monitor.py` "
    "(never hand-typed) at the nominal controller setting, **train seed 0**, object class `glass_jar`, "
    "and saved to `demo_data/failure_report_seed0.json` (see `demo_data/gen_failure_report.py` for the "
    "exact call). True friction is never included in a report -- the solver has to infer trouble from "
    "`slip_mm_before_drop`, same as a real monitor would."
)

failure_report_path = DEMO_DATA / "failure_report_seed0.json"
if failure_report_path.exists():
    failure_report = json.loads(failure_report_path.read_text())
    col1, col2 = st.columns([1, 1])
    with col1:
        st.json(failure_report)
        st.caption(f"source: demo_data/{failure_report_path.name}")
    with col2:
        video_path = ASSETS / "glass_jar_nominal_drop_seed5045.mp4"
        if video_path.exists():
            st.video(str(video_path))
            st.caption(
                "source: demo_assets/glass_jar_nominal_drop_seed5045.mp4 -- a separately pre-rendered "
                "illustrative recording at the same nominal setting and object class, dev seed 5045 "
                "(NOT the JSON report's seed 0 -- no video was rendered for seed 0 itself)."
            )
        else:
            st.info("No pre-rendered video found at demo/assets/glass_jar_nominal_drop_seed5045.mp4.")
else:
    st.error(f"Missing {failure_report_path} -- run `PYTHONPATH=. venv/bin/python3 demo_data/gen_failure_report.py` once.")

with st.expander("Also available: a rule holding the same drop, and a side-by-side (dev seed 5045, illustrative)"):
    for name, caption in (
        ("glass_jar_rule_hold_seed5045.mp4", "the same setup, with a passing rule's grip/accel applied -- holds"),
        ("glass_jar_side_by_side_seed5045.mp4", "nominal vs. rule-applied, side by side"),
    ):
        p = ASSETS / name
        if p.exists():
            st.video(str(p))
            st.caption(f"source: demo_assets/{name} -- {caption}")

st.divider()

# ============================================================================
# 2. Proposals (recorded)
# ============================================================================
st.header("2. Proposals an LLM actually made (recorded, not replayed live)")
st.markdown(
    "There have been **two rounds** of revised runs against this failure. **Revision 1** is the "
    "pre-registered one (`docs/REVISION.md`): 10 runs, Check 1 gated on **100** train seeds, result "
    "**0/10** gate-passing rules. **Revision 2** is a **post hoc** follow-up (`docs/REVISION2.md`), run "
    "after revision 1 ended 0/10, changing exactly one thing -- Check 1 gated on **300** train seeds "
    "instead of 100 -- and nothing else (same gate, thresholds, selection key, prompts, solver, models). "
    "Its result: **10/10** gate-passing rules. Revision 1's 0/10 remains the **primary registered "
    "result**; revision 2 is reported alongside it, clearly labeled as post hoc."
)

GATE_LABEL_REV1 = "gate: v2.1 (combined failure Wilson upper bound <= 0.4, crush must be 0), 100 train seeds -- revision 1 (registered)"
GATE_LABEL_REV2 = "gate: v2.1 (combined failure Wilson upper bound <= 0.4, crush must be 0), 300 train seeds -- revision 2 (post hoc)"

# ---- 2a. Registered outcomes: both revisions ----
st.subheader("Registered outcomes: revision 1 and revision 2")

rev1_path = DEMO_DATA / "REVISED_RUNS_RESULT.md"
rev2_result_path = DEMO_DATA / "REVISION2_RESULT.md"
rev2_set_path = DEMO_DATA / "REVISION2_FINAL_SET.md"

col_rev1, col_rev2 = st.columns(2)
with col_rev1:
    st.markdown("**Revision 1 (registered, 100 seeds)**")
    if rev1_path.exists():
        outcome_text = rev1_path.read_text()
        commit_m = re.search(r"commit ([0-9a-f]{7,40})", outcome_text)
        arm_lines = re.findall(
            r"Arm ([AB]) \(([^)]+)\): (\d+) rules?; (\d+) rejected by Check 1, (\d+) declined, (\d+) errors?; (\d+) HTTP requests?",
            outcome_text,
        )
        st.caption(f"source: demo_data/{rev1_path.name} (copied from docs/REVISED_RUNS_RESULT.md)")
        if commit_m:
            st.markdown(f"Runs executed at commit `{commit_m.group(1)}`.")
        st.error(
            "**0/10 gate-passing rules.** Per `docs/SUCCESS_CRITERION.md` item F (criterion F), the "
            "final held-out evaluation was **not run** on this result -- seeds 3000-3199 stayed unused "
            "at this point."
        )
        for arm, model, rules, rejected, declined, errors, http in arm_lines:
            st.caption(f"Arm {arm} ({model}): {rules} rules; {rejected} rejected, {declined} declined, {errors} error(s)")
        with st.expander("Full file"):
            st.markdown(outcome_text)
    else:
        st.info("demo_data/REVISED_RUNS_RESULT.md not found.")

with col_rev2:
    st.markdown("**Revision 2 (post hoc, 300 seeds)**")
    if rev2_result_path.exists():
        rev2_text = rev2_result_path.read_text()
        commit_m2 = re.search(r"commit ([0-9a-f]{7,40})", rev2_text)
        st.caption(f"source: demo_data/{rev2_result_path.name} (copied from docs/REVISION2_RESULT.md)")
        if commit_m2:
            st.markdown(f"Evaluated at commit `{commit_m2.group(1)}`.")
        st.success(
            "**10/10 gate-passing rules** (arm A 5/5, arm B 5/5) -- all 10 also meet "
            "`docs/SUCCESS_CRITERION.md` item D on the final held-out seeds (3000-3199, used once)."
        )
        st.caption("3 distinct (grip, accel) policies among the 10 rules -- see the held-out table below.")
        with st.expander("Full file"):
            st.markdown(rev2_text)
    else:
        st.info("demo_data/REVISION2_RESULT.md not found.")

st.warning(
    "**Revision 2 is post hoc.** It was decided to widen Check 1's seed count only after seeing that "
    "revision 1 ended 0/10 -- that is a real deviation from the \"decide before you look\" discipline "
    "the rest of this project otherwise follows, and `docs/REVISION2.md` says so plainly. The registered "
    "**primary** result stays revision 1's 0/10; revision 2's 10/10 is reported alongside it, not in "
    "place of it."
)

st.divider()

# ---- 2b. Held-out result (revision 2) ----
st.subheader("Held-out result (revision 2, seeds 3000-3199, used once)")
if rev2_result_path.exists():
    rev2_text = rev2_result_path.read_text()
    m = re.search(r"## Outcome.*?(?=\n## )", rev2_text, re.DOTALL)
    if m:
        st.markdown(m.group(0))
    st.error(
        "**Simulation only, one object class (`glass_jar`).** Residual failure across the three "
        "policies is **18-30%** even after the rule -- this is not a solved problem. These are **3 "
        "distinct findings, not 10 independent successes** (several rules share the same (grip, accel) "
        "policy and give identical results). The policies are **not ranked** against each other -- "
        "differences among them are within sampling noise (~+/-9 points at n=100). The **naive 0.9x-crush "
        "(22.5 N) heuristic** is statistically comparable on task success to the **20.0 N / 0.5 m/s^2** "
        "policy in Cell C (p=0.27) -- but the heuristic **crushed the jar in 4 of 100 runs** there, while "
        "every one of the 10 accepted rules crushed **0**. Cell C under the realistic default (sharing "
        "**denied**, owner's flag off) is unchanged from baseline; the table above uses the "
        "sharing-**allowed** counterfactual, per criterion D(ii)."
    )
    with st.expander("Full docs/REVISION2_RESULT.md"):
        st.markdown(rev2_text)
else:
    st.info("demo_data/REVISION2_RESULT.md not found.")

st.divider()

# ---- 2c. Recorded proposals, per revision, grouped by arm ----
def _render_recorded_runs(loop_files, gate_label):
    runs_by_model = defaultdict(list)
    for lf in loop_files:
        run = json.loads(lf.read_text())
        runs_by_model[run["model"]].append((lf, run))

    if not loop_files:
        st.info("No recorded loop files found.")
        return

    for model in sorted(runs_by_model):
        st.markdown(f"**Arm: {model}**")
        for lf, run in sorted(runs_by_model[model], key=lambda t: t[1]["timestamp"]):
            passed_label = "CHECK1_PASSED (a rule was accepted)" if run.get("chosen_rule") else f"NO_SAFE_RULE ({run.get('no_safe_rule_reason')})"
            st.markdown(f"Run {run['timestamp']} -- {run['model']} -- {passed_label}")
            st.caption(
                f"source: demo_data/loop_records/{Path(lf).relative_to(DEMO_DATA / 'loop_records')} -- "
                f"object_class={run.get('object_class')}, is_stub={run.get('is_stub')} -- {gate_label}"
            )

            rows = []
            decline_reasons = []
            for a in run["attempts"]:
                rule = a.get("rule")
                c1 = a.get("check1")
                if a.get("no_safe_rule_reason"):
                    verdict = "declined"
                    first_reason = "(see decline reasons below)"
                    decline_reasons.append((a["round"], a["candidate_index"], a["no_safe_rule_reason"]))
                elif a.get("error"):
                    verdict = "error"
                    first_reason = a["error"]
                elif c1 is not None:
                    verdict = "PASSED" if c1["passed"] else "FAILED"
                    first_reason = c1["reasons"][0] if c1["reasons"] else ""
                else:
                    verdict = "?"
                    first_reason = ""
                rows.append(
                    {
                        "round": a["round"],
                        "candidate": a["candidate_index"],
                        "grip_force_min_n": rule.get("grip_force_min_n") if rule else None,
                        "carry_accel_max_ms2": rule.get("carry_accel_max_ms2") if rule else None,
                        "check1_verdict": verdict,
                        "first_reason": first_reason,
                    }
                )
            st.dataframe(rows, width="stretch", hide_index=True)

            if decline_reasons:
                with st.expander(f"Decline reasons ({len(decline_reasons)}) -- the model's own text, verbatim"):
                    for rnd, cand, reason in decline_reasons:
                        st.markdown(f"**round {rnd}, candidate {cand}:** {reason}")


loop_records_dir = DEMO_DATA / "loop_records"
loop_files_rev1 = sorted(loop_records_dir.glob("loop_*.json")) if loop_records_dir.exists() else []
loop_files_rev2 = sorted((loop_records_dir / "revised2").glob("loop_*.json")) if (loop_records_dir / "revised2").exists() else []

st.subheader("Recorded proposals: revision 1 (registered, 100 seeds, 0/10)")
_render_recorded_runs(loop_files_rev1, GATE_LABEL_REV1)

st.divider()

st.subheader("Recorded proposals: revision 2 (post hoc, 300 seeds, 10/10)")
_render_recorded_runs(loop_files_rev2, GATE_LABEL_REV2)

st.divider()

# ---- 2d. "Why revision 1 passed nothing and revision 2 did" -- derived only from the recorded ----
# Check 1 results above for revision 1's ladder; revision 2's half is a stated hypothesis, not a
# re-derivation (that would require re-running revision 1's exact candidates at 300 seeds, which
# this demo doesn't do -- see section 3 for a live 300-seed run on seeds of your own choosing).
st.subheader("Why revision 1 passed nothing and revision 2 did")
loop_files = loop_files_rev1  # the ladder below is revision 1's (100-seed) data only


def _combined_failure_table(loop_files):
    """(grip, accel) -> (combined_failure_rate, crush_count_after), derived purely by arithmetic
    on fields already stored in demo_data/loop_records/*.json (drop_rate_after, crush_count_after,
    n_trials) -- no new simulation, no extrapolation beyond what each file already recorded. Runs
    are deterministic given (object_class, grip, accel, seeds), so the same (grip, accel) pair
    recorded in more than one run is checked for agreement, not averaged."""
    seen = {}
    conflicts = []
    for lf in loop_files:
        run = json.loads(lf.read_text())
        for a in run["attempts"]:
            rule, c1 = a.get("rule"), a.get("check1")
            if not rule or not c1:
                continue
            key = (rule["grip_force_min_n"], rule["carry_accel_max_ms2"])
            n = c1["n_trials"]
            n_drop = round(c1["drop_rate_after"] * n)
            crush = c1["crush_count_after"]
            combined = round((n_drop + crush) / n, 2) if n else None
            row = (combined, crush)
            if key not in seen:
                seen[key] = row
            elif seen[key] != row:
                conflicts.append((key, seen[key], row, lf.name))
    return seen, conflicts


combined_table, _conflicts = _combined_failure_table(loop_files)

if combined_table:
    st.markdown(
        "**Revision 1, at 100 seeds:** across every candidate actually replayed in revision 1's runs, "
        "the grip that avoids most drops sits right next to the grip that starts crushing the jar. Below "
        "20 N, the jar keeps slipping out of the gripper; by 22 N it already crushes on some seeds, and "
        "by 25 N (the datasheet crush limit) it crushes on nearly half of them. There is no grip value "
        "in between where the combined drop-or-crush failure rate is both low *and* confidently below "
        "the gate's 0.4 ceiling with zero crushing -- which is why the v2.1 gate never passed a rule in "
        "revision 1."
    )
    # A small "ladder" through the recorded grip values, one representative (grip, accel) row per
    # distinct grip (preferring accel=1.0, the nominal accel, when it was tried at that grip) --
    # every number in it is read, or arithmetic-derived, straight from demo_data/loop_records/*.json.
    by_grip = defaultdict(dict)
    for (grip, accel), (combined, crush) in combined_table.items():
        by_grip[grip][accel] = (combined, crush)
    ladder_rows = []
    for grip in sorted(by_grip):
        accels = by_grip[grip]
        accel = 1.0 if 1.0 in accels else sorted(accels)[0]
        combined, crush = accels[accel]
        ladder_rows.append(
            {"grip_force_min_n": grip, "carry_accel_max_ms2": accel, "combined_failure_rate": combined, "crush_count_after": crush}
        )
    st.dataframe(ladder_rows, width="stretch", hide_index=True)
    st.caption(
        "Observed on train seeds 0-99, one (grip, accel) row per distinct grip, taken only from the "
        "recorded Check 1 results in demo_data/loop_records/ -- combined_failure_rate = "
        "(drops + crushes) / trials, computed from each file's own drop_rate_after and "
        "crush_count_after. No new simulation was run to produce this table."
    )
else:
    st.info("No recorded Check 1 results with both a rule and a check1 outcome were found.")

st.markdown(
    "**Revision 2, at 300 seeds: the same (grip, accel) region passed.** Nothing else about the gate, "
    "thresholds, prompts, solver, or models changed between the two revisions -- the only change was "
    "the train-seed count (100 -> 300), per `docs/REVISION2.md`."
)
st.info(
    "**Likely reason (stated plainly, not re-derived here): a tighter sampling interval.** Check 1's "
    "gate compares a Wilson 95% *upper bound* against a fixed 0.4 ceiling, not the raw point estimate -- "
    "at 100 seeds that interval is wide enough that a candidate whose TRUE combined-failure rate is just "
    "under 0.4 can still have its upper bound land just over it by chance; tripling the seed count to "
    "300 narrows that interval substantially, which can turn some of those same borderline candidates "
    "into clean passes without the part's real behavior changing at all. **This is a hypothesis about "
    "why, not a finding** -- revision 2's prompts/solver/gate were otherwise identical to revision 1's, "
    "and nobody re-ran revision 1's exact failing candidates at 300 seeds to confirm this mechanism "
    "specifically (as opposed to, e.g., the LLM simply proposing different grip values this time around). "
    "That check was **not done** as part of either revision. Section 3 below lets you run the real gate "
    "at both seed counts yourself, live, if you want to probe this further."
)

st.divider()

# ============================================================================
# 3. Try it yourself (live, CPU)
# ============================================================================
st.header("3. Try it yourself -- run the real Check 1")
st.markdown(
    "Pick a grip force and carry acceleration for `glass_jar`, and a seed count. This builds a real "
    "`rules.schema.Rule` and runs it through `pipeline/check1.py`'s `check1()` -- the exact function "
    "`pipeline/loop.py` calls for every LLM-proposed candidate -- a real MuJoCo replay, not a lookup, "
    "for anything not already precomputed below. **No boundary value from an earlier run is assumed to "
    "still hold**; it is re-measured, not recalled, for any input run live. **Timing depends on the "
    "host** this app is running on -- on one real measurement, a single 300-seed call took ~11s with "
    "16 CPU cores available, ~38-43s restricted to 2 cores, and ~87-89s restricted to 1 core; a 100-seed "
    "call took ~30s on 1 core. A shared or constrained server can land anywhere in that range or beyond "
    "it, which is exactly why this page asks before running one live."
)

def _load_precomputed_300():
    path = DEMO_DATA / "check1_300_precomputed.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {(e["grip_force_min_n"], e["carry_accel_max_ms2"]): e for e in data["entries"]}


PRECOMPUTED_300 = _load_precomputed_300()


def _render_check1_result(passed, n_trials, drops_before, drops_after, crush_count, other_unchanged,
                           drop_rate_after, drop_rate_after_ci95, combined_failure_rate, combined_failure_ci95,
                           reasons, source_caption):
    if passed:
        st.success("Check 1: **PASSED**")
    else:
        st.error("Check 1: **FAILED**")
    st.caption(source_caption)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("drops before", drops_before)
    m2.metric("drops after", drops_after)
    m3.metric("crushes after", crush_count)
    m4.metric("other classes unchanged", str(other_unchanged))
    st.write(
        f"drop rate after: **{drop_rate_after:.3f}** "
        f"(Wilson 95% CI: [{drop_rate_after_ci95[0]:.3f}, {drop_rate_after_ci95[1]:.3f}])"
    )
    if combined_failure_rate is not None:
        st.write(
            f"combined drop-or-crush failure rate: **{combined_failure_rate:.3f}** "
            f"(Wilson 95% CI upper bound {combined_failure_ci95[1]:.3f})"
        )
    st.markdown("**Reasons:**")
    for r in reasons:
        st.markdown(f"- {r}")


seed_choice = st.radio(
    "Check 1 train seeds",
    options=[100, 300],
    index=1,
    format_func=lambda n: f"{n} (revision 1 gate, seeds 0-99)" if n == 100 else f"{n} (revision 2 gate, seeds 0-299, post hoc)",
    horizontal=True,
)
col_a, col_b = st.columns(2)
with col_a:
    grip = st.number_input(
        "grip_force_min_n (N)", min_value=5.00, max_value=30.00, value=20.00, step=0.01, format="%.2f"
    )
with col_b:
    accel = st.slider("carry_accel_max_ms2 (m/s^2)", min_value=0.5, max_value=6.0, value=0.5, step=0.1)
st.caption(
    "The gate compares a Wilson 95% upper bound to a fixed ceiling, not the raw point estimate -- "
    "nearby grip values, or the same value at a different seed count, can land on different sides of "
    "that boundary. This is sampling noise interacting with the gate, not a property of the part."
)

precomputed_key = (round(grip, 2), round(accel, 1))
precomputed_entry = PRECOMPUTED_300.get(precomputed_key) if seed_choice == 300 else None

if precomputed_entry is not None:
    st.markdown(
        f"**PRECOMPUTED** -- grip={precomputed_entry['grip_force_min_n']} N, "
        f"accel={precomputed_entry['carry_accel_max_ms2']} m/s^2 at 300 train seeds is one of this demo's "
        "5 documented inputs. Shown instantly from a cached result; nothing ran just now."
    )
    _render_check1_result(
        precomputed_entry["passed"], precomputed_entry["n_trials"], precomputed_entry["drops_before"],
        precomputed_entry["drops_after"], precomputed_entry["crush_count"], precomputed_entry["other_classes_unchanged"],
        precomputed_entry["drop_rate_after"], precomputed_entry["drop_rate_after_ci95"],
        precomputed_entry["combined_failure_rate"], precomputed_entry["combined_failure_ci95"],
        precomputed_entry["reasons"],
        "source: demo_data/check1_300_precomputed.json -- a real check1() call on real train seeds "
        "0-299, run once ahead of time (see demo_data/gen_check1_300_precomputed.py), not a live run.",
    )
else:
    est = "~1-2 minutes" if seed_choice == 300 else "~30-60 seconds"
    run_clicked = st.button(
        f"Run live Check 1 on train seeds 0-{seed_choice - 1} (about {est} on a shared server -- timing depends on the host)",
        type="primary",
    )
    if run_clicked:
        rule = Rule(
            rule_id=f"r_demo_tryit_{int(dt.datetime.now().timestamp())}",
            object_class="glass_jar",
            grip_force_min_n=grip,
            carry_accel_max_ms2=accel,
            proposed_by="demo-user",
            created_at=dt.datetime.now(dt.timezone.utc),
            rationale=f"Demo try-it-yourself rule: grip_force_min_n={grip} N, carry_accel_max_ms2={accel} m/s^2 for glass_jar.",
            source_report_ids=["demo-try-it-yourself"],
        )
        t0 = dt.datetime.now()
        with st.spinner(f"Replaying {seed_choice} train seeds, before and after, via the real simulator (sim/run_trial.py)... timing depends on the host (measured 11s-89s across 16/2/1 CPU cores on one machine; could be slower or faster here)."):
            outcome = check1_mod.check1(rule, range(seed_choice))
        elapsed_s = (dt.datetime.now() - t0).total_seconds()
        st.session_state["tryit_outcome"] = outcome
        st.session_state["tryit_elapsed_s"] = elapsed_s
        st.session_state["tryit_seed_choice"] = seed_choice

    if "tryit_outcome" in st.session_state:
        outcome = st.session_state["tryit_outcome"]
        rec = outcome.record
        _render_check1_result(
            outcome.passed, outcome.n_trials, rec.drops_before, rec.drops_after, rec.crush_count,
            rec.other_classes_unchanged, outcome.drop_rate_after, outcome.drop_rate_after_ci95,
            outcome.combined_failure_rate, outcome.combined_failure_ci95, outcome.reasons,
            f"LIVE run, this process, just now -- took {st.session_state['tryit_elapsed_s']:.1f}s on "
            f"this machine, {outcome.n_trials} train seeds (0-{st.session_state['tryit_seed_choice'] - 1}).",
        )

st.divider()

# ============================================================================
# 4. Human review (Check 2)
# ============================================================================
st.header("4. Human review (Check 2)")

EXAMPLE_RULE_ID = "r_a8e42ed5_2"
real_rule_json_path = DEMO_DATA / "revision2_rules" / "accepted_rule_20261016T171221Z.json"
real_audit_source_path = DEMO_DATA / "loop_records" / "revised2" / "loop_20261016T171221Z_audit.jsonl"

if real_rule_json_path.exists() and real_audit_source_path.exists():
    accepted_data = json.loads(real_rule_json_path.read_text())
    example_rule = Rule.model_validate(accepted_data["rule"])

    st.markdown(
        f"This is the **real** rule `{EXAMPLE_RULE_ID}` -- the first (not the best) of revision 2's 10 "
        "gate-passing rules, chosen for this walkthrough for exactly that reason: it's the first one "
        "that reached CHECK1_PASSED, from run A1 (`demo_data/revision2_rules/accepted_rule_20261016T171221Z.json`). "
        "Its lifecycle below is the **real audit trail** copied from "
        f"`demo_data/loop_records/revised2/{real_audit_source_path.name}` (originally "
        "`pipeline/runs/loop_20261016T171221Z_audit.jsonl`) into a session-local working copy, so this "
        "page can demonstrate the real `rules/lifecycle.py`/`rules/audit.py` functions on it without "
        "mutating the canonical file. Nothing here is hand-built."
    )
    st.json(accepted_data["rule"])

    if "demo_audit_path" not in st.session_state:
        fd, path = tempfile.mkstemp(prefix="fleet_demo_audit_", suffix=".jsonl")
        os.close(fd)
        with open(path, "w") as f_out, open(real_audit_source_path) as f_in:
            f_out.write(f_in.read())
        st.session_state["demo_audit_path"] = path

    audit_path = st.session_state["demo_audit_path"]
    check2_state = current_state(audit_path, EXAMPLE_RULE_ID)
    st.write(f"Rule `{EXAMPLE_RULE_ID}` lifecycle state (replayed from the real log): **{check2_state.value if check2_state else 'None'}**")

    if check2_state == State.CHECK1_PASSED:
        c1, c2 = st.columns(2)
        if c1.button("Approve (actor: alice)"):
            transition(audit_path, EXAMPLE_RULE_ID, State.APPROVED, "alice")
            st.rerun()
        if c2.button("Reject (actor: alice)"):
            transition(audit_path, EXAMPLE_RULE_ID, State.REJECTED, "alice")
            st.rerun()
    elif check2_state == State.APPROVED:
        approved_events = [e for e in audit.read_events(audit_path) if e.rule_id == EXAMPLE_RULE_ID and e.event == "APPROVED"]
        note = approved_events[-1].details.get("note", "") if approved_events else ""
        actor = approved_events[-1].actor if approved_events else "?"
        st.info(
            f"**Already decided: APPROVED**, by actor `{actor}` -- which, honestly, was not a real human "
            f"reviewer. The real audit log's own note says: \"{note}\". `pipeline/evaluate_heldout.py` "
            "auto-approves Check 2 for its non-interactive held-out measurement; a real deployment "
            "requires an actual person via `rules/review_cli.py`. (Re-deciding a rule once it leaves "
            "CHECK1_PASSED is not a legal transition in `rules/lifecycle.py` -- there is no Approve/Reject "
            "button to show here for that reason, not because this page is hiding one.)"
        )
    else:
        st.info(f"Already decided: {check2_state.value if check2_state else 'None'}.")
else:
    st.info(
        f"The real audit trail for `{EXAMPLE_RULE_ID}` is not available in demo_data -- expected "
        f"{real_rule_json_path} and {real_audit_source_path}."
    )

st.divider()

# ============================================================================
# 5. Permission gate
# ============================================================================
st.header("5. Permission gate: does an approved rule cross owners?")

if real_rule_json_path.exists() and "demo_audit_path" in st.session_state:
    example_rule = Rule.model_validate(json.loads(real_rule_json_path.read_text())["rule"])
    audit_path = st.session_state["demo_audit_path"]

    st.markdown(
        f"Using the same real rule `{EXAMPLE_RULE_ID}` above: Cell A and Cell B belong to `owner1`; Cell "
        "C belongs to `owner2` (same illustrative cell setup as `rules/demo_e2e.py`; the three cells "
        "themselves are a demo construct -- the real audit trail already contains real SHARED/BLOCKED "
        "events from exactly this cell layout, from when `pipeline/evaluate_heldout.py` ran it for real). "
        "`rules.policy.can_share()` decides, and logs its decision, every time it's called."
    )

    rule_state = current_state(audit_path, EXAMPLE_RULE_ID)
    if rule_state != State.APPROVED:
        st.info(f"Rule is **{rule_state.value if rule_state else 'None'}**, not APPROVED -- `policy.can_share()` will deny both A->B and A->C ('rule not approved') until it's approved in section 4.")

    base_params = dict(check1_mod.BASELINE_PARAMS)  # the real nominal setting, not an arbitrary illustrative value
    cell_a = CellState("A", "owner1", base_params)
    cell_b = CellState("B", "owner1", base_params)
    cell_c = CellState("C", "owner2", base_params)

    st.markdown(
        "`rules/policy.py` has no separate \"receiving owner grants permission\" control today: "
        "`REQUIRE_RECEIVER_OPT_IN` defaults to `False` (a placeholder, see `rules/DECISIONS.md`), so the "
        "only lever that matters is the **source** owner's (`owner1`, Cell A/B's owner) `sharing_enabled` "
        "flag. Toggle it below to see the A->C decision change -- there is nothing for Cell C's owner to "
        "grant."
    )
    owner1_sharing = st.checkbox("owner1.sharing_enabled (source owner allows rules to cross to other owners)", value=False)

    owners = {
        "owner1": policy.Owner("owner1", sharing_enabled=owner1_sharing),
        "owner2": policy.Owner("owner2", sharing_enabled=False),
    }

    col_ab, col_ac = st.columns(2)
    with col_ab:
        if st.button("Evaluate A -> B (same owner)"):
            st.session_state["decision_b"] = policy.can_share(audit_path, example_rule, cell_a, cell_b, owners)
        if "decision_b" in st.session_state:
            d = st.session_state["decision_b"]
            (st.success if d.allowed else st.error)(f"A -> B: allowed={d.allowed} ({d.reason})")
    with col_ac:
        if st.button("Evaluate A -> C (different owner)"):
            st.session_state["decision_c"] = policy.can_share(audit_path, example_rule, cell_a, cell_c, owners)
        if "decision_c" in st.session_state:
            d = st.session_state["decision_c"]
            (st.success if d.allowed else st.error)(f"A -> C: allowed={d.allowed} ({d.reason})")
else:
    st.info(f"The real audit trail for `{EXAMPLE_RULE_ID}` is not available in demo_data -- see section 4.")

st.divider()

# ============================================================================
# 6. Audit log
# ============================================================================
st.header("6. Audit log")

if "demo_audit_path" in st.session_state:
    audit_path = st.session_state["demo_audit_path"]
    st.markdown(
        f"This session's working copy starts from the **real** recorded history of `{EXAMPLE_RULE_ID}` "
        "(PROPOSED, CHECK1_PASSED, APPROVED, SHARED/BLOCKED), plus whatever you triggered in sections 4-5 "
        "above -- every line was written by the real `rules/audit.append_event()`, hash-chained to the "
        "line before it."
    )

    events = audit.read_events(audit_path)
    st.dataframe(
        [{"ts": e.ts, "event": e.event, "rule_id": e.rule_id, "actor": e.actor, "details": json.dumps(e.details)} for e in events],
        width="stretch",
        hide_index=True,
    )

    if st.button("Verify chain"):
        result = audit.verify_chain(audit_path)
        if result is True:
            st.success("audit.verify_chain(): True -- the whole log is intact.")
        else:
            st.error(f"audit.verify_chain(): broken at line {result}.")

    with st.expander("What a tampered copy looks like"):
        st.markdown("A second, in-memory copy of the same log with one event's `details` edited **after the fact** -- the hash was never recomputed, so the chain breaks exactly there.")
        lines = Path(audit_path).read_text().splitlines() if Path(audit_path).exists() else []
        if len(lines) >= 1:
            tampered_lines = list(lines)
            tampered_record = json.loads(tampered_lines[0])
            tampered_record["details"] = {**tampered_record["details"], "tampered": True}
            tampered_lines[0] = json.dumps(tampered_record)
            fd, tampered_path = tempfile.mkstemp(prefix="fleet_demo_audit_tampered_", suffix=".jsonl")
            with os.fdopen(fd, "w") as f:
                f.write("\n".join(tampered_lines) + "\n")
            tampered_result = audit.verify_chain(tampered_path)
            os.remove(tampered_path)
            st.code("\n".join(tampered_lines[:3]), language="json")
            st.error(f"audit.verify_chain() on the tampered copy: broken at line {tampered_result}")
        else:
            st.info("No events logged yet.")
else:
    st.info(f"The real audit trail for `{EXAMPLE_RULE_ID}` is not available in demo_data -- see section 4.")

st.divider()

# ============================================================================
# 7. Registered results
# ============================================================================
st.header("7. Registered results (final held-out evaluation)")
st.caption(
    "This section's template (below) was registered under the ORIGINAL (revision-1-only) protocol, "
    "which never produced a gate-passing rule to evaluate (section 2). Revision 2's real held-out "
    "result -- which did run, on seeds 3000-3199 -- is shown in section 2's \"Held-out result\" panel, "
    "not here; this template is kept as-is rather than silently repurposed."
)
results_path = DEMO_DATA / "registered_results.json"
if results_path.exists():
    registered = json.loads(results_path.read_text())
    all_null = registered.get("evaluation_ts") is None and registered.get("arms", {}).get("A", {}).get("rules") is None
    if all_null:
        st.info(
            "**Results pending under this original template.** Per `docs/FINAL_PROTOCOL.md` / "
            "`docs/SUCCESS_CRITERION.md`, that evaluation runs once, after every revised loop run across "
            "both arms is done. Revision 1 (both arms) ended 0/10 gate-passing rules, so under the "
            "original protocol there was nothing to evaluate -- `demo_data/registered_results.json` is "
            "still the null template shown below. (Revision 2 is a separate, post hoc follow-up -- see "
            "section 2.)"
        )
    else:
        st.subheader("Headline")
        st.json(registered.get("headline"))
        st.subheader("Per-arm summary")
        st.json(registered.get("arms"))
        st.subheader("Feasibility table (baselines)")
        st.json(registered.get("feasibility_table"))
    with st.expander("Full file (template, with field meanings)"):
        st.json(registered)
    st.caption(f"source: demo_data/{results_path.name}")
else:
    st.info("results pending (demo_data/registered_results.json not found)")

st.divider()
st.caption("BOLT by Pink Panthers -- simulation only")
