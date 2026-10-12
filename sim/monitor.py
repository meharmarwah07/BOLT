"""The Monitor: plain rule-based code that turns a failed TrialResult into
a FailureReport (CONTEXT.md section 4 step 2 / section 7). Not an LLM.

True friction is never included -- the Solver must infer it from
slip_mm_before_drop, same as a real monitor would (CONTEXT.md section 7).

NOTE for whoever owns Part B: checked llm/solver.py -- it currently only
reads report_id off a FailureReport, not outcome or drop_phase, and every
existing fixture/example only ever uses outcome="drop". This Monitor also
emits outcome="crush" for over-grip failures (a real failure Part A can
detect), but nothing downstream is built to actually make use of that yet.
Flag before relying on crush reports reaching the Solver meaningfully --
per CONTEXT.md rule 3, this isn't a unilateral contract change (outcome
has no enum in the schema), just a heads-up that it's new.
"""
import uuid


def make_failure_report(trial_result, cell_id, owner_id):
    """Returns None for a successful trial -- only failures get reported,
    matching CONTEXT.md section 4 step 2 ("when one cell fails... monitor
    writes a structured failure report")."""
    if trial_result.outcome == "ok":
        return None
    return {
        "report_id": f"fr_{uuid.uuid4().hex[:8]}",
        "cell_id": cell_id,
        "owner_id": owner_id,
        "object_class": trial_result.object_class,
        "controller": dict(trial_result.controller),
        "outcome": trial_result.outcome,
        "drop_phase": trial_result.drop_phase,
        "slip_mm_before_drop": trial_result.slip_mm_before_drop,
        "est_mass_kg": trial_result.est_mass_kg,
        "seed": trial_result.seed,
    }


if __name__ == "__main__":
    import argparse
    import json

    from cell_model import OBJECT_CLASSES
    from run_trial import run_trial

    p = argparse.ArgumentParser()
    p.add_argument("--object-class", choices=list(OBJECT_CLASSES), default="glass_jar")
    p.add_argument("--grip-force", type=float, default=2.0)
    p.add_argument("--carry-accel", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cell-id", default="A")
    p.add_argument("--owner-id", default="owner1")
    args = p.parse_args()

    result = run_trial(
        args.object_class,
        {"grip_force_n": args.grip_force, "carry_accel_ms2": args.carry_accel},
        args.seed,
    )
    report = make_failure_report(result, args.cell_id, args.owner_id)
    print("TRIAL outcome:", result.outcome)
    print("FAILURE REPORT:" if report else "No report -- trial succeeded.")
    if report:
        print(json.dumps(report, indent=2))
