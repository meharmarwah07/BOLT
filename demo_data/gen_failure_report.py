"""One-off script: generate the real Cell A FailureReport for the demo.

Run once, by hand, from the repo root:
    PYTHONPATH=. venv/bin/python3 demo_data/gen_failure_report.py

Uses sim/run_trial.py + sim/monitor.py directly (never hand-typed), at the
nominal setting (pipeline/check1.py's BASELINE_PARAMS), train seed 0,
object_class glass_jar, cell A / owner1 -- matching pipeline/loop.py's
real_failure_reports(). Writes demo_data/failure_report_seed0.json.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(ROOT, "sim")
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from monitor import make_failure_report  # sim/monitor.py
from run_trial import run_trial  # sim/run_trial.py

NOMINAL_PARAMS = {"grip_force_n": 5.0, "carry_accel_ms2": 1.0}  # pipeline/check1.py BASELINE_PARAMS

if __name__ == "__main__":
    result = run_trial("glass_jar", NOMINAL_PARAMS, seed=0)
    report = make_failure_report(result, cell_id="A", owner_id="owner1")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "failure_report_seed0.json")
    if report is None:
        raise SystemExit(f"seed 0 at nominal setting did not fail (outcome={result.outcome}) -- nothing to report")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"outcome={result.outcome} drop_phase={result.drop_phase} slip_mm={result.slip_mm_before_drop}")
    print(f"wrote {out_path}")
