"""Smoke-test solver.solve() against a real (or stub) OpenAI-compatible model.

Loads synthetic FailureReports (fixtures/synthetic_reports.json), calls
solver.solve() once per report (optionally repeated several times per
report via --repeats), and scores each call against hidden ground truth
(fixtures/truth.json). That truth file is NEVER sent to the model -- it is
read only here, after the fact, to judge the reply.

IMPORTANT: every "effective" / "correct refusal" verdict this script
prints or writes is an ANALYTIC stand-in, computed from the formula
N >= m * (g + a) / (2 * mu). It is NOT a MuJoCo simulator replay and is
not evidence about the real physics -- it only checks internal consistency
between the model's proposed numbers and the hidden mu used to construct
each fixture.

Usage:
    cd ~/fleet/llm && env -u PYTHONPATH PYTHONPATH=$HOME/fleet python smoke_real.py [--repeats N]

Against the local stub (no GPU, no cost -- plumbing check only):
    # terminal 1
    python fake_server.py
    # terminal 2
    env -u PYTHONPATH PYTHONPATH=$HOME/fleet python smoke_real.py

Against a real model (costs GPU time -- run only after a clean stub run):
    FLEET_LLM_BASE_URL=http://<host>:<port>/v1 FLEET_LLM_MODEL=<model-name> \\
        env -u PYTHONPATH PYTHONPATH=$HOME/fleet python smoke_real.py
"""
import argparse
import json
import os
import time

import solver
from rule_check import NoSafeRule, Rule
from stats import wilson

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(HERE, "fixtures")
RUNS_DIR = os.path.join(HERE, "runs")

G = 9.81


def load_fixtures():
    with open(os.path.join(FIXTURES_DIR, "synthetic_reports.json"), "r", encoding="utf-8") as f:
        reports = json.load(f)["reports"]
    with open(os.path.join(FIXTURES_DIR, "truth.json"), "r", encoding="utf-8") as f:
        truth = json.load(f)["cases"]
    return reports, truth


def classify_outcome(rule_or_none, error_text):
    if error_text is not None:
        return "error"
    if isinstance(rule_or_none, Rule):
        return "valid_rule"
    if isinstance(rule_or_none, NoSafeRule):
        return "no_safe_rule"
    raise ValueError(f"unexpected solve() result type: {type(rule_or_none)!r}")


def is_rule_effective(rule, report, truth_case, g=G):
    """ANALYTIC stand-in, not a simulator result: does the proposed grip
    force meet the no-slip requirement given the fixture's hidden mu?"""
    required_n = report["est_mass_kg"] * (g + rule.carry_accel_max_ms2) / (2 * truth_case["hidden_mu"])
    return rule.grip_force_min_n >= required_n


def score_call(report, truth_case, rule_or_none, error_text):
    """Score one solve() outcome against hidden ground truth.

    Returns {"outcome": "valid_rule"|"no_safe_rule"|"error",
             "effective": bool|None, "refusal_correct": bool|None}.
    """
    outcome = classify_outcome(rule_or_none, error_text)
    verdict = {"outcome": outcome, "effective": None, "refusal_correct": None}
    if outcome == "valid_rule":
        verdict["effective"] = is_rule_effective(rule_or_none, report, truth_case)
    elif outcome == "no_safe_rule":
        verdict["refusal_correct"] = not truth_case["feasible"]
    return verdict


def run_once(report):
    start = time.perf_counter()
    rule_or_none, error_text = None, None
    try:
        rule_or_none = solver.solve([report])
    except Exception as e:
        error_text = str(e)
    latency_s = time.perf_counter() - start
    return rule_or_none, error_text, latency_s


def rate_with_ci(k, n):
    if n == 0:
        return None
    low, high = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson_low": low, "wilson_high": high}


def aggregate(records):
    feasible = [r for r in records if r["feasible"]]
    infeasible = [r for r in records if not r["feasible"]]
    valid_json_k = sum(1 for r in records if r["outcome"] != "error")
    effective_k = sum(1 for r in feasible if r["outcome"] == "valid_rule" and r["effective"])
    correct_refusal_k = sum(1 for r in infeasible if r["outcome"] == "no_safe_rule")
    wrong_refusal_k = sum(1 for r in feasible if r["outcome"] == "no_safe_rule")
    return {
        "valid_json_rate": rate_with_ci(valid_json_k, len(records)),
        "effective_rule_rate_among_feasible": rate_with_ci(effective_k, len(feasible)),
        "correct_refusal_rate_among_infeasible": rate_with_ci(correct_refusal_k, len(infeasible)),
        "wrong_refusal_rate_among_feasible": rate_with_ci(wrong_refusal_k, len(feasible)),
    }


def fmt_rate(label, stat):
    if stat is None:
        return f"  {label}: n/a (n=0)"
    return (f"  {label}: {stat['k']}/{stat['n']} = {stat['rate']:.3f} "
            f"(95% Wilson [{stat['wilson_low']:.3f}, {stat['wilson_high']:.3f}])")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=1, help="sample each fixture this many times (default 1)")
    args = parser.parse_args()

    reports, truth = load_fixtures()

    print(f"FLEET_LLM_BASE_URL={solver.BASE_URL}  FLEET_LLM_MODEL={solver.MODEL}  repeats={args.repeats}")
    print("NOTE: effectiveness/refusal verdicts below are an ANALYTIC stand-in")
    print("(N >= m*(g+a)/(2*mu)), not a MuJoCo simulator replay.\n")

    records = []
    for report in reports:
        truth_case = truth[report["report_id"]]
        for repeat_index in range(args.repeats):
            rule_or_none, error_text, latency_s = run_once(report)
            verdict = score_call(report, truth_case, rule_or_none, error_text)
            if isinstance(rule_or_none, (Rule, NoSafeRule)):
                rule_field = rule_or_none.model_dump(mode="json")
            else:
                rule_field = None
            record = {
                "report_id": report["report_id"],
                "object_class": report["object_class"],
                "repeat_index": repeat_index,
                "outcome": verdict["outcome"],
                "effective": verdict["effective"],
                "refusal_correct": verdict["refusal_correct"],
                "feasible": truth_case["feasible"],
                "latency_s": latency_s,
                "rule": rule_field,
                "error": error_text,
            }
            records.append(record)
            print(f"{record['report_id']:>20s} #{repeat_index}  {record['outcome']:12s}  "
                  f"{latency_s:6.2f}s  effective={record['effective']}  refusal_correct={record['refusal_correct']}")

    print()
    for object_class in sorted({r["object_class"] for r in records}):
        class_records = [r for r in records if r["object_class"] == object_class]
        print(f"[{object_class}] n={len(class_records)}")
        for label, stat in aggregate(class_records).items():
            print(fmt_rate(label, stat))
        print()

    print("[overall] (ANALYTIC stand-in, not simulator output)")
    for label, stat in aggregate(records).items():
        print(fmt_rate(label, stat))

    os.makedirs(RUNS_DIR, exist_ok=True)
    out_path = os.path.join(RUNS_DIR, f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "base_url": solver.BASE_URL,
            "model": solver.MODEL,
            "repeats": args.repeats,
            "note": "effective/refusal_correct are an ANALYTIC stand-in (N >= m*(g+a)/(2*mu)), not simulator output",
            "records": records,
        }, f, indent=2)
    print(f"\nwrote {len(records)} per-call results to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
