# Revision 2: final held-out result

Evaluated exactly as registered in docs/REVISION2_FINAL_SET.md (commit 6d2e208): one invocation, k = 10 rules, Cell B seeds 3000-3099, Cell C seeds 3100-3199, --check1-seeds 300, threshold 0.05/10 = 0.005. Seeds 3000-3199 are now spent. No change to rules, prompts, gate or selection after the first look.

Status: revision 2 is post hoc. The registered primary result of revision 1 stays 0/10 (docs/REVISED_RUNS_RESULT.md).

## Outcome (criterion D, all 10 rules PASS; per-arm A 5/5, B 5/5)

| policy (grip N / accel) | rules | B drop | C drop (sharing allowed) | crushes | B McNemar p |
|---|---|---|---|---|---|
| 20.0 / 1.0 | A1 A3 A4 A5 | 0.25 | 0.30 | 0 | 5.294e-23 |
| 20.0 / 0.5 | A2 B1 B3 B5 | 0.24 | 0.27 | 0 | 2.647e-23 |
| 21.0 / 0.5 | B2 B4 | 0.18 | 0.24 | 0 | 4.136e-25 |

Baseline (no rule) drop rate 1.000 in both cells. Identical policies give identical results; there are 3 distinct findings, not 10 independent successes. Differences among the three policies are within sampling noise (95% CI about +/-9 points at n=100) and no ranking is claimed.

Cell C with sharing denied (the realistic default, owner's flag off) is unchanged from the baseline; the verdicts above use the sharing-allowed counterfactual per criterion D(ii).

## Not shown

Residual failure is 18-30%. One object class, simulation only. The naive 0.9x-crush heuristic is statistically comparable on task success for the 20.0/0.5 policy in Cell C (p=0.27) but crushed the jar in 4 of 100 runs; the rules crushed 0. The grid-search baseline found no passing point (50 train seeds, 6x6) and is not a comparison.

## Provenance (sha256)

- report .json: 46fbad65d652b1d336419b1278ae1490a124533d2b6304db9c71afb0112cc5c0
- report .md: 33ace0a9188054edc501e9d3113bf7ea414d839eff7bac2e38cf9840a9f92c91
- heldout_access.log: 1e0bbfeb74f9d6dc48fce8809ca2e0824935529a6a05251067d3f8fbc949f76a
- final_r2.log: b2a09ace60d0ef3e14f837507e9f3021e0e2ea82602ba97a7205c1cfebdad425
- rule files: docs/revision2_rules/ (hashes in docs/REVISION2_FINAL_SET.md)
