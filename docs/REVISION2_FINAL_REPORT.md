# Held-out evaluation (multi-rule, k=10)

Multi-rule provenance re-check (--check1-seeds): train seeds 0-299 (300 total).

NOTICE: this is a non-interactive demo script. Check 2 (human approval) is AUTO-APPROVED here for the purpose of this held-out evaluation -- a real deployment requires an actual human reviewer (rules/review_cli.py), not this script. The baseline ('none' / 'without the rule') condition in every table below is the nominal setting {'grip_force_n': 5.0, 'carry_accel_ms2': 1.0} (grip_force_n, carry_accel_ms2) -- see pipeline/check1.py's BASELINE_PARAMS for how and why it was chosen. docs/SUCCESS_CRITERION.md item D(i): Bonferroni-adjusted significance threshold 0.05/10 = 0.005 (k = 10 rules evaluated in this invocation). Cell C verdicts below use the 'C_sharing_allowed' counterfactual (would this rule hurt/help Cell C if shared), not the realistic default decision, which is reported separately.

| rule_id | class | B drop (none) | B drop (rule) | B McNemar p | Bonferroni thr. | C drop (none) | C drop (rule) | B crush none->rule | C crush none->rule | verdict D |
|---|---|---|---|---|---|---|---|---|---|---|
| r_a8e42ed5_2 | glass_jar | 1.000 | 0.250 | 5.294e-23 | 0.005 | 1.000 | 0.300 | 0->0 | 0->0 | PASS |
| r_6d83263f_2 | glass_jar | 1.000 | 0.240 | 2.647e-23 | 0.005 | 1.000 | 0.270 | 0->0 | 0->0 | PASS |
| r_abcfcb78_1 | glass_jar | 1.000 | 0.250 | 5.294e-23 | 0.005 | 1.000 | 0.300 | 0->0 | 0->0 | PASS |
| r_2d8a7a90_4 | glass_jar | 1.000 | 0.250 | 5.294e-23 | 0.005 | 1.000 | 0.300 | 0->0 | 0->0 | PASS |
| r_cd264c2a_1 | glass_jar | 1.000 | 0.250 | 5.294e-23 | 0.005 | 1.000 | 0.300 | 0->0 | 0->0 | PASS |
| r_9559550c_7 | glass_jar | 1.000 | 0.240 | 2.647e-23 | 0.005 | 1.000 | 0.270 | 0->0 | 0->0 | PASS |
| r_0838945e_7 | glass_jar | 1.000 | 0.180 | 4.136e-25 | 0.005 | 1.000 | 0.240 | 0->0 | 0->0 | PASS |
| r_ba19eb12_9 | glass_jar | 1.000 | 0.240 | 2.647e-23 | 0.005 | 1.000 | 0.270 | 0->0 | 0->0 | PASS |
| r_6b1ef4fa_7 | glass_jar | 1.000 | 0.180 | 4.136e-25 | 0.005 | 1.000 | 0.240 | 0->0 | 0->0 | PASS |
| r_aa2d65bc_9 | glass_jar | 1.000 | 0.240 | 2.647e-23 | 0.005 | 1.000 | 0.270 | 0->0 | 0->0 | PASS |

## Rule r_a8e42ed5_2 (glass_jar)

Cell B can_share: allowed=True (same owner)

### Cell B

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 17 | 2 | 0.170 | [0.109, 0.255] | 0.810 | 1.405 | 81 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 45 | 0.010 | [0.002, 0.054] | 0.540 | 1.405 | 54 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 1.0 | 100 | 25 | 0 | 0.250 | [0.175, 0.343] | 0.750 | 0.994 | 75 | 0 |

### Cell B: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 75 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 10 | 0.1796 |
| ceiling_near_crush_min_accel | 34 | 13 | 0.0031 |
| grid_mildest_passing | 75 | 0 | 0.0000 |

Cell C_sharing_denied can_share: allowed=False (different owner and source owner's sharing flag is off)

### Cell C_sharing_denied

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |

### Cell C_sharing_denied: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 0 | 0 | 1.0000 |
| naive_0.9x_crush_min_accel | 0 | 78 | 0.0000 |
| ceiling_near_crush_min_accel | 0 | 49 | 0.0000 |
| grid_mildest_passing | 0 | 0 | 1.0000 |

Cell C_sharing_allowed can_share: allowed=True (different owner, source owner's sharing flag is on)

### Cell C_sharing_allowed

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 1.0 | 100 | 30 | 0 | 0.300 | [0.219, 0.396] | 0.700 | 0.994 | 70 | 0 |

### Cell C_sharing_allowed: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 70 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 12 | 0.0768 |
| ceiling_near_crush_min_accel | 31 | 10 | 0.0015 |
| grid_mildest_passing | 70 | 0 | 0.0000 |

criterion D: (i) Cell B drop rate lower & p<0.005: True (p=5.294e-23); (ii) Cell C drop rate not higher: True; (iii) crush not increased: True => **PASS**

## Rule r_6d83263f_2 (glass_jar)

Cell B can_share: allowed=True (same owner)

### Cell B

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 17 | 2 | 0.170 | [0.109, 0.255] | 0.810 | 1.405 | 81 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 45 | 0.010 | [0.002, 0.054] | 0.540 | 1.405 | 54 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 0.5 | 100 | 24 | 0 | 0.240 | [0.167, 0.332] | 0.760 | 1.405 | 76 | 0 |

### Cell B: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 76 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 9 | 0.2668 |
| ceiling_near_crush_min_accel | 34 | 12 | 0.0016 |
| grid_mildest_passing | 76 | 0 | 0.0000 |

Cell C_sharing_denied can_share: allowed=False (different owner and source owner's sharing flag is off)

### Cell C_sharing_denied

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |

### Cell C_sharing_denied: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 0 | 0 | 1.0000 |
| naive_0.9x_crush_min_accel | 0 | 78 | 0.0000 |
| ceiling_near_crush_min_accel | 0 | 49 | 0.0000 |
| grid_mildest_passing | 0 | 0 | 1.0000 |

Cell C_sharing_allowed can_share: allowed=True (different owner, source owner's sharing flag is on)

### Cell C_sharing_allowed

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 0.5 | 100 | 27 | 0 | 0.270 | [0.193, 0.364] | 0.730 | 1.405 | 73 | 0 |

### Cell C_sharing_allowed: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 73 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 9 | 0.2668 |
| ceiling_near_crush_min_accel | 34 | 10 | 0.0004 |
| grid_mildest_passing | 73 | 0 | 0.0000 |

criterion D: (i) Cell B drop rate lower & p<0.005: True (p=2.647e-23); (ii) Cell C drop rate not higher: True; (iii) crush not increased: True => **PASS**

## Rule r_abcfcb78_1 (glass_jar)

Cell B can_share: allowed=True (same owner)

### Cell B

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 17 | 2 | 0.170 | [0.109, 0.255] | 0.810 | 1.405 | 81 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 45 | 0.010 | [0.002, 0.054] | 0.540 | 1.405 | 54 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 1.0 | 100 | 25 | 0 | 0.250 | [0.175, 0.343] | 0.750 | 0.994 | 75 | 0 |

### Cell B: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 75 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 10 | 0.1796 |
| ceiling_near_crush_min_accel | 34 | 13 | 0.0031 |
| grid_mildest_passing | 75 | 0 | 0.0000 |

Cell C_sharing_denied can_share: allowed=False (different owner and source owner's sharing flag is off)

### Cell C_sharing_denied

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |

### Cell C_sharing_denied: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 0 | 0 | 1.0000 |
| naive_0.9x_crush_min_accel | 0 | 78 | 0.0000 |
| ceiling_near_crush_min_accel | 0 | 49 | 0.0000 |
| grid_mildest_passing | 0 | 0 | 1.0000 |

Cell C_sharing_allowed can_share: allowed=True (different owner, source owner's sharing flag is on)

### Cell C_sharing_allowed

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 1.0 | 100 | 30 | 0 | 0.300 | [0.219, 0.396] | 0.700 | 0.994 | 70 | 0 |

### Cell C_sharing_allowed: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 70 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 12 | 0.0768 |
| ceiling_near_crush_min_accel | 31 | 10 | 0.0015 |
| grid_mildest_passing | 70 | 0 | 0.0000 |

criterion D: (i) Cell B drop rate lower & p<0.005: True (p=5.294e-23); (ii) Cell C drop rate not higher: True; (iii) crush not increased: True => **PASS**

## Rule r_2d8a7a90_4 (glass_jar)

Cell B can_share: allowed=True (same owner)

### Cell B

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 17 | 2 | 0.170 | [0.109, 0.255] | 0.810 | 1.405 | 81 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 45 | 0.010 | [0.002, 0.054] | 0.540 | 1.405 | 54 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 1.0 | 100 | 25 | 0 | 0.250 | [0.175, 0.343] | 0.750 | 0.994 | 75 | 0 |

### Cell B: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 75 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 10 | 0.1796 |
| ceiling_near_crush_min_accel | 34 | 13 | 0.0031 |
| grid_mildest_passing | 75 | 0 | 0.0000 |

Cell C_sharing_denied can_share: allowed=False (different owner and source owner's sharing flag is off)

### Cell C_sharing_denied

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |

### Cell C_sharing_denied: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 0 | 0 | 1.0000 |
| naive_0.9x_crush_min_accel | 0 | 78 | 0.0000 |
| ceiling_near_crush_min_accel | 0 | 49 | 0.0000 |
| grid_mildest_passing | 0 | 0 | 1.0000 |

Cell C_sharing_allowed can_share: allowed=True (different owner, source owner's sharing flag is on)

### Cell C_sharing_allowed

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 1.0 | 100 | 30 | 0 | 0.300 | [0.219, 0.396] | 0.700 | 0.994 | 70 | 0 |

### Cell C_sharing_allowed: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 70 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 12 | 0.0768 |
| ceiling_near_crush_min_accel | 31 | 10 | 0.0015 |
| grid_mildest_passing | 70 | 0 | 0.0000 |

criterion D: (i) Cell B drop rate lower & p<0.005: True (p=5.294e-23); (ii) Cell C drop rate not higher: True; (iii) crush not increased: True => **PASS**

## Rule r_cd264c2a_1 (glass_jar)

Cell B can_share: allowed=True (same owner)

### Cell B

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 17 | 2 | 0.170 | [0.109, 0.255] | 0.810 | 1.405 | 81 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 45 | 0.010 | [0.002, 0.054] | 0.540 | 1.405 | 54 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 1.0 | 100 | 25 | 0 | 0.250 | [0.175, 0.343] | 0.750 | 0.994 | 75 | 0 |

### Cell B: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 75 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 10 | 0.1796 |
| ceiling_near_crush_min_accel | 34 | 13 | 0.0031 |
| grid_mildest_passing | 75 | 0 | 0.0000 |

Cell C_sharing_denied can_share: allowed=False (different owner and source owner's sharing flag is off)

### Cell C_sharing_denied

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |

### Cell C_sharing_denied: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 0 | 0 | 1.0000 |
| naive_0.9x_crush_min_accel | 0 | 78 | 0.0000 |
| ceiling_near_crush_min_accel | 0 | 49 | 0.0000 |
| grid_mildest_passing | 0 | 0 | 1.0000 |

Cell C_sharing_allowed can_share: allowed=True (different owner, source owner's sharing flag is on)

### Cell C_sharing_allowed

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 1.0 | 100 | 30 | 0 | 0.300 | [0.219, 0.396] | 0.700 | 0.994 | 70 | 0 |

### Cell C_sharing_allowed: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 70 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 12 | 0.0768 |
| ceiling_near_crush_min_accel | 31 | 10 | 0.0015 |
| grid_mildest_passing | 70 | 0 | 0.0000 |

criterion D: (i) Cell B drop rate lower & p<0.005: True (p=5.294e-23); (ii) Cell C drop rate not higher: True; (iii) crush not increased: True => **PASS**

## Rule r_9559550c_7 (glass_jar)

Cell B can_share: allowed=True (same owner)

### Cell B

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 17 | 2 | 0.170 | [0.109, 0.255] | 0.810 | 1.405 | 81 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 45 | 0.010 | [0.002, 0.054] | 0.540 | 1.405 | 54 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 0.5 | 100 | 24 | 0 | 0.240 | [0.167, 0.332] | 0.760 | 1.405 | 76 | 0 |

### Cell B: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 76 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 9 | 0.2668 |
| ceiling_near_crush_min_accel | 34 | 12 | 0.0016 |
| grid_mildest_passing | 76 | 0 | 0.0000 |

Cell C_sharing_denied can_share: allowed=False (different owner and source owner's sharing flag is off)

### Cell C_sharing_denied

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |

### Cell C_sharing_denied: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 0 | 0 | 1.0000 |
| naive_0.9x_crush_min_accel | 0 | 78 | 0.0000 |
| ceiling_near_crush_min_accel | 0 | 49 | 0.0000 |
| grid_mildest_passing | 0 | 0 | 1.0000 |

Cell C_sharing_allowed can_share: allowed=True (different owner, source owner's sharing flag is on)

### Cell C_sharing_allowed

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 0.5 | 100 | 27 | 0 | 0.270 | [0.193, 0.364] | 0.730 | 1.405 | 73 | 0 |

### Cell C_sharing_allowed: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 73 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 9 | 0.2668 |
| ceiling_near_crush_min_accel | 34 | 10 | 0.0004 |
| grid_mildest_passing | 73 | 0 | 0.0000 |

criterion D: (i) Cell B drop rate lower & p<0.005: True (p=2.647e-23); (ii) Cell C drop rate not higher: True; (iii) crush not increased: True => **PASS**

## Rule r_0838945e_7 (glass_jar)

Cell B can_share: allowed=True (same owner)

### Cell B

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 17 | 2 | 0.170 | [0.109, 0.255] | 0.810 | 1.405 | 81 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 45 | 0.010 | [0.002, 0.054] | 0.540 | 1.405 | 54 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 21.0 | 0.5 | 100 | 18 | 0 | 0.180 | [0.117, 0.267] | 0.820 | 1.405 | 82 | 0 |

### Cell B: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 82 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 7 | 6 | 1.0000 |
| ceiling_near_crush_min_accel | 33 | 5 | 0.0000 |
| grid_mildest_passing | 82 | 0 | 0.0000 |

Cell C_sharing_denied can_share: allowed=False (different owner and source owner's sharing flag is off)

### Cell C_sharing_denied

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |

### Cell C_sharing_denied: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 0 | 0 | 1.0000 |
| naive_0.9x_crush_min_accel | 0 | 78 | 0.0000 |
| ceiling_near_crush_min_accel | 0 | 49 | 0.0000 |
| grid_mildest_passing | 0 | 0 | 1.0000 |

Cell C_sharing_allowed can_share: allowed=True (different owner, source owner's sharing flag is on)

### Cell C_sharing_allowed

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 21.0 | 0.5 | 100 | 24 | 0 | 0.240 | [0.167, 0.332] | 0.760 | 1.405 | 76 | 0 |

### Cell C_sharing_allowed: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 76 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 7 | 9 | 0.8036 |
| ceiling_near_crush_min_accel | 37 | 10 | 0.0001 |
| grid_mildest_passing | 76 | 0 | 0.0000 |

criterion D: (i) Cell B drop rate lower & p<0.005: True (p=4.136e-25); (ii) Cell C drop rate not higher: True; (iii) crush not increased: True => **PASS**

## Rule r_ba19eb12_9 (glass_jar)

Cell B can_share: allowed=True (same owner)

### Cell B

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 17 | 2 | 0.170 | [0.109, 0.255] | 0.810 | 1.405 | 81 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 45 | 0.010 | [0.002, 0.054] | 0.540 | 1.405 | 54 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 0.5 | 100 | 24 | 0 | 0.240 | [0.167, 0.332] | 0.760 | 1.405 | 76 | 0 |

### Cell B: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 76 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 9 | 0.2668 |
| ceiling_near_crush_min_accel | 34 | 12 | 0.0016 |
| grid_mildest_passing | 76 | 0 | 0.0000 |

Cell C_sharing_denied can_share: allowed=False (different owner and source owner's sharing flag is off)

### Cell C_sharing_denied

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |

### Cell C_sharing_denied: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 0 | 0 | 1.0000 |
| naive_0.9x_crush_min_accel | 0 | 78 | 0.0000 |
| ceiling_near_crush_min_accel | 0 | 49 | 0.0000 |
| grid_mildest_passing | 0 | 0 | 1.0000 |

Cell C_sharing_allowed can_share: allowed=True (different owner, source owner's sharing flag is on)

### Cell C_sharing_allowed

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 0.5 | 100 | 27 | 0 | 0.270 | [0.193, 0.364] | 0.730 | 1.405 | 73 | 0 |

### Cell C_sharing_allowed: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 73 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 9 | 0.2668 |
| ceiling_near_crush_min_accel | 34 | 10 | 0.0004 |
| grid_mildest_passing | 73 | 0 | 0.0000 |

criterion D: (i) Cell B drop rate lower & p<0.005: True (p=2.647e-23); (ii) Cell C drop rate not higher: True; (iii) crush not increased: True => **PASS**

## Rule r_6b1ef4fa_7 (glass_jar)

Cell B can_share: allowed=True (same owner)

### Cell B

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 17 | 2 | 0.170 | [0.109, 0.255] | 0.810 | 1.405 | 81 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 45 | 0.010 | [0.002, 0.054] | 0.540 | 1.405 | 54 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 21.0 | 0.5 | 100 | 18 | 0 | 0.180 | [0.117, 0.267] | 0.820 | 1.405 | 82 | 0 |

### Cell B: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 82 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 7 | 6 | 1.0000 |
| ceiling_near_crush_min_accel | 33 | 5 | 0.0000 |
| grid_mildest_passing | 82 | 0 | 0.0000 |

Cell C_sharing_denied can_share: allowed=False (different owner and source owner's sharing flag is off)

### Cell C_sharing_denied

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |

### Cell C_sharing_denied: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 0 | 0 | 1.0000 |
| naive_0.9x_crush_min_accel | 0 | 78 | 0.0000 |
| ceiling_near_crush_min_accel | 0 | 49 | 0.0000 |
| grid_mildest_passing | 0 | 0 | 1.0000 |

Cell C_sharing_allowed can_share: allowed=True (different owner, source owner's sharing flag is on)

### Cell C_sharing_allowed

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 21.0 | 0.5 | 100 | 24 | 0 | 0.240 | [0.167, 0.332] | 0.760 | 1.405 | 76 | 0 |

### Cell C_sharing_allowed: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 76 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 7 | 9 | 0.8036 |
| ceiling_near_crush_min_accel | 37 | 10 | 0.0001 |
| grid_mildest_passing | 76 | 0 | 0.0000 |

criterion D: (i) Cell B drop rate lower & p<0.005: True (p=4.136e-25); (ii) Cell C drop rate not higher: True; (iii) crush not increased: True => **PASS**

## Rule r_aa2d65bc_9 (glass_jar)

Cell B can_share: allowed=True (same owner)

### Cell B

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 17 | 2 | 0.170 | [0.109, 0.255] | 0.810 | 1.405 | 81 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 45 | 0.010 | [0.002, 0.054] | 0.540 | 1.405 | 54 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 0.5 | 100 | 24 | 0 | 0.240 | [0.167, 0.332] | 0.760 | 1.405 | 76 | 0 |

### Cell B: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 76 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 9 | 0.2668 |
| ceiling_near_crush_min_accel | 34 | 12 | 0.0016 |
| grid_mildest_passing | 76 | 0 | 0.0000 |

Cell C_sharing_denied can_share: allowed=False (different owner and source owner's sharing flag is off)

### Cell C_sharing_denied

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |

### Cell C_sharing_denied: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 0 | 0 | 1.0000 |
| naive_0.9x_crush_min_accel | 0 | 78 | 0.0000 |
| ceiling_near_crush_min_accel | 0 | 49 | 0.0000 |
| grid_mildest_passing | 0 | 0 | 1.0000 |

Cell C_sharing_allowed can_share: allowed=True (different owner, source owner's sharing flag is on)

### Cell C_sharing_allowed

| condition | grip_force_n | carry_accel_ms2 | n | drops | crushes | drop rate | 95% CI | task success | mean duration (s) | fixed | newly broken |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5.0 | 1.0 | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| naive_0.9x_crush_min_accel | 22.5 | 0.5 | 100 | 18 | 4 | 0.180 | [0.117, 0.267] | 0.780 | 1.405 | 78 | 0 |
| ceiling_near_crush_min_accel | 24.9 | 0.5 | 100 | 1 | 50 | 0.010 | [0.002, 0.054] | 0.490 | 1.405 | 49 | 0 |
| grid_mildest_passing | - | - | 100 | 100 | 0 | 1.000 | [0.963, 1.000] | 0.000 | 0.994 | 0 | 0 |
| accepted_rule | 20.0 | 0.5 | 100 | 27 | 0 | 0.270 | [0.193, 0.364] | 0.730 | 1.405 | 73 | 0 |

### Cell C_sharing_allowed: paired exact McNemar (accepted_rule vs each baseline, task success)

| baseline | b (accepted-only success) | c (baseline-only success) | p-value (two-sided) |
|---|---|---|---|
| none | 73 | 0 | 0.0000 |
| naive_0.9x_crush_min_accel | 4 | 9 | 0.2668 |
| ceiling_near_crush_min_accel | 34 | 10 | 0.0004 |
| grid_mildest_passing | 73 | 0 | 0.0000 |

criterion D: (i) Cell B drop rate lower & p<0.005: True (p=2.647e-23); (ii) Cell C drop rate not higher: True; (iii) crush not increased: True => **PASS**
