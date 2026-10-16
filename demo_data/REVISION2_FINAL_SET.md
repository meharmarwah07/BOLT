# Revision 2: evaluated set and final command (registered before the final run)

Post hoc follow-up to docs/REVISION2.md. docs/SUCCESS_CRITERION.md applies unchanged. The registered primary result of revision 1 stays 0/10.

Evaluated set: every gate-passing rule from all 10 revision-2 runs (Check 1 at 300 train seeds), no selection among runs. k = 10, Bonferroni threshold 0.05/10 = 0.005. Distinct policies: 3 -- (grip 20.0, accel 1.0) x4 [arm A], (20.0, 0.5) x4 [A x1, B x3], (21.0, 0.5) x2 [B]. Identical policies will give identical results; they are reported as one finding, not independent successes.

| run | rule_id | policy | file sha256 |
|---|---|---|---|
| A1 | r_a8e42ed5_2 | 20.0 / 1.0 | 892ee359d2d12782fde44591986bf37139c7858a7a9ebe2bf7723f1eed8ecaa1 |
| A2 | r_6d83263f_2 | 20.0 / 0.5 | 52c21197f6a9713254b29efce7e95758a80e932f95e460d9276f2e96de752625 |
| A3 | r_abcfcb78_1 | 20.0 / 1.0 | 3488848f2b81ce2938a8925c36c077579f615403a4b3f62bf69f9244d6d3dc15 |
| A4 | r_2d8a7a90_4 | 20.0 / 1.0 | 5616a9998f24546db0f21956d165b017ed41467040bc63ef51a51b502c9ce587 |
| A5 | r_cd264c2a_1 | 20.0 / 1.0 | 002f0c3dee9986ee2f672a8bfc2f36761968f3e2bf50072fdc372106aecd6a37 |
| B1 | r_9559550c_7 | 20.0 / 0.5 | 1ba46d29e93037f47e5fbb88e3f94fed628f39d52b7e6a2495d356dd6bf79dbe |
| B2 | r_0838945e_7 | 21.0 / 0.5 | 28dbffe111168c124dcd8aa7cba0b5d773b50f41a1ada7e2d9ac37e512e07149 |
| B3 | r_ba19eb12_9 | 20.0 / 0.5 | f1798d9c0c2f5dc0beb06f3ed310621cf390792ced70c8c469c7ce184a05c2bd |
| B4 | r_6b1ef4fa_7 | 21.0 / 0.5 | 936ebd4b6e57bc97b87ad30afe2b8165d303bd746cb5e565c2fcc491384d86b4 |
| B5 | r_aa2d65bc_9 | 20.0 / 0.5 | 6d6b02db6242b6c776f340e2e3e6f2bef37f9e92e59c8937bde1f07d2f6a90c6 |

Files: pipeline/runs/accepted_rule_<ts>.json, paired with pipeline/runs/loop_<ts>_audit.jsonl, ts in {20261016T171221Z, 20261016T171347Z, 20261016T171539Z, 20261016T171700Z, 20261016T172042Z, 20261016T172520Z, 20261016T172905Z, 20261016T173243Z, 20261016T173645Z, 20261016T174108Z}.

Final command (one invocation, seeds used once): evaluate_heldout.py --held-out, one --rule-file/--audit-path pair per file above, --check1-seeds 300, Cell B seeds 3000-3099 (--cell-b-start 3000 --cell-b-end 3100), Cell C seeds 3100-3199 (--cell-c-start 3100 --cell-c-end 3200), default access log.

Known before the final run: a --dev plumbing run (n=5 on spent seeds 5000-5105) produced FAIL verdicts that reflect only the sample size. The grid-search baseline row reports "none passes v3" (50 train seeds, 6x6), so it is not a competitor and no claim is made against it. The naive 0.9x-crush heuristic is reported alongside every rule.

Reporting commitment: all 10 rules, both Cell C conditions (sharing denied = realistic default; sharing allowed = counterfactual used for criterion D(ii)), the McNemar tables against every baseline, whatever the outcome. No change to rules, prompts, gate, or selection after the first look (criterion G).
