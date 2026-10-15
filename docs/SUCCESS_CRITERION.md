Registered before any revised run. Defers to docs/FINAL_PROTOCOL.md where they overlap. Not to be edited after the first revised run.
A. Gate: a rule is eligible only if it passes Check 1 as registered in docs/REVISION.md.
B. Evaluated set: EVERY gate-passing rule from EVERY revised run (both arms) is evaluated. No selection among runs (FINAL_PROTOCOL.md). Runs that produce no gate-passing rule are reported as "no safe rule" and stay in the denominator. (Selection of the best round WITHIN one run, as implemented in loop.py, is unchanged.)
C. Final evaluation: seeds 3000-3199, ONE evaluator invocation covering all evaluated rules, default access-log path, run once.
D. Per-rule success: (i) Cell B drop rate with the rule is lower than without it, paired exact McNemar p < 0.05/k, where k = number of rules evaluated (Bonferroni); (ii) Cell C drop rate with the rule is not higher than without it; (iii) crush count does not increase on either cell.
E. Headline reporting, per arm: runs executed; runs yielding a gate-passing rule; of those, how many meet D; Wilson intervals. No "best rule" headline.
F. If zero gate-passing rules exist across all runs: no final evaluation is run, 3000-3199 stay unused, and the reported result is "no safe rule found" with the baselines and per-attempt gate outcomes.
G. No rule, prompt, gate, selection or criterion change after the first revised run or the first look at 3000-3199.
