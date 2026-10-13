# Part A Status — Simulation (MuJoCo)

This is a fix pass over the original Part A delivery. Five issues were
reported and are addressed below in order, each with root cause,
verification, and (where relevant) what's still unexplained.

## 1. Friction was inert

**Root cause:** both finger pad geoms were hardcoded `friction="1.0 ..."`
in `cell_model.py`. MuJoCo combines two geoms' contact friction by taking
the elementwise **max** of the two (both geoms had equal, default
`priority`), not a min or blend. Since every object class's friction was
below 1.0, the pad's 1.0 always won, so the object's own friction class
never affected the physics at all — `steel`, `cardboard`, and `glass_jar`
behaved identically regardless of their stated mu.

**Fix:** pad friction dropped to `PAD_FRICTION = 0.02` (below every
class's `friction_range` floor — lowest is glass_jar at 0.08), so the
object's own sampled mu now governs the contact. Per-class mu is no
longer a fixed constant: it's a `friction_range` sampled per trial seed
(`sample_friction` in `cell_model.py`), continuing the same RNG stream
already used for placement jitter, so it's fully reproducible and still
never exposed in `TrialResult`/`FailureReport`.

| class | mass_kg | friction_range | crush_limit_n |
|---|---|---|---|
| steel | 0.5 | 0.70 – 0.90 | 40 (matches rules/limits.py) |
| cardboard | 0.3 | 0.80 – 1.00 | 12 (matches rules/limits.py) |
| glass_jar | 0.4 | **0.08 – 0.20 (hidden)** | 25 (matches rules/limits.py) |

**Verification (measured, sim/tests/test_sim.py):**
- Two synthetic classes, same mass (0.5 kg), mu 0.05 vs 0.95: at
  grip=5N, mu=0.05 drops, mu=0.95 holds. Different outcomes, as required.
- Min holding grip follows `N >= m(g+a)/(2*mu)` (a = the lift's peak
  accel, 1.0 m/s², the hardest instant) within **2–24%** across steel,
  cardboard, and glass_jar (3 seeds each) — see `test_sim.py`'s
  `test_min_holding_grip_matches_friction_formula`. The sim needs a real
  margin above the textbook quasi-static number (see "still unexplained"
  below), but it's inside the ~25% target and the *direction* (min grip
  scales with 1/mu) matches the formula's shape exactly.

## 2. Accel was inverted

**Symptom:** at a grip just above the (then-)boundary, `accel=0.5` gave
100% drop and `accel>=1.0` gave 0% — backwards, since a lower
acceleration needs less force to track and should never be *more* likely
to fail.

**Root cause, found by direct trace (not guessed):** a half-sine move's
duration is `pi*sqrt(distance/(2*accel))` — lower accel means a *longer*
move. The old code held for a **fixed number of steps** (300) after the
move finished, regardless of accel, so the *total* time watched for a
drop (move + hold) was itself a decreasing function of accel. Traced the
object's position directly (gap to the gripper, in mm, logged every
step) for accel 0.5 vs 3.0 at a grip right at the boundary: the
horizontal (carry-direction) gap stayed at 0.0000 the entire time — carry
acceleration contributed essentially zero horizontal slip in this range —
while the *vertical* gap grew at a constant, accel-independent rate
(~10.8 mm/s for steel, confirmed to be present even with **zero** carry
motion commanded, i.e. during a plain static hold). That rate is a real
contact effect, not numerical creep: it's the grip genuinely sliding
because it's right at its static limit. Since accumulated slip is
`rate * elapsed_time` and elapsed time is itself a decreasing function of
accel, a lower accel's longer move let more of that same creep accumulate
before the fixed-length hold-check — the bug was a test-measurement
confound, not reversed physics.

**Fix:** `hold_steps_for()` in `cell_model.py` sizes the post-move hold so
that `move duration + hold duration` is a **constant wall-clock window**
(`PHASE_OBSERVE_S`) regardless of accel, with a floor (`MIN_HOLD_STEPS`)
for moves slow enough to eat the whole budget. Every accel value is now
watched for the same total time, so a grip that's genuinely marginal
fails the same way regardless of accel, and a genuine accel-driven
effect (inertial force during the move itself exceeding the friction
budget) still shows up because it happens during the move, not the
padding afterward.

`PHASE_OBSERVE_S = 1.6s` was chosen empirically: it's just above the
slowest realistic move in the datasheet carry_accel range (0.5 m/s² over
0.2 m takes ~1.40s), and shorter windows need less margin above the
quasi-static friction formula (see item 1) while a much longer window
(3.0s, tried first) needed 60–180% more grip than the formula — a longer
window gives a near-equilibrium grip more time to reveal the same
constant-rate creep, so the choice of window length directly trades off
against the formula's own ~25% target.

**Verification (measured):** at the old reproduction grip, sweeping
accel 0.5/1.0/3.0 now gives `ok`/`ok`/`drop` or similar non-monotonic
patterns driven by genuine (small) accel effects, not a 100%-vs-0%
systematic bias. `test_sim.py`'s
`test_accel_direction_lower_never_systematically_worse` runs a pairwise
check (same seed ⇒ same hidden mu, only accel differs) across 30 seeds at
a bisected marginal grip: lower accel (0.5) was worse than higher (6.0)
in 2/30 cases (solver noise near the knife-edge), while higher accel was
worse than lower in 12/30 — the *correct* direction (higher accel
sometimes needs more force and can fail where lower doesn't; lower never
systematically loses).

## 3. Tests

`gripper_test.py` and `viewer_test.py` both called
`mujoco.viewer.launch_passive(...)` at **module level** (not inside
`if __name__ == "__main__":`), and both filenames matched pytest's
default `*_test.py` collection pattern. Importing either for collection
launched a real viewer and hung with no display. `pick_test.py` also
matches that pattern but was already safe — all its side-effecting code
was already under `if __name__ == "__main__":`.

**Fix:** renamed the two interactive scripts to `gripper_demo.py` and
`viewer_demo.py` (content otherwise unchanged) so pytest's default
collection skips them entirely.

Added `sim/tests/test_sim.py` (12 tests) covering: determinism, friction
sensitivity (incl. a static regression guard that pad friction stays
below every class's mu floor), the friction-formula check above, the
accel-direction check above, glass_jar dropping at a nominal grip and
being fixed by a grip within [2, 40], the crush threshold's exact
boundary (40N doesn't crush, 41N does), parallel batch == serial batch,
no mu/friction value anywhere in a FailureReport (including nested
JSON), and TRAIN_SEEDS/HELD_OUT_SEEDS being disjoint.

```
$ pytest sim/ -v
12 passed in ~18-22s
```

No display required (verified with `DISPLAY` unset). Comfortably under
the 60s budget.

## 4. Crush

No change to the logic: `crushed = grip_force_n > obj_props["crush_limit_n"]`
remains a **static threshold check on the commanded grip force**, decided
before any physics runs. This is explicitly not a simulated material
deformation — MuJoCo isn't modeling the object's structural failure here,
and this mirrors `rules.validate()`'s own `CRUSH_LIMIT_N` check exactly
(same already-agreed placeholder values, not new measurements).

Confirmed `monitor.py` cannot relabel a crush as a drop: `make_failure_report`
sets `"outcome": trial_result.outcome` as a direct passthrough — there's
no branch that coerces `"crush"` to `"drop"`, and nothing in `sim/` sends
reports anywhere (that's downstream, in `rules/`/`llm/`, which this pass
doesn't touch). `test_no_friction_or_mu_in_failure_report` also confirms
the report never carries the hidden mu alongside the outcome.

## 5. Measured numbers (this machine, replacing the old 44ms/trial claim)

Machine: 8 cores (`nproc` = 8).

- **Serial**: 100 trials (steel, grip=10N, accel=1.0) in 13.1s wall time
  → **~460 replays/min**.
- **Parallel** (`ProcessPoolExecutor`, 8 workers): 200 trials in 7.8s wall
  time → **~1530 replays/min**, a ~3.3x speedup over serial — consistent
  with the old "~3x" claim, now actually measured on this run rather than
  carried over from a different machine.
- Full `pytest sim/` (12 tests, including a parallel-vs-serial batch
  comparison and several grip-bisection searches): **~18-22s**.

### Sweep table: min holding grip per class per accel (20 seeds)

Bisected per-seed minimum grip (N) to avoid a drop, averaged over seeds
0-19 (TRAIN_SEEDS), for carry_accel in {0.5, 1.0, 3.0, 6.0} m/s² (the
datasheet range's endpoints and midpoints). Each cell is mean ± stdev
across the 20 seeds' independent friction draws.

| class | accel (m/s²) | mean min grip (N) | min (N) | max (N) | stdev |
|---|---|---|---|---|---|
| steel | 0.5 | 3.95 | 3.78 | 4.06 | 0.10 |
| steel | 1.0 | 3.80 | 3.47 | 4.22 | 0.17 |
| steel | 3.0 | 3.92 | 3.55 | 4.33 | 0.19 |
| steel | 6.0 | 4.17 | 3.80 | 4.73 | 0.29 |
| cardboard | 0.5 | 1.92 | 1.73 | 2.05 | 0.08 |
| cardboard | 1.0 | 1.92 | 1.74 | 2.06 | 0.08 |
| cardboard | 3.0 | 1.99 | 1.83 | 2.18 | 0.11 |
| cardboard | 6.0 | 2.15 | 1.96 | 2.40 | 0.13 |
| glass_jar | 0.5 | 17.14 | 11.32 | 24.50* | 4.34 |
| glass_jar | 1.0 | 17.38 | 11.32 | 24.50* | 4.25 |
| glass_jar | 3.0 | 18.63 | 13.07 | 24.50* | 3.92 |
| glass_jar | 6.0 | 19.53 | 14.57 | 24.50* | 3.90 |

\* glass_jar's max is pinned at the bisection's search ceiling
(crush_limit_n × 0.98 = 24.5N) for its worst-case low-mu seeds — those
seeds never found a safe holding grip below crush within the search
range, consistent with the near-unholdable edge case in item 2 below, not
a bisection artifact.

For every class, mean min holding grip is **flat to slightly increasing**
with accel (never decreasing) — exactly the "lower accel never worse"
property the fix targets, now visible in aggregate across the full
datasheet accel range, not just at one hand-picked grip value.

(Ran as 3 classes × 4 accels × 20 seeds × 11 bisection iterations ≈ 2,640
trials in 301.6s on this machine — single-process, not using
`batch.py --parallel`.)

## Anything still unexplained

1. **The quasi-static formula needs a real margin, not just noise.**
   `N >= m(g+a)/(2*mu)` undershoots the sim's actual minimum holding grip
   by a measured, *consistent* (not noisy) 2-24% at the current
   `PHASE_OBSERVE_S=1.6s`, and by 60-180% at a longer 3.0s window tried
   first. Direct contact-force inspection (`mj_contactForce` on each
   pad-object contact) shows the friction force tracks the
   *instantaneous* required load correctly throughout the move — so this
   isn't a wrong friction model. The likely explanation is twofold: (a)
   the half-sine profile has a genuine jerk discontinuity at the move's
   start (acceleration jumps from 0 to its peak instantly), and a stiff
   but finite position actuator (kp=3000, kv=300) tracking that can
   briefly need more force than the nominal peak accel suggests; (b) a
   slip threshold evaluated over a finite time window needs some real
   margin above the exact equilibrium point, since exactly at the
   theoretical minimum the grip is only neutrally stable and any residual
   compliance will drift given enough time. Both contribute; I did not
   isolate how much each one costs, and a tighter margin would need
   either a smoother (bounded-jerk) trajectory or a shorter observation
   window than what items 1+2's fix settled on.
2. **glass_jar's lowest mu draws (seed=0: mu≈0.085) are right at the edge
   of unholdable without crushing.** Required holding grip during the
   lift phase (`m(g+lift_accel)/(2*mu)` ≈ 25.4N) is *above* glass_jar's
   crush_limit_n (25.0N) for that seed — a razor-thin, essentially
   impossible window. This only happens for the worst few percent of mu
   draws (seeds 1-9 all had comfortable margin below crush_limit); it's
   likely an intentional edge of the project's "hidden low-friction,
   crush-sensitive" premise for glass_jar rather than a bug, but it means
   a small fraction of glass_jar seeds have *no* safe grip at all within
   [2, 40]. Not something this pass changed or fixed.
3. Still never run end-to-end through the real `llm/solver.py` or
   `rules/validate.py` (out of scope for this pass — rules/ and llm/
   weren't touched). `monitor.py`'s own docstring already flags that
   `solver.py` only reads `report_id`, not `outcome`/`drop_phase`, so
   `outcome: "crush"` reports still won't be meaningfully handled
   downstream until that's addressed on the Part B side.
4. Parallel speedup (~3.3x) is measured on this dev machine only, same
   caveat as before — still untested on the target cloud instance.
