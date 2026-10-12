"""Seeded batch runs -- the base for Check 1's replays and for measuring
drop rate with a confidence interval (CONTEXT.md section 8).

Seed convention (ours; CONTEXT.md only pins down the held-out range):
TRAIN_SEEDS is for development and Check 1. HELD_OUT_SEEDS (1000-1199, as
CONTEXT.md section 5 specifies) is reserved for the final measurement only
-- never use it while iterating on a rule.
"""
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from run_trial import TrialResult, run_trial

TRAIN_SEEDS = range(0, 200)
HELD_OUT_SEEDS = range(1000, 1200)


@dataclass
class BatchSummary:
    object_class: str
    controller: dict
    n_trials: int
    n_ok: int
    n_drop: int
    n_crush: int
    drop_rate: float
    drop_rate_ci95: tuple
    results: list = field(repr=False)


def _wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval -- behaves better than a naive normal
    approximation at small n or when the proportion is near 0 or 1, both
    of which happen a lot here (many settings hold 100% or drop 100%)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def _run_one(args):
    """Module-level (picklable) worker for ProcessPoolExecutor -- a local
    closure can't be sent to a worker process."""
    object_class, params, seed = args
    return run_trial(object_class, params, seed)


def run_batch(object_class, params, seeds, parallel=False, max_workers=None):
    seeds = list(seeds)
    if parallel:
        tasks = [(object_class, params, seed) for seed in seeds]
        with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as pool:
            results = list(pool.map(_run_one, tasks))
    else:
        results = [run_trial(object_class, params, seed) for seed in seeds]
    n = len(results)
    n_drop = sum(1 for r in results if r.outcome == "drop")
    n_crush = sum(1 for r in results if r.outcome == "crush")
    n_ok = n - n_drop - n_crush
    lo, hi = _wilson_ci(n_drop, n)
    return BatchSummary(
        object_class=object_class,
        controller=dict(params),
        n_trials=n,
        n_ok=n_ok,
        n_drop=n_drop,
        n_crush=n_crush,
        drop_rate=round(n_drop / n, 4) if n else 0.0,
        drop_rate_ci95=(round(lo, 4), round(hi, 4)),
        results=results,
    )


if __name__ == "__main__":
    import argparse

    from cell_model import OBJECT_CLASSES

    p = argparse.ArgumentParser()
    p.add_argument("--object-class", choices=list(OBJECT_CLASSES), default="glass_jar")
    p.add_argument("--grip-force", type=float, default=10.0)
    p.add_argument("--carry-accel", type=float, default=1.0)
    p.add_argument("--held-out", action="store_true", help="use HELD_OUT_SEEDS instead of TRAIN_SEEDS")
    p.add_argument("--n", type=int, default=None, help="limit to the first N seeds (for a quick check)")
    p.add_argument("--parallel", action="store_true")
    args = p.parse_args()

    seeds = HELD_OUT_SEEDS if args.held_out else TRAIN_SEEDS
    if args.n:
        seeds = list(seeds)[: args.n]

    summary = run_batch(
        args.object_class,
        {"grip_force_n": args.grip_force, "carry_accel_ms2": args.carry_accel},
        seeds,
        parallel=args.parallel,
    )
    print(f"BATCH RESULT (headless, {len(seeds)} seeds, {'held-out' if args.held_out else 'train'}):")
    for k in ("object_class", "controller", "n_trials", "n_ok", "n_drop", "n_crush", "drop_rate", "drop_rate_ci95"):
        print(f"  {k}: {getattr(summary, k)}")
