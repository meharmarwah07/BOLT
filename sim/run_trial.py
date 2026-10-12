"""run_trial(object_class, params, seed) -> TrialResult.

The Part A deliverable that the Monitor (FailureReport generator) and
Check 1's replays are built on. Headless only, no viewer/threading -- each
call builds its own fresh MjModel/MjData, so calls are independent and
safe to run in parallel (parallel execution itself is untested so far).

Reuses the exact physics validated interactively in pick_test.py: the same
object classes, the same force-controlled fingers, the same half-sine
lift/carry motion profiles, the same noslip_iterations fix. See
cell_model.py for that shared setup and pick_test.py's history for the
bugs that were found and fixed along the way (table-geom overlap, jerky
acceleration profile, a solver-breaking kinematic override, MuJoCo's
default-off no-slip solver pass).
"""
import math
from dataclasses import asdict, dataclass
from typing import Optional

import mujoco
import numpy as np

from cell_model import OBJECT_CLASSES, build_xml, motion_profile

# Fixed implementation details, NOT controller parameters. Only
# grip_force_n and carry_accel_ms2 (in `params`) are real controller knobs
# per CONTEXT.md section 7.
LIFT_HEIGHT_M = 0.15
LIFT_ACCEL_MS2 = 1.0
CARRY_DISTANCE_M = 0.2

SLIP_DROP_THRESHOLD_MM = 20.0

# Represents imprecise upstream part placement (a real feeder/conveyor
# wouldn't place a part at an exact (0,0) every time). +/-5mm is a
# simulation design choice, not a measured tolerance.
PLACEMENT_JITTER_M = 0.005


@dataclass
class TrialResult:
    object_class: str
    seed: int
    controller: dict
    outcome: str  # "ok" | "drop" | "crush"
    drop_phase: Optional[str]  # "lift" | "carry" | None
    slip_mm_before_drop: float
    est_mass_kg: float  # ground truth from the sim, NOT a noisy sensor estimate

    def as_dict(self):
        return asdict(self)


def run_trial(object_class, params, seed):
    grip_force_n = params["grip_force_n"]
    carry_accel_ms2 = params["carry_accel_ms2"]
    obj_props = OBJECT_CLASSES[object_class]

    rng = np.random.default_rng(seed)
    obj_x, obj_y = rng.uniform(-PLACEMENT_JITTER_M, PLACEMENT_JITTER_M, size=2)

    model = mujoco.MjModel.from_xml_string(build_xml(object_class, obj_x, obj_y))
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    lift_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_motor")
    carry_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "carry_motor")
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    grip_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")

    def step(lift_pos, carry_pos=0.0):
        data.ctrl[lift_act] = lift_pos
        data.ctrl[carry_act] = carry_pos
        mujoco.mj_step(model, data)

    def gap_xz():
        return (
            data.xpos[obj_bid][0] - data.xpos[grip_bid][0],
            data.xpos[obj_bid][2] - data.xpos[grip_bid][2],
        )

    def slip_mm_from(baseline_xz):
        gx, gz = gap_xz()
        bx, bz = baseline_xz
        return math.hypot(gx - bx, gz - bz) * 1000.0

    crushed = grip_force_n > obj_props["crush_limit_n"]

    # Phase 1: settle.
    for _ in range(300):
        step(0.0)

    # Phase 2: grip. Ramp finger force 0 -> grip_force_n, then hold.
    ramp_steps = 200
    for i in range(ramp_steps):
        f = grip_force_n * (i + 1) / ramp_steps
        data.ctrl[0] = f
        data.ctrl[1] = -f
        step(0.0)
    for _ in range(300):
        step(0.0)

    grasp_dropped = data.xpos[obj_bid][2] < 0.33
    drop_phase = None
    max_slip_mm = 0.0

    # Phase 3: lift.
    baseline_lift = gap_xz()
    t = 0.0
    while True:
        pos, vel, done = motion_profile(t, LIFT_HEIGHT_M, LIFT_ACCEL_MS2)
        data.ctrl[0] = grip_force_n
        data.ctrl[1] = -grip_force_n
        step(pos)
        t += dt
        max_slip_mm = max(max_slip_mm, slip_mm_from(baseline_lift))
        if done:
            break
    for _ in range(300):
        step(LIFT_HEIGHT_M)
        max_slip_mm = max(max_slip_mm, slip_mm_from(baseline_lift))

    lift_dropped = (
        grasp_dropped
        or data.xpos[obj_bid][2] < 0.33
        or max_slip_mm > SLIP_DROP_THRESHOLD_MM
    )
    if lift_dropped:
        drop_phase = "lift"

    # Phase 4: carry. Only if the lift actually held.
    if not lift_dropped:
        baseline_carry = gap_xz()
        t = 0.0
        while True:
            pos, vel, done = motion_profile(t, CARRY_DISTANCE_M, carry_accel_ms2)
            data.ctrl[0] = grip_force_n
            data.ctrl[1] = -grip_force_n
            step(LIFT_HEIGHT_M, pos)
            t += dt
            max_slip_mm = max(max_slip_mm, slip_mm_from(baseline_carry))
            if done:
                break
        for _ in range(300):
            step(LIFT_HEIGHT_M, CARRY_DISTANCE_M)
            max_slip_mm = max(max_slip_mm, slip_mm_from(baseline_carry))

        carry_dropped = (
            data.xpos[obj_bid][2] < 0.33 or max_slip_mm > SLIP_DROP_THRESHOLD_MM
        )
        if carry_dropped:
            drop_phase = "carry"

    if crushed:
        outcome = "crush"
    elif drop_phase is not None:
        outcome = "drop"
    else:
        outcome = "ok"

    return TrialResult(
        object_class=object_class,
        seed=seed,
        controller={"grip_force_n": grip_force_n, "carry_accel_ms2": carry_accel_ms2},
        outcome=outcome,
        drop_phase=drop_phase,
        slip_mm_before_drop=round(max_slip_mm, 2),
        est_mass_kg=obj_props["mass_kg"],
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--object-class", choices=list(OBJECT_CLASSES), default="steel")
    p.add_argument("--grip-force", type=float, default=10.0)
    p.add_argument("--carry-accel", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    result = run_trial(
        args.object_class,
        {"grip_force_n": args.grip_force, "carry_accel_ms2": args.carry_accel},
        args.seed,
    )
    print("TRIAL RESULT (headless, this run):")
    for k, v in result.as_dict().items():
        print(f"  {k}: {v}")
