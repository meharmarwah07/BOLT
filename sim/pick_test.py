"""Step 2: pick and carry (grasp, lift vertically, then carry horizontally).

Fingers are force-controlled motors: ctrl is the grip force in Newtons
directly, so grip_force_n is a real physical quantity, not a position
setpoint. The gripper's lift (vertical) and carry (horizontal) motions are
each driven by a stiff position actuator tracking a smooth (half-sine)
trajectory, rather than being teleported directly via qpos/qvel -- an
earlier version did that and it broke the solver's own force balance
between the gripper and the grasped object, causing persistent slip that
looked locally fine but never self-corrected. The real actuators have some
small PD tracking lag behind the commanded trajectory, which is a
legitimate dynamic effect, not an artifact.

carry_accel_ms2 is the real controller parameter from CONTEXT.md section 7
(datasheet range 0.5-6.0 m/s^2). lift_accel is NOT a controller parameter --
it's a fixed implementation detail of how the gripper gets to carry height.

Still missing vs. the full brief: object classes, seeded randomization,
run_trial()'s real signature, and the FailureReport generator.

Headless by default so results are reproducible without a viewer. Pass
--viewer to watch it.
"""
import argparse
import math
import threading
import time

import glfw
import mujoco
import mujoco.viewer

from cell_model import OBJECT_CLASSES, build_xml, motion_profile


class _RestartRequested(Exception):
    """Raised by step() when the viewer's reset key was pressed mid-trial."""


def run(object_class, grip_force_n, lift_height, lift_accel, carry_distance, carry_accel, use_viewer):
    obj_props = OBJECT_CLASSES[object_class]
    model = mujoco.MjModel.from_xml_string(build_xml(object_class))
    data = mujoco.MjData(model)

    lift_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_motor")
    carry_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "carry_motor")

    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    grip_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")

    # Backspace is MuJoCo's own "reset" key, but launch_passive doesn't know
    # our phases are scripted in Python -- it would just reset qpos/qvel and
    # sit there. This callback catches the keypress ourselves so we can
    # actually restart the whole trial.
    reset_requested = threading.Event()

    def on_key(keycode):
        if keycode == glfw.KEY_BACKSPACE:
            reset_requested.set()

    viewer_ctx = None
    if use_viewer:
        viewer_ctx = mujoco.viewer.launch_passive(model, data, key_callback=on_key)

    # The on-screen "Reset" button in the Simulation panel is handled
    # natively by MuJoCo's own UI thread -- it calls mj_resetData on `data`
    # directly and never reaches our key_callback (that only fires for real
    # keyboard events). Catching the keypress alone misses it, and racing
    # our main thread's mj_step against that reset is what caused the
    # occasional corruption. Watching data.time for a backward jump catches
    # *any* external reset, however it was triggered, without needing a
    # thread lock.
    last_seen_time = 0.0

    # Earlier version drove the lift joint by overwriting qpos/qvel directly
    # every step (teleporting it to a scripted trajectory). That bypasses
    # MuJoCo's constraint solver's own force balance between the gripper and
    # the grasped object: contact/friction forces looked locally correct
    # each instant, but the object slipped continuously anyway because the
    # override broke the solver's self-consistency between steps. Driving
    # the same trajectory through a real position actuator instead lets the
    # solver handle gripper/object coupling the normal, tested way -- at the
    # cost of some small PD tracking lag behind the commanded trajectory.
    def step(lift_pos, carry_pos=0.0):
        nonlocal last_seen_time
        if reset_requested.is_set() or data.time < last_seen_time:
            raise _RestartRequested()
        step_start = time.time()
        data.ctrl[lift_act] = lift_pos
        data.ctrl[carry_act] = carry_pos
        mujoco.mj_step(model, data)
        last_seen_time = data.time
        if viewer_ctx is not None:
            viewer_ctx.sync()
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    def run_one_trial():
        # Crush is a simple threshold check against the commanded force, not
        # a simulated material failure -- MuJoCo isn't modeling deformation
        # here, so this mirrors exactly what rules.validate() already does
        # with CRUSH_LIMIT_N.
        crushed = grip_force_n > obj_props["crush_limit_n"]

        # Phase 1: settle. Object falls onto the table, fingers open (ctrl=0).
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

        grasp_dropped = data.xpos[obj_bid][2] < 0.33  # fell back to table top

        def gap_xz():
            return (
                data.xpos[obj_bid][0] - data.xpos[grip_bid][0],
                data.xpos[obj_bid][2] - data.xpos[grip_bid][2],
            )

        def slip_mm_from(baseline_xz):
            gx, gz = gap_xz()
            bx, bz = baseline_xz
            return math.hypot(gx - bx, gz - bz) * 1000.0

        SLIP_DROP_THRESHOLD_MM = 20.0
        drop_phase = None
        dt = model.opt.timestep

        # Phase 3: lift. Drive the lift actuator along the smooth profile.
        baseline_lift = gap_xz()
        max_slip_lift_mm = 0.0
        t = 0.0
        while True:
            pos, vel, done = motion_profile(t, lift_height, lift_accel)
            data.ctrl[0] = grip_force_n
            data.ctrl[1] = -grip_force_n
            step(pos)
            t += dt
            max_slip_lift_mm = max(max_slip_lift_mm, slip_mm_from(baseline_lift))
            if done:
                break
        for _ in range(300):  # hold at top, see if it keeps slipping
            step(lift_height)
            max_slip_lift_mm = max(max_slip_lift_mm, slip_mm_from(baseline_lift))

        lift_dropped = (
            grasp_dropped
            or data.xpos[obj_bid][2] < 0.33
            or max_slip_lift_mm > SLIP_DROP_THRESHOLD_MM
        )
        if lift_dropped:
            drop_phase = "lift"

        # Phase 4: carry. Only attempt it if the lift actually held -- carrying
        # a part that's already dropped isn't a meaningful test.
        max_slip_carry_mm = 0.0
        if not lift_dropped:
            baseline_carry = gap_xz()
            t = 0.0
            while True:
                pos, vel, done = motion_profile(t, carry_distance, carry_accel)
                data.ctrl[0] = grip_force_n
                data.ctrl[1] = -grip_force_n
                step(lift_height, pos)
                t += dt
                max_slip_carry_mm = max(max_slip_carry_mm, slip_mm_from(baseline_carry))
                if done:
                    break
            for _ in range(300):  # hold at the end of the carry
                step(lift_height, carry_distance)
                max_slip_carry_mm = max(max_slip_carry_mm, slip_mm_from(baseline_carry))

            carry_dropped = (
                data.xpos[obj_bid][2] < 0.33
                or max_slip_carry_mm > SLIP_DROP_THRESHOLD_MM
            )
            if carry_dropped and drop_phase is None:
                drop_phase = "carry"

        final_obj_z = float(data.xpos[obj_bid][2])
        final_grip_z = float(data.xpos[grip_bid][2])
        final_obj_x = float(data.xpos[obj_bid][0])
        final_grip_x = float(data.xpos[grip_bid][0])

        if crushed:
            outcome = "crush"
        elif drop_phase is not None:
            outcome = "drop"
        else:
            outcome = "ok"

        return {
            "object_class": object_class,
            "outcome": outcome,
            "crushed": crushed,
            "grip_force_n": grip_force_n,
            "lift_height_m": lift_height,
            "lift_accel_ms2": lift_accel,
            "carry_distance_m": carry_distance,
            "carry_accel_ms2": carry_accel,
            "dropped_during_grasp": grasp_dropped,
            "drop_phase": drop_phase,
            "dropped": drop_phase is not None,
            "max_slip_lift_mm": round(max_slip_lift_mm, 2),
            "max_slip_carry_mm": round(max_slip_carry_mm, 2),
            "final_object_x": round(final_obj_x, 4),
            "final_gripper_x": round(final_grip_x, 4),
            "final_object_z": round(final_obj_z, 4),
            "final_gripper_z": round(final_grip_z, 4),
        }

    if viewer_ctx is None:
        return run_one_trial()

    result = None
    while viewer_ctx.is_running():
        reset_requested.clear()
        mujoco.mj_resetData(model, data)
        last_seen_time = 0.0
        try:
            result = run_one_trial()
            print("SIMULATION RESULT (this run):")
            for k, v in result.items():
                print(f"  {k}: {v}")
            print("(press Backspace in the viewer to reset and replay)")
            idle_carry = 0.0 if result["drop_phase"] == "lift" else carry_distance
            while viewer_ctx.is_running() and not reset_requested.is_set():
                step(lift_height, idle_carry)
        except _RestartRequested:
            continue
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--object-class", choices=list(OBJECT_CLASSES), default="steel")
    p.add_argument("--grip-force", type=float, default=10.0, help="N per finger motor")
    p.add_argument("--lift-height", type=float, default=0.15, help="m to raise the gripper")
    p.add_argument("--lift-accel", type=float, default=1.0, help="m/s^2 (fixed, not a controller param)")
    p.add_argument("--carry-distance", type=float, default=0.2, help="m to move horizontally")
    p.add_argument("--carry-accel", type=float, default=1.0, help="m/s^2, the real carry_accel_ms2 controller param (datasheet range 0.5-6.0)")
    p.add_argument("--viewer", action="store_true")
    args = p.parse_args()

    result = run(
        args.object_class,
        args.grip_force,
        args.lift_height,
        args.lift_accel,
        args.carry_distance,
        args.carry_accel,
        args.viewer,
    )
    if not args.viewer:
        print("SIMULATION RESULT (headless):")
        for k, v in result.items():
            print(f"  {k}: {v}")
