"""Shared MuJoCo model/physics for the gripper cell.

Used by both pick_test.py (interactive viewer tool) and run_trial.py (the
headless batch API). Factored out so the two can't drift apart on object
properties or the motion profile math.
"""
import math

# crush_limit_n matches rules/limits.py's CRUSH_LIMIT_N exactly -- that's an
# already-agreed placeholder shared with Parts B/C, not something to
# reinvent here. mass_kg and friction are NEW: nothing upstream defined
# them, so these are our own simulation design choices, not measurements.
# friction is deliberately low for glass_jar (the project's hidden-failure
# story) and is NOT included in FailureReport -- the Solver has to infer it
# from slip_mm_before_drop, same as a real monitor would.
OBJECT_CLASSES = {
    "steel": {"mass_kg": 0.5, "friction": 0.8, "crush_limit_n": 40.0},
    "cardboard": {"mass_kg": 0.3, "friction": 0.9, "crush_limit_n": 12.0},
    "glass_jar": {"mass_kg": 0.4, "friction": 0.15, "crush_limit_n": 25.0},
}

XML_TEMPLATE = """
<mujoco>
  <!-- noslip_iterations=0 (the default) lets MuJoCo's soft-contact solver
       allow a slow, persistent sliding creep even on a grip well within
       the friction cone (measured: ~2.2 mm/s of pure numerical slip at
       grip_force=20N, holding static). The dedicated no-slip post-solve
       pass eliminates it (measured: 0.000 mm over a 5 s hold). -->
  <option timestep="0.002" gravity="0 0 -9.81" noslip_iterations="5"/>

  <worldbody>
    <light pos="0 0 2"/>

    <geom name="floor" type="plane" size="1 1 0.1"/>
    <geom name="table" type="box" pos="0 0 0.32" size="0.20 0.20 0.02" rgba="0.5 0.5 0.5 1"/>

    <body name="gripper_base" pos="0 0 0.5">
      <joint name="lift_slide" type="slide" axis="0 0 1" range="-0.05 0.4" damping="0.1"/>
      <joint name="carry_slide" type="slide" axis="1 0 0" range="-0.3 0.3" damping="0.1"/>
      <geom name="base" type="box" size="0.08 0.09 0.02" mass="0.5"
            contype="0" conaffinity="0" rgba="0.2 0.3 0.8 1"/>

      <body name="left_finger" pos="0 -0.06 -0.075">
        <joint name="left_slide" type="slide" axis="0 1 0" damping="0.5" range="-0.01 0.045"/>
        <geom name="left_pad" type="box" size="0.015 0.02 0.08" mass="0.1"
              friction="1.0 0.005 0.0001" rgba="0.2 0.8 0.3 1"/>
      </body>

      <body name="right_finger" pos="0 0.06 -0.075">
        <joint name="right_slide" type="slide" axis="0 1 0" damping="0.5" range="-0.01 0.045"/>
        <geom name="right_pad" type="box" size="0.015 0.02 0.08" mass="0.1"
              friction="1.0 0.005 0.0001" rgba="0.9 0.3 0.2 1"/>
      </body>
    </body>

    <body name="object" pos="{obj_x} {obj_y} 0.365">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="box" size="0.025 0.025 0.025" mass="{mass_kg}"
            friction="{friction} 0.005 0.0001" rgba="0.8 0.7 0.2 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="left_motor" joint="left_slide" gear="1" ctrlrange="-40 40"/>
    <motor name="right_motor" joint="right_slide" gear="1" ctrlrange="-40 40"/>
    <position name="lift_motor" joint="lift_slide" kp="3000" kv="300" ctrlrange="-0.05 0.4"/>
    <position name="carry_motor" joint="carry_slide" kp="3000" kv="300" ctrlrange="-0.3 0.3"/>
  </actuator>
</mujoco>
"""


def build_xml(object_class, obj_x=0.0, obj_y=0.0):
    props = OBJECT_CLASSES[object_class]
    return XML_TEMPLATE.format(
        mass_kg=props["mass_kg"], friction=props["friction"], obj_x=obj_x, obj_y=obj_y
    )


def motion_profile(t, distance, accel):
    """Half-sine velocity profile covering `distance` with peak acceleration
    `accel`. Returns (pos, vel, done).

    An earlier bang-bang (accel then -accel) version had a step
    discontinuity in acceleration at the midpoint -- a sudden 2*accel jump
    in the force needed to hold the grasped object -- which was enough to
    break the grip outright in testing. A half-sine has continuous
    acceleration throughout the move (only the start/end instants have a
    jump, same as any finite-force motion), so the required holding force
    never has a sudden reversal mid-move.
    """
    duration = math.pi * math.sqrt(distance / (2 * accel))
    if t >= duration:
        return distance, 0.0, True
    v_peak = math.pi * distance / (2 * duration)
    phase = math.pi * t / duration
    pos = v_peak * duration / math.pi * (1 - math.cos(phase))
    vel = v_peak * math.sin(phase)
    return pos, vel, False
