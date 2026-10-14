"""Shared MuJoCo model/physics for the gripper cell.

Used by both pick_test.py (interactive viewer tool) and run_trial.py (the
headless batch API). Factored out so the two can't drift apart on object
properties or the motion profile math.
"""
import math

# crush_limit_n matches rules/limits.py's CRUSH_LIMIT_N exactly -- that's an
# already-agreed placeholder shared with Parts B/C, not something to
# reinvent here. mass_kg and friction_range are NEW: nothing upstream
# defined them, so these are our own simulation design choices, not
# measurements. friction_range is deliberately low for glass_jar (the
# project's hidden-failure story) and the sampled value is NOT included in
# FailureReport/TrialResult -- the Solver has to infer it from
# slip_mm_before_drop, same as a real monitor would. It is jittered per
# trial seed (see sample_friction) rather than fixed, so a class's true mu
# is never directly observable from repeated runs at the same grip force.
OBJECT_CLASSES = {
    "steel": {"mass_kg": 0.5, "friction_range": (0.70, 0.90), "crush_limit_n": 40.0},
    "cardboard": {"mass_kg": 0.3, "friction_range": (0.80, 1.00), "crush_limit_n": 12.0},
    "glass_jar": {"mass_kg": 0.4, "friction_range": (0.08, 0.20), "crush_limit_n": 25.0},
}

# Represents imprecise upstream part placement (a real feeder/conveyor
# wouldn't place a part at an exact (0,0) every time). +/-5mm is a
# simulation design choice, not a measured tolerance.
PLACEMENT_JITTER_M = 0.005


def sample_friction(object_class, rng):
    lo, hi = OBJECT_CLASSES[object_class]["friction_range"]
    return float(rng.uniform(lo, hi))


def nominal_friction(object_class):
    """Midpoint of the class's friction_range -- used only where no seed is
    given (e.g. pick_test.py's interactive default), never for the
    statistical sweeps, which must use sample_friction(..., rng)."""
    lo, hi = OBJECT_CLASSES[object_class]["friction_range"]
    return (lo + hi) / 2.0


XML_TEMPLATE = """
<mujoco>
  <!-- noslip_iterations=5 enables MuJoCo's dedicated no-slip post-solve
       pass. An earlier version of this comment claimed noslip_iterations=0
       (the default) produced ~2.2 mm/s of persistent numerical slip at
       grip_force=20N holding static; that could not be reproduced on this
       build (measured <0.001mm over a 5s static hold at both 0 and 5) and
       has been removed rather than left uncorrected. -->
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
        <!-- MuJoCo combines two geoms' contact friction by taking the
             elementwise MAX (equal priority, the default here), not the
             min or some blend. A pad friction of 1.0 was therefore always
             the winner against every object's mu (all < 1.0), which made
             the object's own friction class completely inert. Pad
             friction is set to {pad_friction}, below every object class's
             friction_range floor (lowest is glass_jar at 0.08), so the
             object's own sampled mu is what actually governs the contact. -->
        <geom name="left_pad" type="box" size="0.015 0.02 0.08" mass="0.1"
              friction="{pad_friction} 0.005 0.0001" rgba="0.2 0.8 0.3 1"/>
      </body>

      <body name="right_finger" pos="0 0.06 -0.075">
        <joint name="right_slide" type="slide" axis="0 1 0" damping="0.5" range="-0.01 0.045"/>
        <geom name="right_pad" type="box" size="0.015 0.02 0.08" mass="0.1"
              friction="{pad_friction} 0.005 0.0001" rgba="0.9 0.3 0.2 1"/>
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


PAD_FRICTION = 0.02


def build_xml(object_class, obj_x=0.0, obj_y=0.0, friction=None):
    props = OBJECT_CLASSES[object_class]
    if friction is None:
        friction = nominal_friction(object_class)
    return XML_TEMPLATE.format(
        mass_kg=props["mass_kg"],
        friction=friction,
        pad_friction=PAD_FRICTION,
        obj_x=obj_x,
        obj_y=obj_y,
    )


def phase_duration(distance, accel):
    """Wall-clock duration of a half-sine move covering `distance` at peak
    acceleration `accel`. Shared with motion_profile so callers can size a
    accel-independent observation window (see PHASE_OBSERVE_S) without
    duplicating this formula."""
    return math.pi * math.sqrt(distance / (2 * accel))


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
    duration = phase_duration(distance, accel)
    if t >= duration:
        return distance, 0.0, True
    v_peak = math.pi * distance / (2 * duration)
    phase = math.pi * t / duration
    pos = v_peak * duration / math.pi * (1 - math.cos(phase))
    vel = v_peak * math.sin(phase)
    return pos, vel, False


# How long (wall-clock) a drop check watches a phase (move + hold) before
# deciding ok/drop. MUST be a fixed constant, not derived from the move's
# own duration -- see hold_steps_for's docstring for why. Set to just
# above the slowest realistic move in the datasheet carry_accel range
# (0.5 m/s^2 over CARRY_DISTANCE_M=0.2m takes ~1.40s) so the floor below
# essentially never overrides it within that range -- a window much
# longer than the slowest move gives a near-equilibrium grip more total
# time to reveal a slow creep, which inflates the empirical min-holding
# grip well past the quasi-static m*(g+a)/(2*mu) estimate (measured: a
# 3.0s window needed ~60-180% more grip than the formula; 1.6s needs
# ~2-24%, still comfortably within the ~25% the formula is supposed to
# hold to for a real mu/mass/accel combination in this cell's range).
PHASE_OBSERVE_S = 1.6

# Floor on hold time after the move finishes, in steps, even if the move
# alone ate the whole PHASE_OBSERVE_S budget (e.g. a very low accel on a
# long distance). Keeps a minimum settle-and-check window in that case.
MIN_HOLD_STEPS = 150


def hold_steps_for(distance, accel, dt):
    """Steps to hold after a phase's move finishes, sized so that
    (move duration + hold duration) is the SAME fixed PHASE_OBSERVE_S
    regardless of accel.

    Root cause this exists to fix: a half-sine move's duration is
    math.pi*sqrt(distance/(2*accel)) -- lower accel means a longer move.
    The old code held for a FIXED NUMBER OF STEPS after the move
    regardless of accel, so the total watched time (move+hold) was itself
    a decreasing function of accel. Near a marginal grip, the object
    creeps out of the fingers at a roughly constant rate (confirmed by
    direct trace: ~10.8 mm/s for steel at grip_force just above the old
    "boundary", present even with zero carry motion commanded -- this is
    the contact genuinely sliding at a grip right at its static limit,
    not a numerical artifact). Accumulated slip is therefore mostly a
    function of elapsed time, not of accel. Lower accel -> longer move ->
    more total elapsed time before the slip check -> more accumulated
    creep -> spuriously flagged as a drop, even though accel itself
    (checked via the gripper-object X gap during the move) contributed
    ~0 mm of horizontal slip in that trace. Holding every accel value to
    the same total observation window removes that confound: a genuinely
    marginal grip now fails the same way regardless of accel, and a
    genuine accel-driven failure (inertial force during the move
    exceeding the friction budget) still shows up because it happens
    during the move itself, not the padding afterward.
    """
    duration = phase_duration(distance, accel)
    return max(MIN_HOLD_STEPS, round((PHASE_OBSERVE_S - duration) / dt))
