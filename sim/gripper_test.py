
import time
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <light pos="0 0 2"/>

    <!-- Floor -->
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <geom name="table" type="box" pos="0 0 0.32" size="0.20 0.20 0.02" rgba="0.5 0.5 0.5 1"/>

    <!-- Fixed gripper frame -->
    <body name="gripper_base" pos="0 0 0.5">
      <geom name="base" type="box"
      size="0.08 0.04 0.02"
      contype="0" conaffinity="0"
      rgba="0.2 0.3 0.8 1"/>

      <!-- Left finger: slides along the Y axis -->
      <body name="left_finger" pos="0 -0.06 -0.10">
        <joint name="left_slide" type="slide" axis="0 1 0"/>
        <geom name="left_pad" type="box"
              size="0.015 0.02 0.05"
              mass="0.1"
              rgba="0.2 0.8 0.3 1"/>
      </body>

      <!-- Right finger: slides along the Y axis -->
      <body name="right_finger" pos="0 0.06 -0.10">
        <joint name="right_slide" type="slide" axis="0 1 0"/>
        <geom name="right_pad" type="box"
              size="0.015 0.02 0.05"
              mass="0.1"
              rgba="0.9 0.3 0.2 1"/>
      </body>
    </body>

    <!-- Object placed between the fingers -->
    <body name="object" pos="0 0 0.365">
      <freejoint/>
      <geom name="object_geom" type="box"
            size="0.025 0.025 0.025"
            mass="0.2"
            rgba="0.8 0.7 0.2 1"/>
    </body>
  </worldbody>

  
<actuator>
  <position name="left_motor" joint="left_slide" kp="35" kv="7" ctrlrange="-0.04 0.04"/>
  <position name="right_motor" joint="right_slide" kp="35" kv="7" ctrlrange="-0.04 0.04"/>
</actuator>

</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)


left_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "left_slide"
)
right_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "right_slide"
)

left_qpos = model.jnt_qposadr[left_id]
right_qpos = model.jnt_qposadr[right_id]


step_count = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Let the box settle before closing the fingers
        data.ctrl[0] = 0.025
        data.ctrl[1] = -0.025

        mujoco.mj_step(model, data)
        viewer.sync()

        if step_count % 500 == 0:
            print(
                f"Time: {data.time:.2f}s | "
                f"Left: {data.qpos[left_qpos]:.3f}m | "
                f"Right: {data.qpos[right_qpos]:.3f}m"
            )

        step_count += 1
        time.sleep(model.opt.timestep)
