import mujoco

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 0.5">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.02 0.02 0.03" mass="0.3"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

for step in range(1000):
    mujoco.mj_step(model, data)
    if step % 100 == 0:
        print(f"t={data.time:.2f}s  height z={data.qpos[2]:.4f} m")

print("final height:", round(float(data.qpos[2]), 4), "m (box half-height is 0.03)")
