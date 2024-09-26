# test_mujoco.py
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_string("""
<mujoco>
    <worldbody>
        <light name="light" pos="0 0 2"/>
        <geom name="ground" type="plane" pos="0 0 0" size="5 5 0.1"/>
    </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)

# Create the viewer
viewer = mujoco.viewer.launch_passive(model, data)

# Keep the script running until the viewer window is closed
try:
    while viewer.is_running():
        pass
except KeyboardInterrupt:
    pass