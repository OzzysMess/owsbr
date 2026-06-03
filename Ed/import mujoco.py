import mujoco
import mujoco.viewer
# python -m mujoco.viewer --mjcf=/path/to/segway_1d_wheel.xml
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()