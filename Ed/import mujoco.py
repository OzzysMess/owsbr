import mujoco
import mujoco.viewer
# python -m mujoco.viewer --mjcf=/path/to/segway_1d_wheel.xml
model = mujoco.MjModel.from_xml_path(r"C:\Users\edward\OneDrive - City University of Hong Kong\uw\me569\owsbr\Ed\model\segway_1d_wheel.xml")
data  = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()