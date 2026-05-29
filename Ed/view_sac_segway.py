import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import SAC

folder = os.path.dirname(os.path.abspath(__file__))

xml_path = os.path.join(folder, "segway_ed.xml")
model_path = os.path.join(folder, "sac_segway.zip")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

sac_model = SAC.load(model_path)

dt = model.opt.timestep
U_LIMIT = 100.0

def get_obs():
    return np.array([
        data.qpos[0],
        data.qpos[1],
        data.qvel[0],
        data.qvel[1],
    ], dtype=np.float32)

# initial condition
data.qpos[0] = 0.0
data.qpos[1] = 0.10
data.qvel[0] = 0.0
data.qvel[1] = 0.0
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 4.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 80
    viewer.cam.lookat[:] = [0.0, 0.0, 0.5]

    print("SAC viewer opened. Click the MuJoCo window if it is behind VS Code.")
    print("Close the viewer window to stop.")

    while viewer.is_running():
        obs = get_obs()

        action, _ = sac_model.predict(obs, deterministic=True)
        u = float(action[0])
        u = np.clip(u, -U_LIMIT, U_LIMIT)

        data.ctrl[0] = u

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)