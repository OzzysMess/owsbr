import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("segway.xml")
data = mujoco.MjData(model)

data.qpos[0] = 0.0
data.qpos[1] = 0.01

data.qvel[0] = 0.0
data.qvel[1] = 0.0

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        x = data.qpos[0]
        theta = data.qpos[1]

        x_dot = data.qvel[0]
        theta_dot = data.qvel[1]

        # Simple inverted pendulum controller
        u = 80*theta + 15*theta_dot - 2*x - 4*x_dot

        # Saturate force
        u = max(min(u, 100), -100)

        data.ctrl[0] = u

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)