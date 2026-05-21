import mujoco
import mujoco.viewer
import time
import os
import numpy as np
from scipy.linalg import solve_continuous_are   

folder = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(folder, "segway.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

dt = model.opt.timestep

def get_state(d):
    return np.array([d.qpos[0], d.qpos[1], d.qvel[0], d.qvel[1]])

def set_state(d, state):
    d.qpos[0] = state[0]
    d.qpos[1] = state[1]
    d.qvel[0] = state[2]
    d.qvel[1] = state[3]
    mujoco.mj_forward(model, d)

def f_discrete(state, u):
    d = mujoco.MjData(model)
    set_state(d, state)
    d.ctrl[0] = u
    mujoco.mj_step(model, d)
    return get_state(d)

# Linearize around upright equilibrium
x_eq = np.array([0.0, 0.0, 0.0, 0.0])
u_eq = 0.0
eps = 1e-5

A = np.zeros((4, 4))
B = np.zeros((4, 1))

for i in range(4):
    dx = np.zeros(4)
    dx[i] = eps
    A[:, i] = (f_discrete(x_eq + dx, u_eq) - f_discrete(x_eq - dx, u_eq)) / (2 * eps)

B[:, 0] = (f_discrete(x_eq, u_eq + eps) - f_discrete(x_eq, u_eq - eps)) / (2 * eps)

# Convert discrete linearization approximately to continuous
A_c = (A - np.eye(4)) / dt
B_c = B / dt

Q = np.diag([1, 500, 5, 50])
R = np.array([[0.1]])

P = solve_continuous_are(A_c, B_c, Q, R)
K = np.linalg.inv(R) @ B_c.T @ P

print("A_c =\n", A_c)
print("B_c =\n", B_c)
print("LQR K =", K)

# Initial condition
data.qpos[0] = 0.0
data.qpos[1] = 0.0

data.qvel[0] = 0.5
data.qvel[1] = 0.0
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.distance = 8.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90
    viewer.cam.lookat[:] = [0, 0, 0.5]

    while viewer.is_running():

        state = get_state(data)

        u = -K @ (state - x_eq)
        u = float(u[0])

        u = np.clip(u, -100, 100)
        data.ctrl[0] = u

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)