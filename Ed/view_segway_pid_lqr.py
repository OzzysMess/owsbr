"""
Live MuJoCo viewer for PID or LQR control of segway_ed.xml.

Use:
    python view_segway_pid_lqr.py --controller PID
    python view_segway_pid_lqr.py --controller LQR

Put this file in the same folder as segway_ed.xml.
"""

import argparse
import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_continuous_are

DEFAULT_CONTROLLER = "PID"

INITIAL_STATE = np.array([0.0, 0.10, 0.0, 0.0])  # [x, theta, x_dot, theta_dot]
U_LIMIT = 100.0
CONTROL_SIGN = 1.0  # change to -1.0 if the model falls faster

Kp = 120.0
Ki = 0.0
Kd = 20.0
theta_des = 0.0


def find_xml_file():
    folder = os.path.dirname(os.path.abspath(__file__))
    choices = [
        os.path.join(folder, "segway_ed.xml"),
        os.path.join(folder, "segway_ed(1).xml"),
        os.path.join(os.getcwd(), "segway_ed.xml"),
        os.path.join(os.getcwd(), "segway_ed(1).xml"),
    ]
    for path in choices:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Put segway_ed.xml in the same folder as this Python file.")


def get_state(d):
    return np.array([d.qpos[0], d.qpos[1], d.qvel[0], d.qvel[1]], dtype=float)


def set_state(model, d, state):
    d.qpos[0] = state[0]
    d.qpos[1] = state[1]
    d.qvel[0] = state[2]
    d.qvel[1] = state[3]
    mujoco.mj_forward(model, d)


def compute_lqr_gain(model, dt):
    def f_discrete(state, u):
        d = mujoco.MjData(model)
        set_state(model, d, state)
        d.ctrl[0] = float(u)
        mujoco.mj_step(model, d)
        return get_state(d)

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

    A_c = (A - np.eye(4)) / dt
    B_c = B / dt

    Q = np.diag([1.0, 500.0, 5.0, 50.0])
    R = np.array([[0.1]])

    P = solve_continuous_are(A_c, B_c, Q, R)
    K_lqr = np.linalg.inv(R) @ B_c.T @ P

    return K_lqr, x_eq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["PID", "LQR"], default=DEFAULT_CONTROLLER)
    args = parser.parse_args()
    controller = args.controller

    xml_path = find_xml_file()
    print("Using XML:", xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    print("Model loaded.")
    print("dt =", dt, "nq =", model.nq, "nv =", model.nv, "nu =", model.nu)
    print("Controller:", controller)

    K_lqr, x_eq = compute_lqr_gain(model, dt)
    print("LQR K =", K_lqr)

    set_state(model, data, INITIAL_STATE)

    integral = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 80
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]

        print("Viewer opened. Click the viewer window if it is behind VS Code/Jupyter.")
        print("Close the viewer window to stop.")

        while viewer.is_running():
            if controller == "PID":
                theta = data.qpos[1]
                theta_dot = data.qvel[1]

                error = theta_des - theta
                integral += error * dt
                derivative = -theta_dot

                u = Kp * error + Ki * integral + Kd * derivative

            elif controller == "LQR":
                state = get_state(data)
                u = -K_lqr @ (state - x_eq)
                u = float(u[0])

            u = CONTROL_SIGN * u
            u = np.clip(u, -U_LIMIT, U_LIMIT)

            # This is the important line: apply control to the model every step.
            data.ctrl[0] = u

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
