import mujoco
import mujoco.viewer
import time

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model = mujoco.MjModel.from_xml_path(os.path.join(script_dir, "segway_2.xml"))
data = mujoco.MjData(model)

# qpos represents the generalized positions:
#   qpos[0] = x: horizontal position of the Segway along the ground
#   qpos[1] = theta: angle of the Segway from vertical (0 = upright, >0 = leaning forward)
#
# qvel represents the generalized velocities (time derivatives of qpos):
#   qvel[0] = x_dot: horizontal velocity
#   qvel[1] = theta_dot: angular velocity (rate of angle change)

# Initial conditions: slightly tilted forward to test the controller
data.qpos[0] = 0.0      # Start at origin
data.qpos[1] = 1.1      # Start tilted 0.1 rad forward

data.qvel[0] = 0.0      # No initial horizontal velocity
data.qvel[1] = 0.0      # No initial angular velocity

mujoco.mj_forward(model, data)

# PID Controller Parameters for keeping Segway upright
Kp = 15.0              # Proportional gain: strong response to angle error
Kd = 3.0               # Derivative gain: damping to prevent oscillation
Ki = .50                # Integral gain: correction for steady-state error

integral_error = 0.0    # Accumulate angle error over time
dt = model.opt.timestep # Simulation timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Current state
        theta = data.qpos[1]           # Current angle from vertical
        theta_dot = data.qvel[1]       # Current angular velocity
        
        # Error: how much the Segway has tilted from upright (0 rad)
        angle_error = theta
        
        # Accumulate integral error
        integral_error += angle_error * dt
        
        # PID control law: u = Kp*e + Ki*∫e + Kd*de/dt
        # This produces a force that:
        # - Increases with angle error (proportional term)
        # - Increases with accumulated error (integral term)
        # - Increases with angular velocity (derivative term - damping)
        u = Kp * angle_error + Ki * integral_error + Kd * theta_dot
        
        # Saturate force to motor limits
        u = max(min(u, .1), -.1)
        
        data.ctrl[0] = u
        
        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)