import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Ed/sac_episode_trace.csv")

plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["x"], label="x position")
plt.axhline(2.0, linestyle="--", label="goal")
plt.xlabel("Time (s)")
plt.ylabel("x (m)")
plt.grid(True)
plt.legend()
plt.savefig("sac_x_trace.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["theta_deg"], label="tilt angle")
plt.axhline(20, linestyle="--")
plt.axhline(-20, linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("theta (deg)")
plt.grid(True)
plt.legend()
plt.savefig("sac_theta_trace.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["torque"], label="torque")
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.grid(True)
plt.legend()
plt.savefig("sac_torque_trace.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved plots:")
print("sac_x_trace.png")
print("sac_theta_trace.png")
print("sac_torque_trace.png")