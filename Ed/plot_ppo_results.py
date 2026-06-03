"""
plot_ppo_results.py
-------------------
Run from terminal:   python plot_ppo_results.py
Or from notebook:    %run plot_ppo_results.py

Produces three files:
  ppo_training.png       — reward + loss vs steps
  ppo_traces.png         — episode traces (position, tilt, torque)
  ppo_traces_live.png    — same but runs a LIVE rollout from saved weights
"""

import mujoco
import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── UPDATE THESE ──────────────────────────────────────────────────────────────
XML_PATH     = r"Ed/segway_1d_wheel.xml"
LOG_PATH     = r"Ed/SavedSeeds/PPO_cmp_seed42.json"
WEIGHTS_PATH = r"Ed/SavedSeeds/nav_ppo_cmp_seed42.pth"

TORQUE_MAX   = 5.0
GROUND_Z     = 0.075
X_NORM, XD_NORM, TH_NORM, THD_NORM = 2.4, 5.0, 1.57, 5.0
X_DONE_LIMIT = 2.8
TH_DONE      = 0.5
X_GOAL       = 2.0
# ─────────────────────────────────────────────────────────────────────────────


# ── helpers (must match notebook exactly) ────────────────────────────────────
def get_obs(data):
    qw,qx,qy,qz = data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]
    theta = float(np.arcsin(np.clip(2.0*(qw*qy - qz*qx), -1.0, 1.0)))
    return np.array([data.qpos[0], data.qvel[0], theta, data.qvel[4]], dtype=np.float32)

def normalize_obs(obs):
    x,xd,th,thd = obs
    return np.array([
        np.clip(x  /X_NORM,  -1,1),
        np.clip(xd /XD_NORM, -1,1),
        np.clip(th /TH_NORM, -1,1),
        np.clip(thd/THD_NORM,-1,1),
    ], dtype=np.float32)


# ── NavPPO (must match notebook exactly) ─────────────────────────────────────
class NavPPO(nn.Module):
    def __init__(self, torque_max=TORQUE_MAX):
        super().__init__(); self.torque_max = torque_max
        self.net = nn.Sequential(
            nn.Linear(5,256), nn.Tanh(),
            nn.Linear(256,256), nn.Tanh(),
            nn.Linear(256,128), nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(128,1)
        self.actor_log_std = nn.Parameter(torch.tensor([-0.5]))
        self.critic        = nn.Linear(128,1)

    def forward(self, x):
        s = self.net(x); return self.actor_mean(s), self.critic(s)

    def get_action(self, obs, goal_x, deterministic=True):
        dist_norm = float(np.clip((goal_x - obs[0]) / 5., -1, 1))
        inp = torch.FloatTensor([*normalize_obs(obs), dist_norm])
        with torch.no_grad():
            mean, val = self(inp)
            std  = self.actor_log_std.exp().clamp(0.1, 2.0)
            dist = torch.distributions.Normal(mean, std)
            raw  = mean if deterministic else dist.rsample()
            logp = dist.log_prob(raw).sum(-1)
            torque = torch.tanh(raw) * self.torque_max
        return torque.item(), logp, val


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TRAINING CURVES  (from JSON log)
# ══════════════════════════════════════════════════════════════════════════════
def plot_training():
    with open(LOG_PATH) as f:
        data = json.load(f)

    rows      = data["rows"]
    steps     = [r["steps"]       for r in rows]
    rewards   = [r["avg_reward"]  for r in rows]
    losses    = [r["loss"]        for r in rows]
    wall_time = data.get("total_wall_time", 0)

    print(f"Log: {len(rows)} points  |  wall time: {wall_time/60:.1f} min")
    print(f"Peak avg reward: {max(rewards):.2f}   Final: {rewards[-1]:.2f}")

    fig, (axr, axl) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PPO Training  ({wall_time/60:.1f} min)", fontsize=13, fontweight="bold")

    axr.plot(steps, rewards, color="#E74C3C", lw=2.5, label=f"avg100 (peak={max(rewards):.1f})")
    axr.axhline(0, color="green", ls="--", alpha=0.5)
    axr.set_xlabel("Environment steps"); axr.set_ylabel("Reward")
    axr.set_title("Training Reward", fontweight="bold")
    axr.legend(); axr.grid(alpha=0.3)

    axl.plot(steps, losses, color="#9B59B6", lw=2, label="PPO Loss")
    axl.axhline(0, color="green", ls="--", alpha=0.5)
    axl.set_xlabel("Environment steps"); axl.set_ylabel("PPO Loss")
    axl.set_title("Training Loss", fontweight="bold")
    axl.legend(); axl.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("ppo_training.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: ppo_training.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EPISODE TRACES  (live rollout from saved weights)
# ══════════════════════════════════════════════════════════════════════════════
def plot_traces():
    # load model
    net = NavPPO(torque_max=TORQUE_MAX)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    net.eval()
    print(f"Loaded weights: {WEIGHTS_PATH}")

    # run one deterministic episode
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.0; data.qpos[2] = GROUND_Z
    data.qpos[3:7] = [1,0,0,0]; data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    obs = get_obs(data)

    ts, xs, ths, torqs = [], [], [], []
    for step in range(3000):
        torque, _, _ = net.get_action(obs, X_GOAL, deterministic=True)
        u = float(np.clip(torque, -TORQUE_MAX, TORQUE_MAX))
        data.ctrl[0] = -u
        mujoco.mj_step(model, data); obs = get_obs(data)
        x, _, th, _ = obs
        ts.append(step * dt); xs.append(x)
        ths.append(np.degrees(th)); torqs.append(torque)
        if abs(x) > X_DONE_LIMIT or abs(th) > TH_DONE:
            print(f"  Fell at t={step*dt:.1f}s  x={x:.3f}m  theta={np.degrees(th):.1f}deg")
            break
        if abs(x - X_GOAL) < 0.25 and abs(th) < 0.35:
            print(f"  Arrived at t={step*dt:.1f}s  x={x:.3f}m")
            break

    ts    = np.array(ts);    xs    = np.array(xs)
    ths   = np.array(ths);   torqs = np.array(torqs)
    arrived = abs(xs[-1] - X_GOAL) < 0.25 and abs(ths[-1]) < 20
    rms   = np.sqrt(np.mean(torqs**2))

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.25, figure=fig)
    fig.suptitle("PPO — Deterministic Episode", fontsize=13, fontweight="bold")

    # position (top row, full width)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(ts, xs, "#E74C3C", lw=2.5, label="x position")
    ax.axhline(0,      color="blue",  ls=":", lw=2, label="A (start)")
    ax.axhline(X_GOAL, color="green", ls=":", lw=2, label=f"B (goal={X_GOAL}m)")
    ax.axhspan(X_GOAL-0.25, X_GOAL+0.25, alpha=0.1, color="green", label="Goal zone ±0.25m")
    if arrived:
        idx = np.where(np.abs(xs - X_GOAL) < 0.25)[0][0]
        ax.axvline(ts[idx], color="green", ls="--", alpha=0.6)
        ax.annotate(f"ARRIVED\nt={ts[idx]:.1f}s",
                    xy=(ts[idx], xs[idx]), xytext=(ts[idx]+0.3, X_GOAL-0.4),
                    color="green", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="green"))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position x (m)")
    ax.set_title("Position: A → B", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # tilt (bottom-left)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ts, ths, "#E74C3C", lw=2, label="theta (deg)")
    ax.axhspan(-20, 20, alpha=0.06, color="green", label="Stable ±20°")
    ax.axhline(0, color="gray", alpha=0.4)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Tilt Angle (°)")
    ax.set_title("Tilt Angle θ", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # torque (bottom-right)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ts, torqs, "#2ECC71", lw=2, label=f"Torque (RMS={rms:.2f} Nm)")
    ax.fill_between(ts, torqs, 0, where=(torqs>0), alpha=0.15, color="red",  label="Forward")
    ax.fill_between(ts, torqs, 0, where=(torqs<0), alpha=0.15, color="blue", label="Backward")
    ax.axhline( TORQUE_MAX, color="orange", ls=":", lw=1.5, label=f"±{TORQUE_MAX} Nm")
    ax.axhline(-TORQUE_MAX, color="orange", ls=":", lw=1.5)
    ax.axhline(0, color="gray", alpha=0.4)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Torque (Nm)")
    ax.set_title(f"Wheel Torque", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.savefig("ppo_traces.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: ppo_traces.png")
    print(f"  Duration {ts[-1]:.1f}s | Max x {max(xs):.3f}m | "
          f"Avg tilt {np.mean(np.abs(ths)):.1f}° | RMS torque {rms:.3f} Nm")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  PPO Results Plotter")
    print("=" * 50)

    print("\n[1/2] Training curves...")
    plot_training()

    print("\n[2/2] Episode traces...")
    plot_traces()

    print("\nAll done.")
