"""
plot_sac_vs_ppo.py
------------------
Compares SAC vs PPO for the same model (Model 1).

Run:  python plot_sac_vs_ppo.py
Output saved to:
  Ed/sac_vs_ppo_training.png
  Ed/sac_vs_ppo_traces.png
"""

import mujoco
import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── UPDATE THESE ──────────────────────────────────────────────────────────────
SAC_LOG_PATH     = r"C:/Users/edward/OneDrive - City University of Hong Kong/uw/me569/owsbr/Ed/SavedSeeds/SAC_cmp_seed42.json"
PPO_LOG_PATH     = r"C:/Users/edward/OneDrive - City University of Hong Kong/uw/me569/owsbr/Ed/SavedSeeds/PPO_cmp_seed42.json"
SAC_WEIGHTS_PATH = r"C:/Users/edward/OneDrive - City University of Hong Kong/uw/me569/owsbr/Ed/SavedSeeds/nav_sac_cmp_seed42.pth"
PPO_WEIGHTS_PATH = r"C:/Users/edward/OneDrive - City University of Hong Kong/uw/me569/owsbr/Ed/SavedSeeds/nav_ppo_cmp_seed42.pth"
XML_PATH = r"C:/Users/edward/OneDrive - City University of Hong Kong/uw/me569/owsbr/Ed/model/segway_1d_wheel.xml"
OUTPUT_DIR       = r"C:/Users/edward/OneDrive - City University of Hong Kong/uw/me569/owsbr/Ed"
TORQUE_MAX       = 5.0
GROUND_Z         = 0.075
X_NORM, XD_NORM, TH_NORM, THD_NORM = 2.4, 5.0, 1.57, 5.0
X_DONE_LIMIT     = 2.8
TH_DONE          = 0.5
X_GOAL           = 2.0
# ─────────────────────────────────────────────────────────────────────────────

COLOR_SAC = "#E74C3C"   # red
COLOR_PPO = "#2E86AB"   # blue


# ── helpers ───────────────────────────────────────────────────────────────────
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

def smooth(values, w=20):
    if len(values) < w: return np.array(values)
    return np.convolve(values, np.ones(w)/w, mode="valid")

def smooth_steps(steps, w=20):
    if len(steps) < w: return steps
    return steps[w-1:]

def load_log(path):
    with open(path) as f: d = json.load(f)
    rows = d.get("rows", d.get("timeseries", []))
    return {"steps":     [r["steps"]      for r in rows],
            "rewards":   [r["avg_reward"] for r in rows],
            "losses":    [r["loss"]       for r in rows],
            "wall_time": d.get("total_wall_time", 0)}


# ── SAC actor network ─────────────────────────────────────────────────────────
class NavActor(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX =  2

    def __init__(self, torque_max=TORQUE_MAX):
        super().__init__(); self.torque_max = torque_max
        self.net = nn.Sequential(
            nn.Linear(5,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(256,1)
        self.log_std_layer = nn.Linear(256,1)

    def forward(self, x):
        h = self.net(x)
        mean    = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std.exp()

    def get_action(self, obs, goal_x, deterministic=True):
        dist_norm = float(np.clip((goal_x - obs[0]) / 5., -1, 1))
        inp = torch.FloatTensor([*normalize_obs(obs), dist_norm]).unsqueeze(0)
        with torch.no_grad():
            mean, std = self(inp)
            raw = mean if deterministic else torch.distributions.Normal(mean, std).rsample()
            torque = torch.tanh(raw) * self.torque_max
        return torque.item(), None, None


# ── PPO actor-critic network ──────────────────────────────────────────────────
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
            mean, _ = self(inp)
            std  = self.actor_log_std.exp().clamp(0.1, 2.0)
            raw  = mean if deterministic else torch.distributions.Normal(mean, std).rsample()
            torque = torch.tanh(raw) * self.torque_max
        return torque.item(), None, None


# ── run one deterministic episode ─────────────────────────────────────────────
def run_episode(net):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    mujoco.mj_resetData(model, data)
    data.qpos[0]=0.0; data.qpos[2]=GROUND_Z
    data.qpos[3:7]=[1,0,0,0]; data.qvel[:]=0.0
    mujoco.mj_forward(model, data)
    obs = get_obs(data)
    ts, xs, ths, torqs = [], [], [], []
    result = "timeout"
    for step in range(3000):
        torque, _, _ = net.get_action(obs, X_GOAL, deterministic=True)
        u = float(np.clip(torque, -TORQUE_MAX, TORQUE_MAX))
        data.ctrl[0] = -u
        mujoco.mj_step(model, data); obs = get_obs(data)
        x, _, th, _ = obs
        ts.append(step*dt); xs.append(x)
        ths.append(np.degrees(th)); torqs.append(torque)
        if abs(x) > X_DONE_LIMIT or abs(th) > TH_DONE:
            result = "fell"; break
        if abs(x - X_GOAL) < 0.25 and abs(th) < 0.35:
            result = "arrived"; break
    return np.array(ts), np.array(xs), np.array(ths), np.array(torqs), result


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Training curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_training(sac, ppo):
    W = 30
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28, figure=fig)
    fig.suptitle("SAC vs PPO — Training Comparison (Model 1)",
                 fontsize=14, fontweight="bold")

    # reward vs steps
    ax = fig.add_subplot(gs[0, :])
    ax.plot(sac["steps"], sac["rewards"], color=COLOR_SAC, alpha=0.15, lw=0.8)
    ax.plot(ppo["steps"], ppo["rewards"], color=COLOR_PPO, alpha=0.15, lw=0.8)
    ax.plot(smooth_steps(sac["steps"], W), smooth(sac["rewards"], W),
            color=COLOR_SAC, lw=2.5, label=f"SAC  (peak={max(sac['rewards']):.1f})")
    ax.plot(smooth_steps(ppo["steps"], W), smooth(ppo["rewards"], W),
            color=COLOR_PPO, lw=2.5, label=f"PPO  (peak={max(ppo['rewards']):.1f})")
    ax.axhline(0, color="green", ls="--", alpha=0.5)
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Avg100 Reward")
    ax.set_title("Training Reward", fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)

    # loss vs steps
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(sac["steps"], sac["losses"], color=COLOR_SAC, alpha=0.15, lw=0.8)
    ax.plot(ppo["steps"], ppo["losses"], color=COLOR_PPO, alpha=0.15, lw=0.8)
    ax.plot(smooth_steps(sac["steps"], W), smooth(sac["losses"], W),
            color=COLOR_SAC, lw=2.5, label="SAC Critic Loss")
    ax.plot(smooth_steps(ppo["steps"], W), smooth(ppo["losses"], W),
            color=COLOR_PPO, lw=2.5, label="PPO Loss")
    ax.axhline(0, color="green", ls="--", alpha=0.5)
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)

    # reward vs wall time
    ax = fig.add_subplot(gs[1, 1])
    wt_sac = np.linspace(0, sac["wall_time"]/60, len(sac["steps"]))
    wt_ppo = np.linspace(0, ppo["wall_time"]/60, len(ppo["steps"]))
    ax.plot(wt_sac, sac["rewards"], color=COLOR_SAC, alpha=0.15, lw=0.8)
    ax.plot(wt_ppo, ppo["rewards"], color=COLOR_PPO, alpha=0.15, lw=0.8)
    ax.plot(smooth_steps(list(wt_sac), W), smooth(sac["rewards"], W),
            color=COLOR_SAC, lw=2.5, label=f"SAC  ({sac['wall_time']/60:.1f} min)")
    ax.plot(smooth_steps(list(wt_ppo), W), smooth(ppo["rewards"], W),
            color=COLOR_PPO, lw=2.5, label=f"PPO  ({ppo['wall_time']/60:.1f} min)")
    ax.axhline(0, color="green", ls="--", alpha=0.5)
    ax.set_xlabel("Wall Time (min)"); ax.set_ylabel("Avg100 Reward")
    ax.set_title("Reward vs Wall Time", fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR + "/sac_vs_ppo_training.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Episode traces
# ══════════════════════════════════════════════════════════════════════════════
def plot_traces(sac_ep, ppo_ep):
    ts_s, xs_s, ths_s, torqs_s, res_s = sac_ep
    ts_p, xs_p, ths_p, torqs_p, res_p = ppo_ep
    rms_s = np.sqrt(np.mean(torqs_s**2))
    rms_p = np.sqrt(np.mean(torqs_p**2))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle("SAC vs PPO — Episode Traces (Model 1)",
                 fontsize=14, fontweight="bold")

    # position
    ax = axes[0]
    ax.plot(ts_s, xs_s, color=COLOR_SAC, lw=2.5, label=f"SAC  [{res_s}]")
    ax.plot(ts_p, xs_p, color=COLOR_PPO, lw=2.5, label=f"PPO  [{res_p}]")
    ax.axhline(0,      color="blue",  ls=":", lw=1.5, label="Start")
    ax.axhline(X_GOAL, color="green", ls=":", lw=1.5, label=f"Goal {X_GOAL}m")
    ax.axhspan(X_GOAL-0.25, X_GOAL+0.25, alpha=0.1, color="green", label="Goal zone ±0.25m")
    ax.set_ylabel("Position x (m)")
    ax.set_title("Position: A → B", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # tilt
    ax = axes[1]
    ax.plot(ts_s, ths_s, color=COLOR_SAC, lw=2, label="SAC")
    ax.plot(ts_p, ths_p, color=COLOR_PPO, lw=2, label="PPO")
    ax.axhspan(-20, 20, alpha=0.06, color="green", label="Stable ±20°")
    ax.axhline(0, color="gray", alpha=0.4)
    ax.set_ylabel("Tilt Angle (°)")
    ax.set_title("Tilt Angle θ", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # torque
    ax = axes[2]
    ax.plot(ts_s, torqs_s, color=COLOR_SAC, lw=2, label=f"SAC  (RMS={rms_s:.2f} Nm)")
    ax.plot(ts_p, torqs_p, color=COLOR_PPO, lw=2, label=f"PPO  (RMS={rms_p:.2f} Nm)")
    ax.axhline( TORQUE_MAX, color="orange", ls=":", lw=1.5, label=f"±{TORQUE_MAX} Nm")
    ax.axhline(-TORQUE_MAX, color="orange", ls=":", lw=1.5)
    ax.axhline(0, color="gray", alpha=0.4)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Torque (Nm)")
    ax.set_title("Wheel Torque", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR + "/sac_vs_ppo_traces.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")

    print(f"\n{'='*55}")
    print(f"  {'':22s}  {'SAC':>12}  {'PPO':>12}")
    print(f"  {'Duration (s)':22s}  {ts_s[-1]:>12.1f}  {ts_p[-1]:>12.1f}")
    print(f"  {'Max x (m)':22s}  {max(xs_s):>12.3f}  {max(xs_p):>12.3f}")
    print(f"  {'Final x (m)':22s}  {xs_s[-1]:>12.3f}  {xs_p[-1]:>12.3f}")
    print(f"  {'Avg tilt (deg)':22s}  {np.mean(np.abs(ths_s)):>12.1f}  {np.mean(np.abs(ths_p)):>12.1f}")
    print(f"  {'Torque RMS (Nm)':22s}  {rms_s:>12.3f}  {rms_p:>12.3f}")
    print(f"  {'Result':22s}  {res_s:>12}  {res_p:>12}")
    print(f"{'='*55}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  SAC vs PPO Comparison Plotter")
    print("=" * 55)

    # load logs
    sac_log = load_log(SAC_LOG_PATH)
    ppo_log = load_log(PPO_LOG_PATH)
    print(f"\nSAC: peak={max(sac_log['rewards']):.1f}  wall={sac_log['wall_time']/60:.1f}min")
    print(f"PPO: peak={max(ppo_log['rewards']):.1f}  wall={ppo_log['wall_time']/60:.1f}min")

    print("\n[1/3] Training curves...")
    plot_training(sac_log, ppo_log)

    print("\n[2/3] Running SAC episode...")
    try:
        sac_net = NavActor(torque_max=TORQUE_MAX)
        sac_net.load_state_dict(torch.load(SAC_WEIGHTS_PATH, map_location="cpu"))
        sac_net.eval()
        sac_ep = run_episode(sac_net)
        print(f"  Result: {sac_ep[4]}  duration={sac_ep[0][-1]:.1f}s  max_x={max(sac_ep[1]):.3f}m")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        sac_ep = None

    print("\n[3/3] Running PPO episode...")
    try:
        ppo_net = NavPPO(torque_max=TORQUE_MAX)
        ppo_net.load_state_dict(torch.load(PPO_WEIGHTS_PATH, map_location="cpu"))
        ppo_net.eval()
        ppo_ep = run_episode(ppo_net)
        print(f"  Result: {ppo_ep[4]}  duration={ppo_ep[0][-1]:.1f}s  max_x={max(ppo_ep[1]):.3f}m")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        ppo_ep = None

    print(f"\nsac_ep = {sac_ep}")
    print(f"ppo_ep = {ppo_ep}")

    if sac_ep and ppo_ep:
        print("\nPlotting episode traces...")
        plot_traces(sac_ep, ppo_ep)
    else:
        print("\nSkipping traces — check SAC_WEIGHTS_PATH and PPO_WEIGHTS_PATH.")
