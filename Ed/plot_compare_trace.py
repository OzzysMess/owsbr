"""
plot_compare.py
---------------
Compares two PPO training runs — training curves + episode traces side by side.

Run:  python plot_compare.py
Output saved to Ed/ppo_comparison.png  and  Ed/ppo_traces_comparison.png
"""

import mujoco
import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── UPDATE THESE ──────────────────────────────────────────────────────────────
LOG_PATH_1      = r"Ed/SavedSeeds/PPO_cmp_seed42.json"
LOG_PATH_2  = r"C:\Users\edward\OneDrive - City University of Hong Kong\uw\me569\owsbr\Arthur\Ozzy_Segway_Data\SavedSeeds\cmp_PPO_seedcmp_seed42.json"
WEIGHTS_PATH_1  = r"Ed/SavedSeeds/nav_ppo_cmp_seed42.pth"
WEIGHTS_PATH_2  = r"Arthur/Ozzy_Segway_Data/SavedSeeds/nav_cmp_seed42.pth"   # update to model 2 weights
XML_PATH_1      = r"Ed/model/segway_1d_wheel.xml"   # model 1 xml
XML_PATH_2      = r"C:\Users\edward\OneDrive - City University of Hong Kong\uw\me569\owsbr\Ozzy\segway_2.xml"   # model 2 xml (change if different)
LABEL_1         = "Model 1"
LABEL_2         = "Model 2"
OUTPUT_DIR      = r"Ed"
TORQUE_MAX      = 5.0
GROUND_Z        = 0.075
X_NORM, XD_NORM, TH_NORM, THD_NORM = 2.4, 5.0, 1.57, 5.0
X_DONE_LIMIT    = 2.8
TH_DONE         = 0.5
X_GOAL          = 2.0
# ─────────────────────────────────────────────────────────────────────────────

COLOR_1 = "#E74C3C"   # red
COLOR_2 = "#2E86AB"   # blue


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
    if len(values) < w: return values
    return np.convolve(values, np.ones(w)/w, mode="valid")

def smooth_steps(steps, w=20):
    if len(steps) < w: return steps
    return steps[w-1:]


# ── NavPPO network ────────────────────────────────────────────────────────────
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
            torque = torch.tanh(raw) * self.torque_max
        return torque.item(), None, val


# ── data loaders ─────────────────────────────────────────────────────────────
def load_model1(path):
    with open(path) as f: d = json.load(f)
    rows = d["rows"]
    return {"steps":   [r["steps"]      for r in rows],
            "rewards": [r["avg_reward"] for r in rows],
            "losses":  [r["loss"]       for r in rows],
            "wall_time": d.get("total_wall_time", 0)}

def load_model2(path):
    with open(path) as f: d = json.load(f)
    ts = d["timeseries"]
    return {"steps":   [r["steps"]      for r in ts],
            "rewards": [r["avg_reward"] for r in ts],
            "losses":  [r["loss"]       for r in ts],
            "wall_time": d.get("total_wall_time", 0)}


# ── rollout ───────────────────────────────────────────────────────────────────
def run_episode(weights_path, xml_path):
    net = NavPPO(torque_max=TORQUE_MAX)
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()

    model = mujoco.MjModel.from_xml_path(xml_path)
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

    return (np.array(ts), np.array(xs), np.array(ths),
            np.array(torqs), result)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Training curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_training(m1, m2):
    W = 30
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28, figure=fig)
    fig.suptitle("PPO Training Comparison", fontsize=14, fontweight="bold")

    # reward vs steps
    ax = fig.add_subplot(gs[0, :])
    ax.plot(m1["steps"], m1["rewards"], color=COLOR_1, alpha=0.15, lw=0.8)
    ax.plot(m2["steps"], m2["rewards"], color=COLOR_2, alpha=0.15, lw=0.8)
    ax.plot(smooth_steps(m1["steps"], W), smooth(m1["rewards"], W),
            color=COLOR_1, lw=2.5, label=f"{LABEL_1}  (peak={max(m1['rewards']):.1f})")
    ax.plot(smooth_steps(m2["steps"], W), smooth(m2["rewards"], W),
            color=COLOR_2, lw=2.5, label=f"{LABEL_2}  (peak={max(m2['rewards']):.1f})")
    ax.axhline(0, color="green", ls="--", alpha=0.5)
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Avg100 Reward")
    ax.set_title("Training Reward", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # loss vs steps
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(m1["steps"], m1["losses"], color=COLOR_1, alpha=0.15, lw=0.8)
    ax.plot(m2["steps"], m2["losses"], color=COLOR_2, alpha=0.15, lw=0.8)
    ax.plot(smooth_steps(m1["steps"], W), smooth(m1["losses"], W),
            color=COLOR_1, lw=2.5, label=LABEL_1)
    ax.plot(smooth_steps(m2["steps"], W), smooth(m2["losses"], W),
            color=COLOR_2, lw=2.5, label=LABEL_2)
    ax.axhline(0, color="green", ls="--", alpha=0.5)
    ax.set_xlabel("Environment steps"); ax.set_ylabel("PPO Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # reward vs wall time
    ax = fig.add_subplot(gs[1, 1])
    wt1 = np.linspace(0, m1["wall_time"]/60, len(m1["steps"]))
    wt2 = np.linspace(0, m2["wall_time"]/60, len(m2["steps"]))
    ax.plot(wt1, m1["rewards"], color=COLOR_1, alpha=0.15, lw=0.8)
    ax.plot(wt2, m2["rewards"], color=COLOR_2, alpha=0.15, lw=0.8)
    ax.plot(smooth_steps(list(wt1), W), smooth(m1["rewards"], W),
            color=COLOR_1, lw=2.5, label=f"{LABEL_1}  ({m1['wall_time']/60:.1f} min)")
    ax.plot(smooth_steps(list(wt2), W), smooth(m2["rewards"], W),
            color=COLOR_2, lw=2.5, label=f"{LABEL_2}  ({m2['wall_time']/60:.1f} min)")
    ax.axhline(0, color="green", ls="--", alpha=0.5)
    ax.set_xlabel("Wall Time (min)"); ax.set_ylabel("Avg100 Reward")
    ax.set_title("Reward vs Wall Time", fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR + "/ppo_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Episode traces side by side
# ══════════════════════════════════════════════════════════════════════════════
def plot_traces_comparison(ep1, ep2):
    ts1, xs1, ths1, torqs1, res1 = ep1
    ts2, xs2, ths2, torqs2, res2 = ep2
    rms1 = np.sqrt(np.mean(torqs1**2))
    rms2 = np.sqrt(np.mean(torqs2**2))

    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.28, figure=fig)
    fig.suptitle("PPO Episode Traces Comparison", fontsize=14, fontweight="bold")

    # ── position ──────────────────────────────────────────────────────────────
    for col, (ts, xs, res, label, color) in enumerate([
            (ts1, xs1, res1, LABEL_1, COLOR_1),
            (ts2, xs2, res2, LABEL_2, COLOR_2)]):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(ts, xs, color=color, lw=2.5, label="x position")
        ax.axhline(0,      color="blue",  ls=":", lw=1.5, label="Start")
        ax.axhline(X_GOAL, color="green", ls=":", lw=1.5, label=f"Goal {X_GOAL}m")
        ax.axhspan(X_GOAL-0.25, X_GOAL+0.25, alpha=0.1, color="green")
        arrived = abs(xs[-1]-X_GOAL)<0.25 and abs(np.degrees(0))<20
        status = f"[{res.upper()}]"
        ax.set_title(f"{label} — Position  {status}", fontweight="bold")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("x (m)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── tilt ──────────────────────────────────────────────────────────────────
    for col, (ts, ths, label, color) in enumerate([
            (ts1, ths1, LABEL_1, COLOR_1),
            (ts2, ths2, LABEL_2, COLOR_2)]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(ts, ths, color=color, lw=2, label="theta (deg)")
        ax.axhspan(-20, 20, alpha=0.06, color="green", label="Stable ±20°")
        ax.axhline(0, color="gray", alpha=0.4)
        ax.set_title(f"{label} — Tilt Angle", fontweight="bold")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Tilt (°)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── torque ────────────────────────────────────────────────────────────────
    for col, (ts, torqs, rms, label, color) in enumerate([
            (ts1, torqs1, rms1, LABEL_1, COLOR_1),
            (ts2, torqs2, rms2, LABEL_2, COLOR_2)]):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(ts, torqs, color=color, lw=2, label=f"Torque (RMS={rms:.2f} Nm)")
        ax.fill_between(ts, torqs, 0, where=(torqs>0), alpha=0.15, color="red",  label="Forward")
        ax.fill_between(ts, torqs, 0, where=(torqs<0), alpha=0.15, color="blue", label="Backward")
        ax.axhline( TORQUE_MAX, color="orange", ls=":", lw=1.5, label=f"±{TORQUE_MAX} Nm")
        ax.axhline(-TORQUE_MAX, color="orange", ls=":", lw=1.5)
        ax.axhline(0, color="gray", alpha=0.4)
        ax.set_title(f"{label} — Wheel Torque", fontweight="bold")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Torque (Nm)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR + "/ppo_traces_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")

    # summary
    print(f"\n{'='*55}")
    print(f"  {'':22s}  {LABEL_1:>12}  {LABEL_2:>12}")
    print(f"  {'Duration (s)':22s}  {ts1[-1]:>12.1f}  {ts2[-1]:>12.1f}")
    print(f"  {'Max x (m)':22s}  {max(xs1):>12.3f}  {max(xs2):>12.3f}")
    print(f"  {'Final x (m)':22s}  {xs1[-1]:>12.3f}  {xs2[-1]:>12.3f}")
    print(f"  {'Avg tilt (deg)':22s}  {np.mean(np.abs(ths1)):>12.1f}  {np.mean(np.abs(ths2)):>12.1f}")
    print(f"  {'Torque RMS (Nm)':22s}  {rms1:>12.3f}  {rms2:>12.3f}")
    print(f"  {'Result':22s}  {res1:>12}  {res2:>12}")
    print(f"{'='*55}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  PPO Comparison Plotter")
    print("=" * 55)

    # load training logs
    m1 = load_model1(LOG_PATH_1)
    m2 = load_model2(LOG_PATH_2)
    print(f"\n{LABEL_1}: peak={max(m1['rewards']):.1f}  wall={m1['wall_time']/60:.1f}min")
    print(f"{LABEL_2}: peak={max(m2['rewards']):.1f}  wall={m2['wall_time']/60:.1f}min")

    print("\n[1/3] Training curves...")
    plot_training(m1, m2)

    print("\n[2/3] Running episode for Model 1...")
    try:
        ep1 = run_episode(WEIGHTS_PATH_1, XML_PATH_1)
        print(f"  Result: {ep1[4]}  duration={ep1[0][-1]:.1f}s  max_x={max(ep1[1]):.3f}m")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        ep1 = None

    print("\n[3/3] Running episode for Model 2...")
    try:
        ep2 = run_episode(WEIGHTS_PATH_2, XML_PATH_2)
        print(f"  Result: {ep2[4]}  duration={ep2[0][-1]:.1f}s  max_x={max(ep2[1]):.3f}m")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        ep2 = None

    if ep1 and ep2:
        print("\nPlotting episode traces comparison...")
        plot_traces_comparison(ep1, ep2)
    else:
        print("\nSkipping traces comparison (one or both weight files missing).")
        print("Update WEIGHTS_PATH_1 and WEIGHTS_PATH_2 at the top of the file.")

    print("\nAll done.")
