"""
plot_compare.py
---------------
Compares two PPO training runs on the same plot.

  Model 1: PPO_cmp_seed42.json          (uses "rows" key)
  Model 2: cmp_PPO_seedcmp_seed42.json  (uses "timeseries" key)

Run:  python plot_compare.py
Output saved to Ed/ppo_comparison.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── UPDATE THESE ──────────────────────────────────────────────────────────────
LOG_PATH_1  = r"Ed/SavedSeeds/PPO_cmp_seed42.json"
LOG_PATH_2  = r"C:\Users\edward\OneDrive - City University of Hong Kong\uw\me569\owsbr\Arthur\Ozzy_Segway_Data\SavedSeeds\cmp_PPO_seedcmp_seed42.json"
LABEL_1     = "Model 1"
LABEL_2     = "Model 2"
OUTPUT_DIR  = r"Ed"
# ─────────────────────────────────────────────────────────────────────────────

COLOR_1 = "#E74C3C"   # red
COLOR_2 = "#1D1DA7"   # blue


def load_model1(path):
    """PPO_cmp_seed42.json — uses 'rows' key."""
    with open(path) as f:
        d = json.load(f)
    rows = d["rows"]
    return {
        "steps":     [r["steps"]       for r in rows],
        "rewards":   [r["avg_reward"]  for r in rows],
        "losses":    [r["loss"]        for r in rows],
        "wall_time": d.get("total_wall_time", 0),
    }


def load_model2(path):
    """cmp_PPO_seedcmp_seed42.json — uses 'timeseries' key."""
    with open(path) as f:
        d = json.load(f)
    ts = d["timeseries"]
    return {
        "steps":     [r["steps"]       for r in ts],
        "rewards":   [r["avg_reward"]  for r in ts],
        "losses":    [r["loss"]        for r in ts],
        "wall_time": d.get("total_wall_time", 0),
    }


def smooth(values, w=20):
    """Simple moving average for cleaner curves."""
    if len(values) < w:
        return values
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode="valid")


def smooth_steps(steps, w=20):
    """Trim steps to match smoothed length."""
    if len(steps) < w:
        return steps
    return steps[w-1:]


# ── load ──────────────────────────────────────────────────────────────────────
m1 = load_model1(LOG_PATH_1)
m2 = load_model2(LOG_PATH_2)

print(f"{LABEL_1}: {len(m1['steps'])} points | "
      f"wall={m1['wall_time']/60:.1f}min | "
      f"peak reward={max(m1['rewards']):.1f} | "
      f"final={m1['rewards'][-1]:.1f}")
print(f"{LABEL_2}: {len(m2['steps'])} points | "
      f"wall={m2['wall_time']/60:.1f}min | "
      f"peak reward={max(m2['rewards']):.1f} | "
      f"final={m2['rewards'][-1]:.1f}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28, figure=fig)
fig.suptitle("PPO Training Comparison — Model 1 vs Model 2",
             fontsize=14, fontweight="bold")

W = 30   # smoothing window

# ── reward (top row, full width) ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
# raw (faint)
ax.plot(m1["steps"], m1["rewards"], color=COLOR_1, alpha=0.15, lw=0.8)
ax.plot(m2["steps"], m2["rewards"], color=COLOR_2, alpha=0.15, lw=0.8)
# smoothed
ax.plot(smooth_steps(m1["steps"], W), smooth(m1["rewards"], W),
        color=COLOR_1, lw=2.5, label=f"{LABEL_1}  (peak={max(m1['rewards']):.1f})")
ax.plot(smooth_steps(m2["steps"], W), smooth(m2["rewards"], W),
        color=COLOR_2, lw=2.5, label=f"{LABEL_2}  (peak={max(m2['rewards']):.1f})")
ax.axhline(0, color="green", ls="--", alpha=0.5, label="Zero reward")
ax.set_xlabel("Environment steps"); ax.set_ylabel("Avg100 Reward")
ax.set_title("Training Reward", fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)

# ── loss (bottom-left) ────────────────────────────────────────────────────────
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

# ── reward vs wall time (bottom-right) ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
# derive wall time per log point
wt1 = np.linspace(0, m1["wall_time"]/60, len(m1["steps"]))
wt2 = np.linspace(0, m2["wall_time"]/60, len(m2["steps"]))
ax.plot(wt1, m1["rewards"], color=COLOR_1, alpha=0.15, lw=0.8)
ax.plot(wt2, m2["rewards"], color=COLOR_2, alpha=0.15, lw=0.8)
ax.plot(smooth_steps(list(wt1), W), smooth(m1["rewards"], W),
        color=COLOR_1, lw=2.5,
        label=f"{LABEL_1}  ({m1['wall_time']/60:.1f} min)")
ax.plot(smooth_steps(list(wt2), W), smooth(m2["rewards"], W),
        color=COLOR_2, lw=2.5,
        label=f"{LABEL_2}  ({m2['wall_time']/60:.1f} min)")
ax.axhline(0, color="green", ls="--", alpha=0.5)
ax.set_xlabel("Wall Time (min)"); ax.set_ylabel("Avg100 Reward")
ax.set_title("Reward vs Wall Time", fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)

plt.tight_layout()
out = OUTPUT_DIR + "/ppo_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {out}")

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Summary")
print(f"{'='*50}")
print(f"  {'':20s}  {LABEL_1:>12}  {LABEL_2:>12}")
print(f"  {'Wall time (min)':20s}  {m1['wall_time']/60:>12.1f}  {m2['wall_time']/60:>12.1f}")
print(f"  {'Total steps':20s}  {max(m1['steps']):>12,}  {max(m2['steps']):>12,}")
print(f"  {'Peak reward':20s}  {max(m1['rewards']):>12.1f}  {max(m2['rewards']):>12.1f}")
print(f"  {'Final reward':20s}  {m1['rewards'][-1]:>12.1f}  {m2['rewards'][-1]:>12.1f}")
print(f"{'='*50}")
