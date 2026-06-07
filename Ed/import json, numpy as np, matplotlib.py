import json, numpy as np, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

LOG_PATH = "Ed/SavedSeeds/SAC_cmp_seed42.json"   # update if needed

# ── load ──────────────────────────────────────────────────────────────────────
with open(LOG_PATH) as f:
    data = json.load(f)

rows       = data["rows"]
steps      = [r["steps"]       for r in rows]
avg_reward = [r["avg_reward"]  for r in rows]
loss       = [r["loss"]        for r in rows]
wall_time  = data.get("total_wall_time", 0)

print(f"Loaded: {len(rows)} log points  |  wall time: {wall_time/60:.1f} min")
print(f"Peak avg reward: {max(avg_reward):.2f}  |  Final: {avg_reward[-1]:.2f}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, (axr, axl) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"SAC Training  ({wall_time/60:.1f} min)", fontsize=13, fontweight="bold")

axr.plot(steps, avg_reward, color="#E74C3C", lw=2.5, label=f"avg100 reward")
axr.axhline(0, color="green", ls="--", alpha=0.5)
axr.set_xlabel("Environment steps"); axr.set_ylabel("Reward")
axr.set_title("Training Reward", fontweight="bold")
axr.legend(); axr.grid(alpha=0.3)

axl.plot(steps, loss, color="#9B59B6", lw=2, label="Critic Loss")
axl.axhline(0, color="green", ls="--", alpha=0.5)
axl.set_xlabel("Environment steps"); axl.set_ylabel("Critic Loss")
axl.set_title("Training Loss", fontweight="bold")
axl.legend(); axl.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("sac_training_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sac_training_plot.png")