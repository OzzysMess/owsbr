# ==========================================================================
# SHARED TRACKER  —  paste this ONCE near the top of BOTH notebooks
# (after the imports cell). Identical in both so the logs are comparable.
# ==========================================================================
import time, json, os
import numpy as np
from collections import deque

class RunLogger:
    """
    Records identical metrics for PPO and SAC against ENVIRONMENT STEPS
    (the fair x-axis for comparing on-policy vs off-policy).

    Call .episode_end(...) once per finished episode.
    Call .save(...) when training finishes.
    """
    def __init__(self, algo, seed, log_every_steps=2000,
                 arrival_window=200, reward_window=100):
        self.algo  = algo            # "PPO" or "SAC"
        self.seed  = seed
        self.t0    = time.time()
        self.log_every = log_every_steps
        self._next_log = log_every_steps

        self.rewards         = []                       # per-episode return
        self.recent_arrivals = deque(maxlen=arrival_window)
        self.reward_window   = reward_window

        self.rows = []               # the time-series we plot later
        self.steps_to_converge = None   # first step where arrival hits 100%

    def episode_end(self, total_steps, ep_count, ep_return, arrived, loss):
        """Call at the end of every episode."""
        self.rewards.append(ep_return)
        self.recent_arrivals.append(1 if arrived else 0)

        avg_reward  = float(np.mean(self.rewards[-self.reward_window:]))
        arrival_pct = float(np.mean(self.recent_arrivals) * 100) if self.recent_arrivals else 0.0

        # convergence = first time the rolling arrival window is 100%
        if (self.steps_to_converge is None
                and len(self.recent_arrivals) == self.recent_arrivals.maxlen
                and arrival_pct >= 100.0):
            self.steps_to_converge = total_steps

        # sample the time-series on a fixed STEP grid (not every episode),
        # so PPO and SAC have comparable x-axis density
        if total_steps >= self._next_log:
            self.rows.append({
                "steps":       int(total_steps),
                "episodes":    int(ep_count),
                "wall_time":   round(time.time() - self.t0, 2),
                "avg_reward":  round(avg_reward, 3),
                "arrival_pct": round(arrival_pct, 2),
                "loss":        round(float(loss), 4),
            })
            self._next_log += self.log_every

        return avg_reward, arrival_pct   # so you can still print them

    def save(self, save_dir, tag, eval_results=None):
        os.makedirs(save_dir, exist_ok=True)
        payload = {
            "algo":              self.algo,
            "seed":              self.seed,
            "total_wall_time":   round(time.time() - self.t0, 2),
            "total_steps":       self.rows[-1]["steps"] if self.rows else 0,
            "total_episodes":    self.rows[-1]["episodes"] if self.rows else 0,
            "steps_to_converge": self.steps_to_converge,   # None = never hit 100%
            "timeseries":        self.rows,
            "eval_results":      eval_results,             # filled in after eval
        }
        path = f"{save_dir}/cmp_{self.algo}_seed{tag}.json"
        json.dump(payload, open(path, "w"), indent=2)
        print(f"  [logger] saved {path}")
        return path
