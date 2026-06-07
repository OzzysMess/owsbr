import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import imageio
import base64
import mujoco
from PIL import Image as PILImage, ImageDraw
from IPython.display import display, HTML


def bring_window_to_foreground():
    """Bring the MuJoCo viewer window to the foreground (Windows only)"""
    if sys.platform == 'win32':
        import ctypes
        import subprocess
        
        try:
            # Method 1: Try using subprocess to activate the last created window
            # This is more reliable than direct ctypes calls
            subprocess.Popen("powershell -Command \"[Windows.System.Launcher]::LaunchUriAsync('ms-settings:') | Out-Null; Start-Sleep -Milliseconds 100\"", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.1)
            
            # Method 2: Try to find and activate any GLFW window (MuJoCo uses GLFW)
            hwnd = ctypes.windll.user32.FindWindowW(ctypes.c_wchar_p("GLFW30"), None)
            if hwnd:
                ctypes.windll.user32.SetForegroundWindow(hwnd)
                time.sleep(0.1)
                return
            
            # Method 3: Use keyboard shortcut to switch windows (Alt+Tab alternative)
            ctypes.windll.user32.keybd_event(0xA4, 0, 0, 0)  # Alt down
            time.sleep(0.05)
            ctypes.windll.user32.keybd_event(0x09, 0, 0, 0)  # Tab down
            time.sleep(0.05)
            ctypes.windll.user32.keybd_event(0x09, 0, 0x2, 0)  # Tab up
            time.sleep(0.05)
            ctypes.windll.user32.keybd_event(0xA4, 0, 0x2, 0)  # Alt up
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Could not bring window to foreground: {e}")


# ==========================================================================
# Plotting and Recording Functions
# ==========================================================================

def plot_loss_reward(logger=None, rewards=None, losses=None,
                     algo="model", save_prefix=None):
    """Plot reward and loss curves from a RunLogger or raw arrays.
    
    Works for PPO or SAC. Both panels use ENVIRONMENT STEPS on the x-axis.
    
    Args:
        logger: RunLogger object (preferred)
        rewards: Raw reward array (fallback)
        losses: Raw loss array (fallback)
        algo: Algorithm name for title
        save_prefix: Prefix for saved PNG file
    """
    save_prefix = save_prefix or algo.lower()

    wall_time = None
    if logger is not None:
        steps_ts  = [r["steps"]      for r in logger.rows]
        reward_ts = [r["avg_reward"] for r in logger.rows]
        loss_ts   = [r["loss"]       for r in logger.rows]
        wall_time = getattr(logger, "total_wall_time", None)
        if wall_time is None and logger.rows:
            wall_time = logger.rows[-1].get("wall_time")
        have_steps = True
    else:
        if rewards is None or losses is None:
            raise ValueError("Pass either logger=, or both rewards= and losses=")
        have_steps = False

    fig, (axr, axl) = plt.subplots(1, 2, figsize=(15, 5))
    title = f"{algo} — Training"
    if wall_time is not None:
        title += f"   (wall time: {wall_time/60:.1f} min)"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ---- LEFT: reward vs STEPS ----
    if have_steps:
        x = np.array(steps_ts); y = np.array(reward_ts)
        axr.plot(x, y, color="#E74C3C", lw=2.5,
                 label=f"avg reward (peak={y.max():.1f})")
        axr.set_xlabel("Environment steps")
    else:
        r_arr = np.array(rewards)
        w = min(100, max(2, len(r_arr)//10))
        roll = np.convolve(r_arr, np.ones(w)/w, mode='valid')
        axr.plot(np.arange(len(r_arr)), r_arr, color="#3498DB", alpha=0.2, lw=0.6,
                 label="per-episode")
        axr.plot(np.arange(len(roll))+w//2, roll, color="#E74C3C", lw=2.5,
                 label=f"{w}-ep avg (peak={roll.max():.1f})")
        axr.set_xlabel("Episode (no step data)")
    axr.axhline(0, color="green", ls="--", alpha=0.5)
    axr.set_ylabel("Reward")
    axr.set_title("Training Reward", fontweight="bold")
    axr.legend(fontsize=9); axr.grid(alpha=0.3)

    # ---- RIGHT: loss vs STEPS ----
    loss_label = "Critic Loss" if algo.upper() == "SAC" else "PPO Loss"
    if have_steps:
        axl.plot(np.array(steps_ts), np.array(loss_ts), color="#9B59B6", lw=2, label="Loss")
        axl.set_xlabel("Environment steps")
    else:
        y = np.array(losses)
        axl.plot(np.arange(len(y)), y, color="#9B59B6", alpha=0.4, lw=0.6)
        k = min(20, max(2, len(y)//10))
        roll = np.convolve(y, np.ones(k)/k, mode='valid')
        axl.plot(np.arange(len(roll)), roll, color="#9B59B6", lw=2.5, label=f"smoothed ({k})")
        axl.set_xlabel("Update / rollout index")
    axl.axhline(0, color="green", ls="--", alpha=0.5)
    axl.set_ylabel(loss_label)
    axl.set_title("Training Loss", fontweight="bold")
    axl.legend(fontsize=9); axl.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss_reward.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_prefix}_loss_reward.png")
    if wall_time is not None:
        print(f"Training (wall) time: {wall_time/60:.1f} min  ({wall_time:.0f} s)")


def record_all_seeds(nav_net, train_seeds=[42], verify_seeds=[788,999,555,321,444],
                     x_start=0.0, x_goal=2.0, max_steps=3000,
                     XML_PATH=None, get_z_for_tilt=None, get_obs=None):
    """Render and save videos for all seeds, with on-screen annotations.
    
    Args:
        nav_net: Trained policy network
        train_seeds: List of training seed values
        verify_seeds: List of verification seed values
        x_start: Starting x position
        x_goal: Goal x position
        max_steps: Maximum steps per episode
        XML_PATH: Path to XML model file
        get_z_for_tilt: Function to get z position for tilt
        get_obs: Function to get observation from data
    """
    if XML_PATH is None or get_z_for_tilt is None or get_obs is None:
        raise ValueError("Must provide XML_PATH, get_z_for_tilt, and get_obs functions")
    
    model    = mujoco.MjModel.from_xml_path(XML_PATH)
    data     = mujoco.MjData(model)
    dt       = model.opt.timestep
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat=np.array([1.0,0.0,0.3]); cam.distance=4.5
    cam.azimuth=90; cam.elevation=-15
    PX_A=160; PX_B=480
    def world_to_px(wx): return int(PX_A+(wx/x_goal)*(PX_B-PX_A))
    all_seeds=[(s,"TRAIN") for s in train_seeds]+[(s,"VERIFY") for s in verify_seeds]
    print("Recording seeds...")
    for seed,role in all_seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        mujoco.mj_resetData(model,data)
        data.qpos[0]=x_start; data.qpos[1]=0.0
        data.qpos[2]=get_z_for_tilt(0.0); data.qpos[3:7]=[1,0,0,0]
        data.qvel[:]=0.0; mujoco.mj_forward(model,data)
        obs=get_obs(data); frames=[]; strict=arrived=fell=False
        for step in range(max_steps):
            with torch.no_grad():
                torque,_,_ = nav_net.get_action(obs,x_goal,deterministic=True)
            data.ctrl[0]=-float(np.clip(torque,-5,5)); data.ctrl[1]=0.0
            mujoco.mj_step(model,data); obs=get_obs(data); x,_,theta,_=obs
            fell=abs(x)>2.8 or abs(theta)>0.5
            arrived=abs(x-x_goal)<0.25 and abs(theta)<0.35
            strict=abs(x-x_goal)<0.10 and abs(theta)<0.35
            renderer.update_scene(data,camera=cam)
            img=PILImage.fromarray(renderer.render()); draw=ImageDraw.Draw(img); W,H=img.size
            prog=float(np.clip(x/x_goal,0,1)); bw=W-40
            pcol=(0,200,0) if strict else (255,140,0) if arrived else (30,100,220)
            draw.rectangle([20,8,W-20,28],fill=(40,40,40))
            draw.rectangle([20,8,20+int(bw*prog),28],fill=pcol)
            draw.text((22,10),"A",fill=(255,255,255)); draw.text((W-28,10),"B",fill=(255,255,255))
            draw.text((W//2-40,10),f"{prog*100:.0f}%  x={x:.3f}m",fill=(255,255,255))
            badge_col=(0,60,140) if role=="TRAIN" else (100,0,140)
            draw.rectangle([8,34,170,58],fill=badge_col)
            draw.text((12,38),f"[{role}] seed={seed}",fill=(255,255,255))
            if strict:   sc=(0,120,0);   st=f"✓ STRICT  x={x:.3f}m  t={step*dt:.1f}s"
            elif arrived:sc=(120,100,0); st=f"~ LOOSE   x={x:.3f}m  t={step*dt:.1f}s"
            elif fell:   sc=(140,0,0);   st=f"✗ FELL    x={x:.3f}m  θ={np.degrees(theta):.1f}°"
            else:        sc=(20,20,70);  st=f"x={x:+.3f}m  θ={np.degrees(theta):+.1f}°  u={torque:+.1f}Nm  t={step*dt:.1f}s"
            draw.rectangle([175,34,W-8,58],fill=sc); draw.text((178,38),st,fill=(255,255,255))
            my=H-55
            a_px=world_to_px(x_start); b_px=world_to_px(x_goal)
            s_px=world_to_px(float(np.clip(x,-0.3,2.5)))
            draw.line([(a_px-5,my+15),(b_px+5,my+15)],fill=(180,180,180),width=3)
            draw.ellipse([a_px-14,my+4,a_px+14,my+26],fill=(50,100,220),outline=(255,255,255),width=2)
            draw.text((a_px-5,my+8),"A",fill=(255,255,255))
            draw.ellipse([b_px-14,my+4,b_px+14,my+26],fill=(0,180,70),outline=(255,255,255),width=2)
            draw.text((b_px-5,my+8),"B",fill=(255,255,255))
            scol=(0,220,100) if strict else (255,210,0)
            draw.ellipse([s_px-9,my+7,s_px+9,my+23],fill=scol,outline=(255,255,255),width=2)
            frames.append(np.array(img))
            if strict or fell: frames+=[frames[-1]]*60; break
        result="STRICT" if strict else "FELL" if fell else f"LOOSE@{x:.2f}m"
        print(f"  [{role}] seed={seed}: {result}")
        fname=f"nav_ppo_{role.lower()}_seed{seed}.mp4"
        imageio.mimsave(fname,frames,fps=30)
        sz=os.path.getsize(fname)//1024; print(f"    saved {fname} ({sz} KB)")
        b64=base64.b64encode(open(fname,"rb").read()).decode()
        display(HTML(f'<b>[{role}] Seed {seed} — {result}</b><br>'
                     f'<video width="640" height="480" controls autoplay loop>'
                     f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'))
    print("\nDone.")

def plot_episode_traces(net, algo="model", x_goal=2.0, y_goal=0.0, save_prefix=None,
                        XML_PATH=None, get_z_for_tilt=None, get_obs=None,
                        x_start=0.0, y_start=0.0, mode="2d"):
    """Plots position, tilt, and torque traces for a single deterministic episode.

    Args:
        net:            Trained policy network
        algo:           Algorithm name for title
        x_goal:         Goal x position
        y_goal:         Goal y position
        save_prefix:    Prefix for saved PNG file
        XML_PATH:       Path to XML model file
        get_z_for_tilt: Function to get z position for tilt
        get_obs:        Function to get observation from data
        x_start:        Starting x position
        y_start:        Starting y position
        mode:           "1d" — original single-axis plot
                        "2d" — extended plots with XY trajectory, yaw, both torques
                        "both" — saves both versions
    """
    if XML_PATH is None or get_z_for_tilt is None or get_obs is None:
        raise ValueError("Must provide XML_PATH, get_z_for_tilt, and get_obs functions")
    if mode not in ("1d", "2d", "both"):
        raise ValueError("mode must be '1d', '2d', or 'both'")

    save_prefix = save_prefix or algo.lower()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    mujoco.mj_resetData(model, data)
    data.qpos[0]   = x_start
    data.qpos[1]   = y_start
    data.qpos[2]   = get_z_for_tilt(0.0)
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qvel[:]   = 0.0
    mujoco.mj_forward(model, data)
    obs = get_obs(data)

    # ── Logging arrays ────────────────────────────────────────────────────────
    ts, xs, ys, ths, psis, w_torqs, r_torqs = [], [], [], [], [], [], []

    for step in range(3000):
        with torch.no_grad():
            w_t, r_t, _, _ = net.get_action(obs, x_goal, y_goal, deterministic=True)

        data.ctrl[0] = -float(np.clip(w_t, -5, 5))
        data.ctrl[1] =  float(np.clip(r_t, -5, 5))
        mujoco.mj_step(model, data)
        obs = get_obs(data)

        # Unpack 2D obs: [x, x_dot, y, y_dot, theta, theta_dot, psi, psi_dot, rw_w]
        x, x_dot, y, y_dot, theta, theta_dot, psi, psi_dot, rw_w = obs
        dist = np.sqrt((x - x_goal)**2 + (y - y_goal)**2)

        ts.append(step * dt)
        xs.append(x);              ys.append(y)
        ths.append(np.degrees(theta))
        psis.append(np.degrees(psi))
        w_torqs.append(w_t);       r_torqs.append(r_t)

        fell    = abs(theta) > 0.5
        arrived = dist < 0.25 and abs(theta) < 0.35
        if fell or arrived:
            break

    ts      = np.array(ts)
    xs      = np.array(xs);      ys      = np.array(ys)
    ths     = np.array(ths);     psis    = np.array(psis)
    w_torqs = np.array(w_torqs); r_torqs = np.array(r_torqs)

    dist_series = np.sqrt((xs - x_goal)**2 + (ys - y_goal)**2)
    arrived     = dist_series[-1] < 0.25 and abs(ths[-1]) < 20
    fell        = abs(ths[-1]) >= 20

    w_rms  = np.sqrt(np.mean(w_torqs**2))
    r_rms  = np.sqrt(np.mean(r_torqs**2))
    w_sat  = np.abs(w_torqs) > 4.8
    r_sat  = np.abs(r_torqs) > 4.8

    # ── Helper: annotate arrival ──────────────────────────────────────────────
    def annotate_arrival(ax, ts, series, goal_val, label="ARRIVED"):
        idx_arr = np.where(dist_series < 0.25)[0]
        if arrived and len(idx_arr):
            idx = idx_arr[0]
            ax.axvline(ts[idx], color="green", ls="--", alpha=0.6)
            ax.annotate(f"{label}\nt={ts[idx]:.1f}s",
                        xy=(ts[idx], series[idx]),
                        xytext=(ts[idx] + 0.2, goal_val),
                        color="green", fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="green"))

    def torque_panel(ax, ts, torqs, label, rms, sat, color):
        ax.plot(ts, torqs, color, lw=2, label=f"{label} (Nm)")
        ax.fill_between(ts, torqs, 0, where=(torqs > 0), alpha=0.15, color="red",   label="Forward")
        ax.fill_between(ts, torqs, 0, where=(torqs < 0), alpha=0.15, color="blue",  label="Backward")
        ax.axhline( 5, color="orange", ls=":", lw=1.5, label="Motor limit ±5 Nm")
        ax.axhline(-5, color="orange", ls=":", lw=1.5)
        ax.axhline( 0, color="gray", alpha=0.4)
        if sat.any():
            ax.fill_between(ts, -5, 5, where=sat, alpha=0.15, color="red", label="Saturated!")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Torque (Nm)")
        ax.set_title(f"{label}  (RMS={rms:.2f} Nm)", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ══════════════════════════════════════════════════════════════════════════
    # 1-D PLOT (original layout, updated to use 2D unpacking)
    # ══════════════════════════════════════════════════════════════════════════
    if mode in ("1d", "both"):
        fig = plt.figure(figsize=(15, 9))
        gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.25, figure=fig)
        fig.suptitle(f"{algo} — Deterministic Episode Behaviour (1D view)",
                     fontsize=13, fontweight="bold")

        # Position (top, full width)
        ax = fig.add_subplot(gs[0, :])
        ax.plot(ts, xs, "#E74C3C", lw=2.5, label="x position")
        ax.axhline(x_start, color="blue",  ls=":", lw=2, label="A (start)")
        ax.axhline(x_goal,  color="green", ls=":", lw=2, label=f"B (goal={x_goal}m)")
        ax.axhspan(x_goal - 0.25, x_goal + 0.25, alpha=0.1, color="green", label="Goal zone ±0.25m")
        annotate_arrival(ax, ts, xs, x_goal - 0.3)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Position x (m)")
        ax.set_title("Position: A → B", fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # Tilt (bottom-left)
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(ts, ths, "#E74C3C", lw=2, label="θ (degrees)")
        ax.axhspan(-20, 20, alpha=0.06, color="green", label="Stable ±20°")
        ax.axhline(0, color="gray", alpha=0.4)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Tilt Angle (°)")
        ax.set_title("Tilt Angle θ", fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # Wheel torque (bottom-right)
        ax = fig.add_subplot(gs[1, 1])
        torque_panel(ax, ts, w_torqs, "Wheel Torque", w_rms, w_sat, "#2ECC71")

        plt.savefig(f"{save_prefix}_episode_traces_1d.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {save_prefix}_episode_traces_1d.png")

    # ══════════════════════════════════════════════════════════════════════════
    # 2-D PLOT
    # ══════════════════════════════════════════════════════════════════════════
    if mode in ("2d", "both"):
        fig = plt.figure(figsize=(18, 12))
        gs  = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.32, figure=fig)
        fig.suptitle(f"{algo} — Deterministic Episode Behaviour (2D view)",
                     fontsize=13, fontweight="bold")

        # ── Top-left: XY trajectory ───────────────────────────────────────────
        ax = fig.add_subplot(gs[0:2, 0])
        # Colour the path by time so you can see progression
        sc = ax.scatter(xs, ys, c=ts, cmap="plasma", s=6, zorder=3, label="Path")
        plt.colorbar(sc, ax=ax, label="Time (s)", pad=0.02)
        ax.plot(x_start, y_start, "bo", ms=10, label=f"A start ({x_start},{y_start})", zorder=5)
        ax.plot(x_goal,  y_goal,  "g*", ms=14, label=f"B goal ({x_goal},{y_goal})",   zorder=5)
        goal_circle = plt.Circle((x_goal, y_goal), 0.25, color="green", alpha=0.15, label="Goal zone r=0.25m")
        ax.add_patch(goal_circle)
        # Heading arrows every N steps
        arrow_every = max(1, len(ts) // 20)
        for i in range(0, len(ts), arrow_every):
            psi_rad = np.radians(psis[i])
            ax.annotate("", xy=(xs[i] + 0.08 * np.cos(psi_rad),
                                 ys[i] + 0.08 * np.sin(psi_rad)),
                        xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="->", color="white", lw=0.8, alpha=0.5))
        if arrived:
            ax.plot(xs[-1], ys[-1], "g^", ms=10, label="Arrived", zorder=6)
        elif fell:
            ax.plot(xs[-1], ys[-1], "rx", ms=12, mew=2.5, label="Fell", zorder=6)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title("XY Trajectory + Heading", fontweight="bold")
        ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # ── Top-middle: distance to goal over time ────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(ts, dist_series, "#E74C3C", lw=2, label="dist to goal")
        ax.axhline(0.25, color="green", ls=":", lw=1.5, label="Goal zone 0.25m")
        ax.axhline(0.10, color="lime",  ls=":", lw=1.5, label="Strict zone 0.10m")
        annotate_arrival(ax, ts, dist_series, 0.3)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
        ax.set_title("Distance to Goal", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # ── Top-right: x and y positions separately ───────────────────────────
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(ts, xs, "#E74C3C", lw=2,   label="x position")
        ax.plot(ts, ys, "#3498DB", lw=2,   label="y position")
        ax.axhline(x_goal, color="#E74C3C", ls=":", lw=1.2, alpha=0.6, label=f"x goal={x_goal}m")
        ax.axhline(y_goal, color="#3498DB", ls=":", lw=1.2, alpha=0.6, label=f"y goal={y_goal}m")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Position (m)")
        ax.set_title("X / Y Positions", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # ── Middle-middle: tilt angle θ ───────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(ts, ths, "#E67E22", lw=2, label="θ tilt (°)")
        ax.axhspan(-20, 20, alpha=0.06, color="green", label="Stable ±20°")
        ax.axhline(0, color="gray", alpha=0.4)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Tilt (°)")
        ax.set_title("Tilt Angle θ", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # ── Middle-right: yaw angle ψ ─────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(ts, psis, "#9B59B6", lw=2, label="ψ yaw (°)")
        # Compute desired heading at each step
        desired_headings = np.degrees(np.arctan2(y_goal - ys, x_goal - xs))
        ax.plot(ts, desired_headings, "#9B59B6", lw=1.2, ls="--", alpha=0.6, label="Desired heading")
        heading_errors = desired_headings - psis
        # Wrap to [-180, 180]
        heading_errors = (heading_errors + 180) % 360 - 180
        ax.plot(ts, heading_errors, "#E74C3C", lw=1.5, ls="-.", alpha=0.8, label="Heading error")
        ax.axhline(0, color="gray", alpha=0.4)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (°)")
        ax.set_title("Yaw ψ vs Desired Heading", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # ── Bottom-left: wheel torque ─────────────────────────────────────────
        ax = fig.add_subplot(gs[2, 0])
        torque_panel(ax, ts, w_torqs, "Wheel Torque", w_rms, w_sat, "#2ECC71")

        # ── Bottom-middle: reaction wheel torque ──────────────────────────────
        ax = fig.add_subplot(gs[2, 1])
        torque_panel(ax, ts, r_torqs, "Reaction Wheel Torque", r_rms, r_sat, "#1ABC9C")

        # ── Bottom-right: phase portrait (x_dot vs x, y_dot vs y) ────────────
        ax = fig.add_subplot(gs[2, 2])
        x_dots = np.gradient(xs, ts)
        y_dots = np.gradient(ys, ts)
        ax.plot(xs, x_dots, "#E74C3C", lw=1.5, alpha=0.85, label="x phase (x vs ẋ)")
        ax.plot(ys, y_dots, "#3498DB", lw=1.5, alpha=0.85, label="y phase (y vs ẏ)")
        ax.plot(xs[0],  x_dots[0],  "o", color="#E74C3C", ms=7)
        ax.plot(xs[-1], x_dots[-1], "s", color="#E74C3C", ms=7)
        ax.plot(ys[0],  y_dots[0],  "o", color="#3498DB", ms=7)
        ax.plot(ys[-1], y_dots[-1], "s", color="#3498DB", ms=7)
        ax.axhline(0, color="gray", alpha=0.4); ax.axvline(0, color="gray", alpha=0.4)
        ax.set_xlabel("Position (m)"); ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Phase Portrait (○=start, □=end)", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        plt.savefig(f"{save_prefix}_episode_traces_2d.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {save_prefix}_episode_traces_2d.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_dist = dist_series[-1]
    print(f"\n{'='*55}\n  {algo} Episode Summary\n{'='*55}")
    print(f"  Duration      : {ts[-1]:.1f}s  ({len(ts)} steps)")
    print(f"  Final pos     : ({xs[-1]:.3f}, {ys[-1]:.3f})m")
    print(f"  Final dist    : {final_dist:.3f}m  (goal={x_goal},{y_goal})")
    print(f"  Avg tilt      : {np.mean(np.abs(ths)):.1f}°")
    print(f"  Max tilt      : {max(np.abs(ths)):.1f}°")
    print(f"  Wheel RMS     : {w_rms:.3f} Nm  (sat {w_sat.mean()*100:.0f}%)")
    print(f"  React RMS     : {r_rms:.3f} Nm  (sat {r_sat.mean()*100:.0f}%)")
    print(f"  Result        : {'ARRIVED' if arrived else 'FELL' if fell else 'TIMEOUT'}")
    print(f"{'='*55}")

    return {
        "final_x": float(xs[-1]),   "final_y": float(ys[-1]),
        "final_dist": float(final_dist),
        "avg_tilt": float(np.mean(np.abs(ths))),
        "w_torque_rms": float(w_rms), "r_torque_rms": float(r_rms),
        "arrived": bool(arrived),     "fell": bool(fell),
        "duration_s": float(ts[-1]),
    }

def record_all_seeds_2D(nav_net, train_seeds=[42], verify_seeds=[788,999,555,321,444],
                     x_start=0.0, x_goal=2.0, y_start=0.0, y_goal=2.0, max_steps=3000,
                     XML_PATH=None, get_z_for_tilt=None, get_obs=None,
                     playback_speed=1, speed_mode="skip"):
    """Render and save videos for all seeds, with on-screen annotations (2D case).
    
    Args:
        nav_net: Trained policy network
        train_seeds: List of training seed values
        verify_seeds: List of verification seed values
        x_start: Starting x position
        x_goal: Goal x position
        y_start: Starting y position
        y_goal: Goal y position
        max_steps: Maximum steps per episode
        XML_PATH: Path to XML model file
        get_z_for_tilt: Function to get z position for tilt
        get_obs: Function to get observation from data
        playback_speed: Integer k — speed multiplier for output video.
        speed_mode:     "skip" = keep every k-th frame (smaller file, same fps).
                        "fps"  = keep all frames, multiply fps by k.
    """
    if XML_PATH is None or get_z_for_tilt is None or get_obs is None:
        raise ValueError("Must provide XML_PATH, get_z_for_tilt, and get_obs functions")
    if playback_speed < 1 or not isinstance(playback_speed, int):
        raise ValueError("playback_speed must be a positive integer")
    if speed_mode not in ("skip", "fps"):
        raise ValueError("speed_mode must be 'skip' or 'fps'")
    
    BASE_FPS = 30
    output_fps = BASE_FPS * playback_speed if speed_mode == "fps" else BASE_FPS

    model    = mujoco.MjModel.from_xml_path(XML_PATH)
    data     = mujoco.MjData(model)
    dt       = model.opt.timestep
    print(f"Time step = {dt:.4f}s, base FPS = {1/dt:.1f}, output FPS = {output_fps} (mode={speed_mode})")
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Overhead camera for 2D visualization
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([(x_start + x_goal) / 2.0, (y_start + y_goal) / 2.0, 0.3])
    cam.distance = 6.0
    cam.azimuth = 90
    cam.elevation = -45  # steeper angle to see 2D ground plane better

    # 2D minimap pixel bounds (for the XY minimap overlay)
    MAP_X0, MAP_Y0 = 10, 320          # top-left corner of minimap on frame
    MAP_W,  MAP_H  = 120, 120         # minimap size in pixels
    MAP_MARGIN     = 0.5              # world-unit padding around goal/start

    def world_to_map(wx, wy):
        """Convert world (x,y) to minimap pixel coords."""
        wx_min = min(x_start, x_goal) - MAP_MARGIN
        wx_max = max(x_start, x_goal) + MAP_MARGIN
        wy_min = min(y_start, y_goal) - MAP_MARGIN
        wy_max = max(y_start, y_goal) + MAP_MARGIN
        px = MAP_X0 + int((wx - wx_min) / (wx_max - wx_min) * MAP_W)
        py = MAP_Y0 + MAP_H - int((wy - wy_min) / (wy_max - wy_min) * MAP_H)  # flip y
        return px, py

    # Goal distance for progress bar normalisation
    goal_dist = np.sqrt((x_goal - x_start)**2 + (y_goal - y_start)**2)
    all_seeds = [(s, "TRAIN") for s in train_seeds] + [(s, "VERIFY") for s in verify_seeds]
    print("Recording seeds...")

    for seed, role in all_seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        mujoco.mj_resetData(model, data)

        # 2D initial position
        data.qpos[0] = x_start
        data.qpos[1] = y_start
        data.qpos[2] = get_z_for_tilt(0.0)
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        obs = get_obs(data)
        frames = []
        strict = arrived = fell = False

        for step in range(max_steps):
            with torch.no_grad():
                w_t, r_t, _, _ = nav_net.get_action(obs, x_goal, y_goal, deterministic=True)

            # ctrl[0] = wheel torque, ctrl[1] = reaction wheel torque
            data.ctrl[0] = -float(np.clip(w_t, -5, 5))
            data.ctrl[1] =  float(np.clip(r_t, -5, 5))
            mujoco.mj_step(model, data)
            obs = get_obs(data)

            # Unpack 2D obs: [x, x_dot, y, y_dot, theta, theta_dot, psi, psi_dot]
            x, x_dot, y, y_dot, theta, theta_dot, psi, psi_dot, rw_w = obs

            dist_to_goal = np.sqrt((x - x_goal)**2 + (y - y_goal)**2)
            fell    = abs(theta) > 0.5
            arrived = dist_to_goal < 0.25 and abs(theta) < 0.35
            strict  = dist_to_goal < 0.10 and abs(theta) < 0.35

            # ── Frame skip: only render/annotate on kept frames ───────────────
            # Always render the very last frame or terminal frames for clean endings
            is_terminal = strict or fell
            should_keep = (step % playback_speed == 0) or is_terminal

            if speed_mode == "skip" and not should_keep:
                if is_terminal:
                    frames += [frames[-1]] * 60
                    break
                continue  # skip rendering this frame entirely

            renderer.update_scene(data, camera=cam)
            img  = PILImage.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            W, H = img.size

            # ── Progress bar (distance to goal) ──────────────────────────────
            prog = float(np.clip(1.0 - dist_to_goal / goal_dist, 0, 1))
            bw   = W - 40
            pcol = (0, 200, 0) if strict else (255, 140, 0) if arrived else (30, 100, 220)
            draw.rectangle([20, 8, W - 20, 28], fill=(40, 40, 40))
            draw.rectangle([20, 8, 20 + int(bw * prog), 28], fill=pcol)
            draw.text((22, 10),   "Start", fill=(255, 255, 255))
            draw.text((W - 50, 10), "Goal",  fill=(255, 255, 255))
            draw.text((W // 2 - 55, 10),
                      f"{prog*100:.0f}%  dist={dist_to_goal:.3f}m",
                      fill=(255, 255, 255))

            # ── Seed / role badge ─────────────────────────────────────────────
            badge_col = (0, 60, 140) if role == "TRAIN" else (100, 0, 140)
            draw.rectangle([8, 34, 170, 58], fill=badge_col)
            draw.text((12, 38), f"[{role}] seed={seed}", fill=(255, 255, 255))

            # ── Speed indicator ───────────────────────────────────────────────
            if playback_speed > 1:
                draw.rectangle([8, 62, 100, 82], fill=(60, 30, 0))
                draw.text((12, 65), f"▶▶ {playback_speed}x", fill=(255, 200, 50))

            # ── Status bar ───────────────────────────────────────────────────
            if strict:
                sc = (0, 120, 0)
                st = f"✓ STRICT  dist={dist_to_goal:.3f}m  t={step*dt:.1f}s"
            elif arrived:
                sc = (120, 100, 0)
                st = f"~ LOOSE   dist={dist_to_goal:.3f}m  t={step*dt:.1f}s"
            elif fell:
                sc = (140, 0, 0)
                st = f"✗ FELL    θ={np.degrees(theta):.1f}°  t={step*dt:.1f}s"
            else:
                sc = (20, 20, 70)
                st = (f"x={x:+.2f} y={y:+.2f}  "
                      f"θ={np.degrees(theta):+.1f}°  ψ={np.degrees(psi):+.1f}°  "
                      f"wT={w_t:+.1f} rT={r_t:+.1f}  t={step*dt:.1f}s")
            draw.rectangle([175, 34, W - 8, 58], fill=sc)
            draw.text((178, 38), st, fill=(255, 255, 255))

            # ── 2D minimap ───────────────────────────────────────────────────
            # Background
            draw.rectangle([MAP_X0, MAP_Y0, MAP_X0 + MAP_W, MAP_Y0 + MAP_H],
                           fill=(20, 20, 40), outline=(150, 150, 150), width=2)
            draw.text((MAP_X0 + 2, MAP_Y0 + 2), "top-down", fill=(180, 180, 180))

            # Start marker (blue A)
            ax, ay = world_to_map(x_start, y_start)
            draw.ellipse([ax-8, ay-8, ax+8, ay+8], fill=(50, 100, 220), outline=(255,255,255), width=1)
            draw.text((ax-4, ay-6), "A", fill=(255, 255, 255))

            # Goal marker (green B)
            bx, by = world_to_map(x_goal, y_goal)
            draw.ellipse([bx-8, by-8, bx+8, by+8], fill=(0, 180, 70), outline=(255,255,255), width=1)
            draw.text((bx-4, by-6), "B", fill=(255, 255, 255))

            # Robot position + heading arrow
            rx, ry = world_to_map(float(np.clip(x, x_start - MAP_MARGIN, x_goal + MAP_MARGIN)),
                                  float(np.clip(y, y_start - MAP_MARGIN, y_goal + MAP_MARGIN)))
            rcol = (0, 220, 100) if strict else (255, 210, 0) if arrived else (220, 80, 80) if fell else (255, 255, 255)
            draw.ellipse([rx-6, ry-6, rx+6, ry+6], fill=rcol, outline=(50,50,50), width=1)
            # Heading arrow using yaw (psi)
            arrow_len = 12
            ax_tip = rx + int(arrow_len * np.cos(psi))
            ay_tip = ry - int(arrow_len * np.sin(psi))  # flip y for screen coords
            draw.line([(rx, ry), (ax_tip, ay_tip)], fill=(255, 100, 100), width=2)

            frames.append(np.array(img))
            if strict or fell:
                frames += [frames[-1]] * 60
                break

        result = "STRICT" if strict else "FELL" if fell else f"LOOSE@({x:.2f},{y:.2f})m"
        print(f"  [{role}] seed={seed}: {result}")

        fname = f"nav_ppo_{role.lower()}_seed{seed}.mp4"
        imageio.mimsave(fname, frames, fps=30)
        sz = os.path.getsize(fname) // 1024
        print(f"    saved {fname} ({sz} KB)")

        b64 = base64.b64encode(open(fname, "rb").read()).decode()
        display(HTML(f'<b>[{role}] Seed {seed} — {result}</b><br>'
                     f'<video width="640" height="480" controls autoplay loop>'
                     f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'))

    print("\nDone.")