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


def plot_episode_traces(net, algo="model", x_goal=2.0, save_prefix=None,
                        XML_PATH=None, get_z_for_tilt=None, get_obs=None):
    """Plots position, tilt, and torque traces for a single deterministic episode.
    
    Useful for visualizing the behavior of a trained policy.
    
    Args:
        net: Trained policy network
        algo: Algorithm name for title
        x_goal: Goal x position
        save_prefix: Prefix for saved PNG file
        XML_PATH: Path to XML model file
        get_z_for_tilt: Function to get z position for tilt
        get_obs: Function to get observation from data
    """
    if XML_PATH is None or get_z_for_tilt is None or get_obs is None:
        raise ValueError("Must provide XML_PATH, get_z_for_tilt, and get_obs functions")
    
    save_prefix = save_prefix or algo.lower()

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    mujoco.mj_resetData(model, data)
    data.qpos[0]=0.0; data.qpos[2]=get_z_for_tilt(0.0)
    data.qpos[3:7]=[1,0,0,0]; data.qvel[:]=0.0
    mujoco.mj_forward(model, data)
    obs=get_obs(data)
    ts,xs,ths,torqs=[],[],[],[]
    prev_x=0.0

    for step in range(3000):
        with torch.no_grad():
            torque,_,_ = net.get_action(obs, x_goal, deterministic=True)
        u=float(np.clip(torque,-5,5)); data.ctrl[0]=-u; data.ctrl[1]=0.0
        mujoco.mj_step(model,data); obs=get_obs(data); x,xd,th,thd=obs
        ts.append(step*dt); xs.append(x); ths.append(np.degrees(th)); torqs.append(torque)
        prev_x=x
        if abs(x)>2.8 or abs(th)>0.5 or (abs(x-x_goal)<0.25 and abs(th)<0.35):
            break

    ts=np.array(ts); xs=np.array(xs); ths=np.array(ths); torqs=np.array(torqs)
    arrived = abs(xs[-1]-x_goal)<0.25 and abs(ths[-1])<20
    rms=np.sqrt(np.mean(torqs**2)); sat=np.abs(torqs)>4.8

    # 2x2 grid: position spans the top row, tilt + torque on the bottom
    fig=plt.figure(figsize=(15,9))
    gs=gridspec.GridSpec(2,2,hspace=0.38,wspace=0.25,figure=fig)
    fig.suptitle(f"{algo} — Deterministic Episode Behaviour",
                 fontsize=13, fontweight="bold")

    # position (spans top)
    ax=fig.add_subplot(gs[0,:])
    ax.plot(ts,xs,"#E74C3C",lw=2.5,label="x position")
    ax.axhline(0,color="blue",ls=":",lw=2,label="A (start)")
    ax.axhline(x_goal,color="green",ls=":",lw=2,label=f"B (goal={x_goal}m)")
    ax.axhspan(x_goal-0.25,x_goal+0.25,alpha=0.1,color="green",label="Goal zone ±0.25m")
    if arrived:
        idx=np.where(np.abs(xs-x_goal)<0.25)[0][0]
        ax.axvline(ts[idx],color="green",ls="--",alpha=0.6)
        ax.annotate(f"ARRIVED\nt={ts[idx]:.1f}s",xy=(ts[idx],xs[idx]),
                    xytext=(ts[idx]+0.2,x_goal-0.3),color="green",fontsize=9,
                    arrowprops=dict(arrowstyle="->",color="green"))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Position x (m)")
    ax.set_title("Position: A → B",fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # tilt (bottom-left)
    ax=fig.add_subplot(gs[1,0])
    ax.plot(ts,ths,"#E74C3C",lw=2,label="θ (degrees)")
    ax.axhspan(-20,20,alpha=0.06,color="green",label="Stable ±20°")
    ax.axhline(0,color="gray",alpha=0.4)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Tilt Angle (°)")
    ax.set_title("Tilt Angle θ",fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # torque (bottom-right)
    ax=fig.add_subplot(gs[1,1])
    ax.plot(ts,torqs,"#2ECC71",lw=2,label="Torque (Nm)")
    ax.fill_between(ts,torqs,0,where=(torqs>0),alpha=0.15,color="red",label="Forward")
    ax.fill_between(ts,torqs,0,where=(torqs<0),alpha=0.15,color="blue",label="Backward")
    ax.axhline(5,color="orange",ls=":",lw=1.5,label="Motor limit ±5Nm")
    ax.axhline(-5,color="orange",ls=":",lw=1.5)
    ax.axhline(0,color="gray",alpha=0.4)
    if sat.any():
        ax.fill_between(ts,-5,5,where=sat,alpha=0.15,color="red",label="Saturated!")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Torque (Nm)")
    ax.set_title(f"Motor Torque  (RMS={rms:.2f} Nm)",fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.savefig(f"{save_prefix}_episode_traces.png",dpi=150,bbox_inches="tight")
    plt.show(); print(f"Saved: {save_prefix}_episode_traces.png")

    print(f"\n{'='*55}\n  {algo} Episode Summary\n{'='*55}")
    print(f"  Duration   : {ts[-1]:.1f}s  ({len(ts)} steps)")
    print(f"  Max x      : {max(xs):.3f}m  / {x_goal}m")
    print(f"  Final x    : {xs[-1]:.3f}m")
    print(f"  Avg tilt   : {np.mean(np.abs(ths)):.1f}°")
    print(f"  Max tilt   : {max(np.abs(ths)):.1f}°")
    print(f"  Torque RMS : {rms:.3f} Nm")
    print(f"  Saturated  : {sat.sum()} / {len(ts)} steps ({sat.mean()*100:.0f}%)")
    print(f"{'='*55}")

    return {"max_x":float(max(xs)),"final_x":float(xs[-1]),
            "avg_tilt":float(np.mean(np.abs(ths))),"torque_rms":float(rms),
            "arrived":bool(arrived),"duration_s":float(ts[-1])}

