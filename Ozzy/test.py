from stable_baselines3 import PPO
import segway_env  # registers "Segway-v0" on import
import gymnasium as gym
import imageio

OUTPUT_PATH = "segway_rollout.mp4"
MODEL_PATH = "ppo_idle_segway"
NUM_STEPS = 1000

env = gym.make("Segway-v0")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save(MODEL_PATH)

env = gym.make("Segway-v0", render_mode="rgb_array")
model = PPO.load(MODEL_PATH)

dt = env.unwrapped.dt
fps = 1.0 / dt

frames = []
observation, info = env.reset(seed=42)
frames.append(env.render())

for i in range(NUM_STEPS):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())

    if terminated or truncated:
        print(f"Episode finished after {i} timesteps")
        observation, info = env.reset()
        frames.append(env.render())

env.close()

imageio.mimsave(OUTPUT_PATH, frames, fps=fps)
print(f"Saved {len(frames)} frames at {fps:.1f} fps -> {OUTPUT_PATH}")
print(f"Video duration: {len(frames) / fps:.1f} seconds")
