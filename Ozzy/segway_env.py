"""
Custom Gymnasium environment wrapping segway_2.xml — a single-wheel,
self-balancing "segway" with a drive wheel and a reaction-wheel flywheel.

Supports three selectable reward modes:
  - "balance" : stay upright, minimize wobble/effort, no navigation goal
  - "forward" : stay upright AND drive toward a target point on the ground
  - "yaw"     : stay upright AND rotate to face a target heading
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

VALID_MODES = ("balance", "forward", "yaw")


class SegwayEnv(MujocoEnv, gym.utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        xml_file=None,
        frame_skip=5,
        reward_mode="balance",
        target_distance_range=(1.0, 3.0),
        success_bonus=10.0,
        **kwargs,
    ):
        if reward_mode not in VALID_MODES:
            raise ValueError(f"reward_mode must be one of {VALID_MODES}, got {reward_mode!r}")

        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "segway_2.xml")

        self.reward_mode = reward_mode
        self.target_distance_range = target_distance_range
        self.success_bonus = success_bonus

        # qpos(9) + qvel(8) + upright(1) + target_dx(1) + target_dy(1) + heading_err(1) = 21
        # (target fields are 0 / unused in "balance" mode, but kept for a consistent obs shape)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=frame_skip,
            observation_space=observation_space,
            **kwargs,
        )
        gym.utils.EzPickle.__init__(self)

        self._segway_body_id = self.model.body("segway").id
        self.target_pos = np.zeros(2)
        self.target_yaw = 0.0
        self._prev_dist_to_target = None

    # ---------- geometry helpers ----------

    def _rotmat(self):
        return self.data.xmat[self._segway_body_id].reshape(3, 3)

    def _upright_cosine(self):
        """1.0 when vertical, decreasing toward 0/negative as it tips."""
        return float(self._rotmat()[2, 2])

    def _yaw(self):
        """Heading angle in the world XY plane, assuming local +x is 'forward'."""
        forward_world = self._rotmat() @ np.array([1.0, 0.0, 0.0])
        return float(np.arctan2(forward_world[1], forward_world[0]))

    def _xy(self):
        return self.data.qpos[0:2].copy()

    @staticmethod
    def _wrap_angle(angle):
        """Wrap to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # ---------- gym API ----------

    def _get_obs(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        upright = self._upright_cosine()

        xy = self._xy()
        target_delta = self.target_pos - xy  # zeros in "balance" mode
        heading_err = self._wrap_angle(self.target_yaw - self._yaw())  # 0 in "balance"/"forward"

        extra = np.array([upright, target_delta[0], target_delta[1], heading_err])
        return np.concatenate([qpos, qvel, extra])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        upright = self._upright_cosine()
        height = self.data.qpos[2]
        ang_vel_penalty = 0.01 * np.sum(np.square(self.data.qvel[3:6]))
        effort_penalty = 0.001 * np.sum(np.square(action))

        reward, task_done = self._compute_reward(upright, ang_vel_penalty, effort_penalty)

        fallen = upright < 0.65
        too_low = height < 0.05
        terminated = bool(fallen or too_low or task_done)

        info = {"upright": upright, "reward_mode": self.reward_mode}
        return obs, reward, terminated, False, info

    def _compute_reward(self, upright, ang_vel_penalty, effort_penalty):
        """Returns (reward, task_done). task_done=True lets a mode end the
        episode early on success (e.g. target reached) with a bonus."""

        if self.reward_mode == "balance":
            reward = upright - ang_vel_penalty - effort_penalty
            return reward, False

        elif self.reward_mode == "forward":
            xy = self._xy()
            dist = float(np.linalg.norm(self.target_pos - xy))
            progress = self._prev_dist_to_target - dist  # positive when getting closer
            self._prev_dist_to_target = dist

            reward = progress + 0.3 * upright - ang_vel_penalty - effort_penalty
            reached = dist < 0.15
            if reached:
                reward += self.success_bonus
            return reward, reached

        elif self.reward_mode == "yaw":
            heading_err = self._wrap_angle(self.target_yaw - self._yaw())
            # reward shrinks as |heading_err| -> 0; max 1.0 when aligned
            alignment = 1.0 - (abs(heading_err) / np.pi)
            reward = alignment + 0.3 * upright - ang_vel_penalty - effort_penalty
            aligned = abs(heading_err) < np.deg2rad(5)
            if aligned:
                reward += self.success_bonus
            return reward, aligned

        raise RuntimeError(f"unhandled reward_mode {self.reward_mode!r}")

    def _sample_target(self):
        xy = self._xy()
        yaw = self._yaw()

        if self.reward_mode == "forward":
            dist = self.np_random.uniform(*self.target_distance_range)
            angle = self.np_random.uniform(-np.pi, np.pi)  # any direction, not just current heading
            self.target_pos = xy + dist * np.array([np.cos(angle), np.sin(angle)])
            self._prev_dist_to_target = float(np.linalg.norm(self.target_pos - xy))
            self.target_yaw = 0.0

        elif self.reward_mode == "yaw":
            self.target_yaw = self.np_random.uniform(-np.pi, np.pi)
            self.target_pos = xy  # unused

        else:  # balance
            self.target_pos = xy
            self.target_yaw = yaw

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(qpos, qvel)
        self._sample_target()
        return self._get_obs()


# Register one generic id (pass reward_mode as a kwarg) plus three convenience ids.
gym.register(id="Segway-v0", entry_point=SegwayEnv, max_episode_steps=1000)

gym.register(
    id="Segway-Balance-v0",
    entry_point=SegwayEnv,
    max_episode_steps=1000,
    kwargs={"reward_mode": "balance"},
)
gym.register(
    id="Segway-Forward-v0",
    entry_point=SegwayEnv,
    max_episode_steps=1000,
    kwargs={"reward_mode": "forward"},
)
gym.register(
    id="Segway-Yaw-v0",
    entry_point=SegwayEnv,
    max_episode_steps=1000,
    kwargs={"reward_mode": "yaw"},
)


if __name__ == "__main__":
    for mode in VALID_MODES:
        env = gym.make("Segway-v0", reward_mode=mode)
        obs, info = env.reset(seed=0)
        total_reward = 0.0
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, info = env.reset()
        print(f"[{mode}] random-policy smoke test total reward: {total_reward:.2f}")
        env.close()
