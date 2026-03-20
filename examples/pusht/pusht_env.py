import dataclasses
import gymnasium as gym
import numpy as np
from collections import deque
from plugrl_env_client.envs.base_env import (
    Action,
    BaseEnv,
    BaseEnvConfig,
    BoolArray,
    Observation,
    RewardArray,
)
from plugrl_env_client.utils.registration import register_env, register_env_config

TASK = "Push the T-shaped block onto the T-shaped target."

UID = "PushT-v1"


@register_env_config(UID)
@dataclasses.dataclass
class PushTConfig(BaseEnvConfig):
    name: str = "gym_pusht/PushT-v0"
    render_mode: str = "rgb_array"
    obs_type: str = "pixels_agent_pos"
    num_stacked_frames: int = 2
    early_termination: bool = True
    sparse_reward: bool = True


@register_env(UID, best_reward_threshold_for_success=0.95, max_episode_steps=250)
class PushTEnv(BaseEnv):
    def __init__(
        self,
        config: PushTConfig,
        process_id: int | None = None,
        total_processes: int | None = None,
    ):
        super().__init__(
            config=config,
            process_id=process_id,
            total_processes=total_processes,
        )
        env = gym.make(
            config.name,
            render_mode=config.render_mode,
            obs_type=config.obs_type,
        )
        self.env = env
        self.obs_queue = deque(maxlen=config.num_stacked_frames)
        self.num_stacked_frames = config.num_stacked_frames
        self.early_termination = config.early_termination
        self.sparse_reward = config.sparse_reward

    def prepare_obs(self, obs: dict) -> Observation:
        self.obs_queue.append(obs)
        if len(self.obs_queue) < self.num_stacked_frames:
            for _ in range(self.num_stacked_frames - len(self.obs_queue)):
                self.obs_queue.appendleft(obs)
        images, states = {}, {}
        for i in range(len(self.obs_queue)):
            frame = self.obs_queue[i]
            images[f"pixels_{i}"] = frame["pixels"][None]
            states[f"agent_pos_{i}"] = frame["agent_pos"][None]
        return Observation(
            images=images,
            states=states,
            text=TASK,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, dict]:
        self.obs_queue.clear()
        raw_obs, info = self.env.reset(seed=seed, options=options)
        obs = self.prepare_obs(raw_obs)
        return obs, info

    def step(
        self, actions: Action
    ) -> tuple[Observation, RewardArray, BoolArray, BoolArray, dict]:
        raw_obs, reward, terminated, truncated, info = self.env.step(actions[0])
        obs = self.prepare_obs(raw_obs)
        if self.sparse_reward:
            reward = float(terminated)
        if not self.early_termination:
            terminated = False
        reward_arr = np.asarray([float(reward)], dtype=np.float32)
        terminated_arr = np.asarray([bool(terminated)], dtype=np.bool_)
        truncated_arr = np.asarray([bool(truncated)], dtype=np.bool_)
        return obs, reward_arr, terminated_arr, truncated_arr, info


if __name__ == "__main__":
    from plugrl_env_client.cli import main

    main()
