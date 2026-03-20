import dataclasses
from typing import Any, cast

import numpy as np
import gymnasium as gym

try:
    import ale_py  # type: ignore

    gym.register_envs(ale_py)
    from .atari_wrappers import (
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
    )
except ImportError:
    raise ImportError(
        "Atari is not installed. Please install it with the 'atari' extra, e.g. 'pip install plugrl-env-client[atari]'"
    )
from plugrl_env_client.envs.base_env import (
    BaseEnv,
    BaseEnvConfig,
    Action,
    BoolArray,
    Observation,
    RewardArray,
)
from plugrl_env_client.utils.registration import register_env, register_env_config

UID = "Atari-v1"


@register_env_config(UID)
@dataclasses.dataclass
class AtariConfig(BaseEnvConfig):
    name: str = "BreakoutNoFrameskip-v4"


@register_env(UID)
class AtariEnv(BaseEnv):
    env: gym.Env

    def __init__(
        self,
        config: AtariConfig,
        process_id: int | None = None,
        total_processes: int | None = None,
    ):
        super().__init__(
            config=config,
            process_id=process_id,
            total_processes=total_processes,
        )
        if self.num_envs != 1:
            raise ValueError("AtariEnv only supports num_envs=1")
        env = gym.make(config.name, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        try:
            action_meanings = cast(Any, env.unwrapped).get_action_meanings()
        except AttributeError:
            action_meanings = None
        if isinstance(action_meanings, (list, tuple)) and "FIRE" in action_meanings:
            env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        self.env = env
        self.game_name = config.name

    def prepare_obs(self, obs: np.ndarray) -> Observation:
        frames = {f"{i}": obs[i : i + 1, :, :, None] for i in range(obs.shape[0])}
        return Observation(
            images=frames,
            states={},
            text=self.game_name,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.prepare_obs(obs), info

    def step(
        self, actions: Action
    ) -> tuple[Observation, RewardArray, BoolArray, BoolArray, dict]:
        action_int = int(actions.item())
        obs, reward, terminated, truncated, info = self.env.step(action_int)
        reward_arr = np.asarray([float(reward)], dtype=np.float32)
        terminated_arr = np.asarray([bool(terminated)], dtype=np.bool_)
        truncated_arr = np.asarray([bool(truncated)], dtype=np.bool_)
        return self.prepare_obs(obs), reward_arr, terminated_arr, truncated_arr, info
