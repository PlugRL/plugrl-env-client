import dataclasses
import importlib.util
import numpy as np
import gymnasium as gym

if importlib.util.find_spec("pygame") is None:
    raise ImportError(
        'pygame is not installed. Please install it with pip install "plugrl-env-client[classic]".'
    )

from plugrl_env_client.envs.base_env import BaseEnv, BaseEnvConfig, Action, Observation
from plugrl_env_client.utils.registration import register_env, register_env_config

UID = "Classic-v1"


@register_env_config(UID)
@dataclasses.dataclass
class ClassicConfig(BaseEnvConfig):
    name: str = "CartPole-v1"


@register_env(UID)
class ClassicEnv(BaseEnv):
    env: gym.Env

    def __init__(
        self,
        config: ClassicConfig,
        num_envs: int = 1,
        process_id: int | None = None,
        total_processes: int | None = None,
    ):
        super().__init__(
            config=config,
            num_envs=num_envs,
            process_id=process_id,
            total_processes=total_processes,
        )
        if self.num_envs != 1:
            raise ValueError("ClassicEnv only supports num_envs=1")
        env = gym.make(config.name, render_mode="rgb_array")
        self.env = env
        self.game_name = config.name

    def prepare_obs(self, obs: np.ndarray) -> Observation:
        frame = self.env.render()
        frames = {"env": np.array(frame)[None, ...]}
        states = {"obs": obs[None, ...]}
        return Observation(
            images=frames,
            states=states,
            text=self.game_name,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation | None, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.prepare_obs(obs), info

    def step(
        self, action: Action
    ) -> tuple[Observation | None, np.ndarray, np.ndarray, np.ndarray, dict]:
        action = int(action.item())
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = np.array([float(reward)], dtype=np.float32)
        terminated = np.array([bool(terminated)], dtype=np.bool_)
        truncated = np.array([bool(truncated)], dtype=np.bool_)
        return self.prepare_obs(obs), reward, terminated, truncated, info
