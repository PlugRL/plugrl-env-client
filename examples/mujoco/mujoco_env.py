import dataclasses
import gymnasium as gym
import numpy as np
from plugrl_env_client.envs.base_env import (
    Action,
    BaseEnv,
    BaseEnvConfig,
    BoolArray,
    Observation,
    RewardArray,
)
from plugrl_env_client.utils.registration import register_env, register_env_config

UID = "MuJoCo-v1"


@register_env_config(UID)
@dataclasses.dataclass
class MuJoCoConfig(BaseEnvConfig):
    name: str = "Hopper-v4"


@register_env(UID)
class MuJoCoEnv(BaseEnv):
    env: gym.Env

    def __init__(
        self,
        config: MuJoCoConfig,
        process_id: int | None = None,
        total_processes: int | None = None,
    ):
        super().__init__(
            config=config,
            process_id=process_id,
            total_processes=total_processes,
        )
        env = gym.make(config.name, render_mode="rgb_array")
        self.env = env
        self.task_name = config.name
        print(env.observation_space)
        print(env.action_space)

    def prepare_obs(self, obs: np.ndarray) -> Observation:
        frame = self.env.render()
        frames = {"env": np.array(frame)[None, ...]}
        states = {"obs": obs[None, ...]}
        return Observation(
            images=frames,
            states=states,
            text=self.task_name,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.prepare_obs(obs), info

    def step(
        self, actions: Action
    ) -> tuple[Observation, RewardArray, BoolArray, BoolArray, dict]:
        obs, reward, terminated, truncated, info = self.env.step(actions[0])
        return (
            self.prepare_obs(obs),
            np.asarray([float(reward)], dtype=np.float32),
            np.asarray([bool(terminated)], dtype=np.bool_),
            np.asarray([bool(truncated)], dtype=np.bool_),
            info,
        )


if __name__ == "__main__":
    from plugrl_env_client.cli import main

    main()
