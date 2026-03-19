import dataclasses
import numpy as np
import gym

try:
    import d4rl.gym_mujoco  # noqa: F401
except ImportError:
    raise ImportError(
        'd4rl is not installed. Please install it with pip install "plugrl-env-client[d4rl]".'
    )
except Exception as e:
    raise ImportError(f"An error occurred while importing d4rl: {e}")

from plugrl_env_client.envs.base_env import BaseEnv, BaseEnvConfig, Action, Observation
from plugrl_env_client.utils.registration import register_env, register_env_config

UID = "D4RL-v1"


@register_env_config(UID)
@dataclasses.dataclass
class D4RLConfig(BaseEnvConfig):
    env_name: str = "hopper-medium-v2"
    use_image: bool = False


@register_env(UID)
class D4RLEnv(BaseEnv):
    env: gym.Env

    def __init__(
        self,
        config: D4RLConfig,
        worker_id: int | None = None,
        total_workers: int | None = None,
    ):
        super().__init__(config=config)
        if self.num_envs != 1:
            raise ValueError("D4RLEnv only supports num_envs=1")
        env = gym.make(config.env_name)
        self.env = env
        self.task_name = config.env_name
        self.use_image = config.use_image

    def prepare_obs(self, obs: np.ndarray) -> Observation:
        if self.use_image:
            frame = self.env.render(mode="rgb_array")
            frames = {"env": np.array(frame)[None, ...]}
        else:
            frames = {}
        states = {"obs": obs[None, ...]}
        return Observation(
            images=frames,
            states=states,
            text=self.task_name,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation | None, dict]:
        obs, info = self.env.reset(seed=seed, options=options, return_info=True)
        return self.prepare_obs(obs), info

    def step(
        self, action: Action
    ) -> tuple[Observation | None, np.ndarray, np.ndarray, np.ndarray, dict]:
        if action.ndim > 1:
            action = action[0]
        obs, reward, done, info = self.env.step(action)
        reward = np.array([float(reward)], dtype=np.float32)
        terminated = np.array([bool(done)], dtype=np.bool_)
        truncated = np.array([False], dtype=np.bool_)
        return self.prepare_obs(obs), reward, terminated, truncated, info
