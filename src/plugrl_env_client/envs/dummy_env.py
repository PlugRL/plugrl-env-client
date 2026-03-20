import numpy as np
import dataclasses

from .base_env import BaseEnv, Observation, Action, BaseEnvConfig
from plugrl_env_client.utils.registration import register_env, register_env_config

UID = "Dummy-v1"


@register_env_config(UID)
@dataclasses.dataclass
class DummyEnvConfig(BaseEnvConfig):
    img_width: int = 224
    img_height: int = 224
    action_dim: int = 7
    state_dim: int = 10
    text: str = "do something"
    terminated_prob: float = 0.01


@register_env(UID, max_episode_steps=200000)
class DummyEnv(BaseEnv):
    img_height: int
    img_width: int
    action_dim: int
    state_dim: int
    text: str
    terminated_prob: float = 0.01
    _obs: Observation | None = None

    def __init__(
        self,
        config: DummyEnvConfig,
        worker_id: int | None = None,
        total_workers: int | None = None,
    ):
        super().__init__(config=config)
        self.img_width = config.img_width
        self.img_height = config.img_height
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        self.text = config.text
        self.terminated_prob = config.terminated_prob

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        if self._obs is None:
            self._obs = self.fake_obs()

        reset_indices = None
        if options is not None:
            reset_indices = options.get("reset_indices")

        if reset_indices is None:
            self._obs = self.fake_obs()
            return self._obs, {}

        indices = np.asarray(reset_indices, dtype=np.int64)
        if indices.size == 0:
            return self._obs, {}

        new_obs = self.fake_obs()
        assert self._obs is not None
        for key in self._obs.images:
            self._obs.images[key][indices] = new_obs.images[key][indices]
        for key in self._obs.states:
            self._obs.states[key][indices] = new_obs.states[key][indices]
        self._obs.text[indices] = new_obs.text[indices]
        return self._obs, {}

    def step(self, actions: Action) -> tuple:
        reward = np.random.rand(self.num_envs).astype(np.float32)
        terminated = (np.random.rand(self.num_envs) < self.terminated_prob).astype(
            np.bool_
        )
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)
        obs = self.fake_obs()
        self._obs = obs
        return obs, reward, terminated, truncated, {}

    def fake_action(self) -> Action:
        return np.random.rand(self.num_envs, self.action_dim).astype(np.float32)

    def fake_obs(self) -> Observation:
        b = self.num_envs
        obs = Observation(
            images={
                "base": np.random.randint(
                    0, 255, (b, self.img_height, self.img_width, 3)
                ).astype(np.uint8),
                "wrist": np.random.randint(
                    0, 255, (b, self.img_height // 2, self.img_width // 2, 3)
                ).astype(np.uint8),
            },
            states={
                "robot_state": np.random.rand(b, self.state_dim),
                "joint_angles": np.random.rand(b, self.state_dim // 2),
            },
            text=self.text,
        )
        return obs
