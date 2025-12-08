import numpy as np
import dataclasses

from .base_env import BaseEnv, Observation, Action, BaseEnvConfig
from vlarl_infra.utils.registration import register_env, register_env_config

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
    
    def __init__(self, config: DummyEnvConfig, worker_id: int | None = None, total_workers: int | None = None):
        super().__init__(config=config)
        self.img_width = config.img_width
        self.img_height = config.img_height
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        self.text = config.text
        self.terminated_prob = config.terminated_prob

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        return self.fake_obs(), {}

    def step(self, action: Action) -> tuple:
        reward = np.random.rand()
        terminated = np.random.rand() < self.terminated_prob
        return self.fake_obs(), reward, terminated, False, {}

    def fake_action(self) -> Action:
        return np.random.rand(1, self.action_dim).astype(np.float32)
    
    def fake_obs(self) -> Observation:
        obs = Observation(
            images={
                "base": np.random.randint(0, 255, (1, self.img_height, self.img_width, 3)).astype(np.uint8),
                "wrist": np.random.randint(0, 255, (1, self.img_height // 2, self.img_width // 2, 3)).astype(np.uint8),
            },
            states={
                "robot_state": np.random.rand(1, self.state_dim),
                "joint_angles": np.random.rand(1, self.state_dim // 2)
            },
            text=self.text,
        )
        return obs