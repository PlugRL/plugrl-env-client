import dataclasses
import numpy as np
import gym

try:
    import d4rl.gym_mujoco
except ImportError:
    raise ImportError('d4rl is not installed. Please install it with pip install "plugrl-worker[dr4rl]".')
except Exception as e:
    raise ImportError(f"An error occurred while importing d4rl: {e}")

from plugrl_worker.envs.base_env import BaseEnv, BaseEnvConfig, Action, Observation
from plugrl_worker.utils.registration import register_env, register_env_config

UID = "D4RL-v1"

@register_env_config(UID)
@dataclasses.dataclass
class D4RLConfig(BaseEnvConfig):
    env_name: str = "hopper-medium-v2"
    
@register_env(UID)
class D4RLEnv(BaseEnv):
    env: gym.Env
    
    def __init__(self, config: D4RLConfig, worker_id: int | None = None, total_workers: int | None = None):
        super().__init__(config=config)
        env = gym.make(config.env_name)
        self.env = env
        self.task_name = config.env_name

    def prepare_obs(self, obs: np.ndarray) -> Observation:
        frame = self.env.render(mode="rgb_array")
        frames = {
            f"env": np.array(frame)[None, ...]
        }
        states = {"obs": obs[None, ...]}
        return Observation(
            images=frames,
            states=states,
            text=self.task_name,
        )
        
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Observation | None, dict]:
        obs, info = self.env.reset(seed=seed, options=options, return_info=True)
        return self.prepare_obs(obs), info
    
    def step(self, action: Action) -> tuple[Observation | None, float, bool, bool, dict]:
        if action.ndim > 1:
            action = action[0]
        obs, reward, done, info = self.env.step(action)
        return self.prepare_obs(obs), float(reward), done, False, info