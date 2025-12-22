import dataclasses
import gymnasium as gym
import numpy as np
from plugrl_worker.envs.base_env import BaseEnv, BaseEnvConfig, Observation, Action
from plugrl_worker.utils.registration import register_env, register_env_config

UID = "MuJoCo-v1"

@register_env_config(UID)
@dataclasses.dataclass
class MuJoCoConfig(BaseEnvConfig):
    name: str = "Hopper-v4"

@register_env(UID)
class MuJoCoEnv(BaseEnv):
    env: gym.Env
    
    def __init__(self, config: MuJoCoConfig, worker_id: int | None = None, total_workers: int | None = None):
        super().__init__(config=config)
        env = gym.make(config.name, render_mode="rgb_array")
        self.env = env
        self.task_name = config.name
        print(env.observation_space)
        print(env.action_space)
        
    def prepare_obs(self, obs: np.ndarray) -> Observation:
        frame = self.env.render()
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
        obs, info = self.env.reset(seed=seed, options=options)
        return self.prepare_obs(obs), info
    
    def step(self, action: Action) -> tuple[Observation | None, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action[0])
        return self.prepare_obs(obs), float(reward), terminated, truncated, info
    
if __name__ == "__main__":
    from plugrl_worker.cli import main
    main()