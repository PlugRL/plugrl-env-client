import gymnasium as gym
from typing import Any, SupportsFloat, TypeVar
from collections import deque
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class RecordSuccessByStep(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        best_reward_threshold_for_success: float | None = None,
        deque_size: int = 100
    ):
        super().__init__(env)
        
        self.best_reward_threshold_for_success = best_reward_threshold_for_success
        self.episode_success_queue = deque(maxlen=deque_size)
        self.current_episode_has_succeeded = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.current_episode_has_succeeded = False
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        is_step_success = float(reward) >= self.best_reward_threshold_for_success if self.best_reward_threshold_for_success is not None else False
        info["is_step_success"] = is_step_success
        
        if is_step_success:
            self.current_episode_has_succeeded = True
            
        if terminated or truncated:
            is_episode_success = self.current_episode_has_succeeded
            self.episode_success_queue.append(is_episode_success)
            mean_success_rate = float(np.mean(self.episode_success_queue))
            
            info["episode"].update({
                "s": is_episode_success,
                "mean_success_rate": mean_success_rate,
            })

        return observation, reward, terminated, truncated, info