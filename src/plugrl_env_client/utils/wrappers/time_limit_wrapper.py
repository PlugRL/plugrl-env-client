from __future__ import annotations

from typing import Any

import numpy as np


class TimeLimitWrapper:
    def __init__(self, env, max_episode_steps: int):
        self.env = env
        self.max_episode_steps = int(max_episode_steps)

        self.num_envs = int(getattr(env, "num_envs", 1))

        self._elapsed_steps = np.zeros((self.num_envs,), dtype=np.int32)

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def _reset_counters(self, indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        self._elapsed_steps[indices] = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        reset_indices = None
        if options is not None:
            reset_indices = options.get("reset_indices")

        if reset_indices is None:
            self._reset_counters(np.arange(self.num_envs))
        else:
            indices = np.asarray(reset_indices, dtype=np.int64)
            self._reset_counters(indices)

        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)

        terminated_arr = np.asarray(terminated, dtype=np.bool_)
        truncated_arr = np.asarray(truncated, dtype=np.bool_)

        self._elapsed_steps += 1

        timeouts = self._elapsed_steps >= self.max_episode_steps

        new_truncated = np.logical_or(truncated_arr, timeouts)

        if isinstance(info, dict):
            info["TimeLimit.truncated"] = np.logical_and(timeouts, ~terminated_arr)

        done = np.logical_or(terminated_arr, new_truncated)

        if np.any(done):
            done_indices = np.nonzero(done)[0]
            self._reset_counters(done_indices)

        if isinstance(truncated, (bool, np.bool_)) and self.num_envs == 1:
            new_truncated = bool(new_truncated[0])

        return obs, reward, terminated, new_truncated, info

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None
