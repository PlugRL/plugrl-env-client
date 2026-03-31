from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class VectorEpisodeStatsWrapper:
    """Adds `info['episode']` stats (r, l, s, mask) for VectorEnv-like envs.

    This wrapper is intentionally lightweight and only relies on the wrapped env
    exposing `num_envs`, `reset(...) -> (obs, info)` and
    `step(actions) -> (obs, reward, terminated, truncated, info)`.

    - r: episode return (sum of rewards)
    - l: episode length (number of steps)
    - s: episode success (derived from per-step reward threshold)

    Notes:
        - Success is defined as: any step reward >= best_reward_threshold_for_success
          within the episode.
        - If `best_reward_threshold_for_success` is None, success is always False.
        - Partial reset via `reset(options={'reset_indices': ...})` is supported
          for the internal counters.
    """

    def __init__(
        self,
        env,
        *,
        best_reward_threshold_for_success: float | None = None,
        deque_size: int = 100,
    ):
        self.env = env
        self._plugrl_episode_stats_wrapper = True
        self.best_reward_threshold_for_success = best_reward_threshold_for_success
        self._success_queue: deque[bool] = deque(maxlen=int(deque_size))

        num_envs = int(getattr(env, "num_envs", 1))
        self._episode_returns = np.zeros((num_envs,), dtype=np.float32)
        self._episode_lengths = np.zeros((num_envs,), dtype=np.int32)
        self._episode_has_succeeded = np.zeros((num_envs,), dtype=np.bool_)

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def _reset_counters(self, indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        self._episode_returns[indices] = 0.0
        self._episode_lengths[indices] = 0
        self._episode_has_succeeded[indices] = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        reset_indices = None
        if options is not None:
            reset_indices = options.get("reset_indices")

        if reset_indices is None:
            self._reset_counters(np.arange(self._episode_returns.shape[0]))
        else:
            indices = np.asarray(reset_indices, dtype=np.int64)
            self._reset_counters(indices)

        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)

        reward_arr = np.asarray(reward, dtype=np.float32)
        terminated_arr = np.asarray(terminated, dtype=np.bool_)
        truncated_arr = np.asarray(truncated, dtype=np.bool_)
        done = np.logical_or(terminated_arr, truncated_arr)

        if reward_arr.shape[:1] != self._episode_returns.shape:
            raise ValueError(
                f"Expected reward shape {self._episode_returns.shape}, got {reward_arr.shape}"
            )

        self._episode_returns += reward_arr
        self._episode_lengths += 1

        if self.best_reward_threshold_for_success is not None:
            is_step_success = reward_arr >= float(
                self.best_reward_threshold_for_success
            )
        else:
            is_step_success = np.zeros_like(done, dtype=np.bool_)

        self._episode_has_succeeded = np.logical_or(
            self._episode_has_succeeded, is_step_success
        )

        # Keep this key consistent with RecordSuccessByStep.
        if isinstance(info, dict):
            info["is_step_success"] = is_step_success

        if np.any(done):
            done_indices = np.nonzero(done)[0]
            # Keep episode stats full-length so downstream env slicing can stay
            # aligned with per-env feedback indices.
            ep_r = np.zeros_like(self._episode_returns, dtype=np.float32)
            ep_l = np.zeros_like(self._episode_lengths, dtype=np.int32)
            ep_s = np.zeros_like(self._episode_has_succeeded, dtype=np.bool_)
            ep_mask = np.zeros_like(self._episode_has_succeeded, dtype=np.bool_)
            ep_r[done_indices] = self._episode_returns[done_indices]
            ep_l[done_indices] = self._episode_lengths[done_indices]
            ep_s[done_indices] = self._episode_has_succeeded[done_indices]
            ep_mask[done_indices] = True

            for s in self._episode_has_succeeded[done_indices].tolist():
                self._success_queue.append(bool(s))
            mean_success_rate = (
                float(np.mean(self._success_queue)) if self._success_queue else 0.0
            )

            if isinstance(info, dict):
                info_episode = info.setdefault("episode", {})
                info_episode.update(
                    {
                        "r": ep_r,
                        "l": ep_l,
                        "s": ep_s,
                        "mask": ep_mask,
                        "mean_success_rate": mean_success_rate,
                    }
                )

        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None
