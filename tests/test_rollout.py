from __future__ import annotations

import numpy as np

from plugrl_env_client.envs.base_env import Observation
from plugrl_env_client.runner.rollout import rollout


class _OneStepDoneEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.single_action_space = type(
            "_Space", (), {"shape": (1,), "dtype": np.dtype(np.float32)}
        )()
        self.reset_calls: list[np.ndarray | None] = []
        self.step_calls = 0

    def reset(self, *, seed=None, options=None):
        del seed
        reset_indices = None if options is None else options.get("reset_indices")
        self.reset_calls.append(
            None
            if reset_indices is None
            else np.asarray(reset_indices, dtype=np.int64).copy()
        )
        obs = Observation(
            images={"cam": np.zeros((1, 4, 4, 3), dtype=np.uint8)},
            states={"state": np.zeros((1, 2), dtype=np.float32)},
            text=np.asarray(["task"], dtype=np.str_),
        )
        return obs, {}

    def step(self, actions):
        del actions
        self.step_calls += 1
        obs = Observation(
            images={"cam": np.full((1, 4, 4, 3), self.step_calls, dtype=np.uint8)},
            states={"state": np.full((1, 2), self.step_calls, dtype=np.float32)},
            text=np.asarray(["task"], dtype=np.str_),
        )
        reward = np.asarray([1.0], dtype=np.float32)
        terminated = np.asarray([True], dtype=np.bool_)
        truncated = np.asarray([False], dtype=np.bool_)
        info = {
            "episode": {
                "r": np.asarray([1.0], dtype=np.float32),
                "l": np.asarray([1], dtype=np.int32),
                "s": np.asarray([True], dtype=np.bool_),
                "mask": np.asarray([True], dtype=np.bool_),
                "mean_success_rate": 1.0,
            }
        }
        return obs, reward, terminated, truncated, info


class _OneChunkAgent:
    def infer(self, obs, *, env_indices, step_ids):
        del obs, env_indices, step_ids
        return {"action": np.zeros((1, 1, 1), dtype=np.float32)}

    def feedback(self, **kwargs):
        del kwargs
        return None


class _SpyRecorder:
    def __init__(self) -> None:
        self.debug_packets_enabled = False
        self.reset_calls = 0
        self.done_calls = 0

    def on_reset(self, obs, info, *, reset_indices):
        del obs, info, reset_indices
        self.reset_calls += 1

    def on_step(self, next_obs, reward, terminated, truncated, info):
        del next_obs, reward, terminated, truncated, info

    def on_episode_done(self, done_indices, obs, info):
        del done_indices, obs, info
        self.done_calls += 1

    def record_timing(self, timing):
        del timing


def test_rollout_does_not_reset_after_final_episode():
    env = _OneStepDoneEnv()
    agent = _OneChunkAgent()
    recorder = _SpyRecorder()

    rollout(
        env,
        agent,
        num_episodes=1,
        replan_steps=1,
        num_envs=1,
        recorder=recorder,
    )

    assert len(env.reset_calls) == 1
    assert env.reset_calls[0] is None
    assert recorder.reset_calls == 1
    assert recorder.done_calls == 1
