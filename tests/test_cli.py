import sys
from typing import cast

import gymnasium as gym
import numpy as np

from plugrl_env_client.cli_rollout import rollout
from plugrl_env_client.envs.base_env import Observation
from plugrl_env_client.websocket_env_client_agent import WebSocketEnvClientAgent


class _FakeAgent:
    def __init__(self, action_chunk: np.ndarray):
        self._action_chunk = action_chunk
        self.infer_calls: list[np.ndarray] = []
        self.feedback_calls: list[dict] = []

    def infer(self, obs: dict, env_indices: np.ndarray):
        self.infer_calls.append(np.asarray(env_indices))
        return {"action": self._action_chunk}

    def feedback(
        self,
        *,
        obs: dict,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: dict,
        env_indices: np.ndarray,
    ) -> None:
        self.feedback_calls.append(
            {
                "env_indices": np.asarray(env_indices),
                "rewards": np.asarray(rewards),
                "terminated": np.asarray(terminated),
                "truncated": np.asarray(truncated),
                "info": info,
            }
        )


class _FakeVecEnv:
    def __init__(
        self,
        *,
        num_envs: int,
        action_dim: int,
        terminated_mask: np.ndarray,
    ) -> None:
        self.num_envs = int(num_envs)
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(action_dim),),
            dtype=np.float32,
        )

        self._terminated_mask = np.asarray(terminated_mask, dtype=np.bool_)
        if self._terminated_mask.shape != (self.num_envs,):
            raise ValueError("terminated_mask must have shape (num_envs,)")

        self.reset_calls: list[dict | None] = []
        self.step_calls: list[np.ndarray] = []

        self._obs = Observation(
            images={"img": np.zeros((self.num_envs, 1, 1, 3), dtype=np.uint8)},
            states={"state": np.zeros((self.num_envs, 1), dtype=np.float32)},
            text="task",
        )

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append(options)
        info = {"foo": np.arange(self.num_envs, dtype=np.int32)}
        return self._obs, info

    def step(self, actions):
        actions_arr = np.asarray(actions, dtype=np.float32)
        self.step_calls.append(actions_arr)

        obs = self._obs
        reward = np.arange(1, self.num_envs + 1, dtype=np.float32)
        terminated = self._terminated_mask.copy()
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)
        info = {"bar": np.arange(10, 10 + self.num_envs, dtype=np.int32)}
        return obs, reward, terminated, truncated, info


def test_cli_main_parses_and_calls_run(monkeypatch):
    import plugrl_env_client.cli as cli

    called: dict[str, object] = {}

    def fake_run(args):
        called["args"] = args

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["prog", "dummy-v1", "--num-episodes", "2"])

    cli.main()

    args = called["args"]
    assert getattr(args, "uid") == "Dummy-v1"
    assert getattr(args, "num_episodes") == 2


def test_rollout_infer_feedback_and_partial_reset_semantics():
    num_envs = 2
    action_dim = 3

    # One-step plan for all envs -> both envs exhaust plan after 1 step.
    action_chunk = np.zeros((1, num_envs, action_dim), dtype=np.float32)

    env = _FakeVecEnv(
        num_envs=num_envs,
        action_dim=action_dim,
        terminated_mask=np.asarray([True, False], dtype=np.bool_),
    )
    agent = _FakeAgent(action_chunk)

    rollout(
        cast(gym.vector.VectorEnv, env),
        cast(WebSocketEnvClientAgent, agent),
        num_episodes=1,
        replan_steps=1,
        num_envs=num_envs,
    )

    assert len(agent.infer_calls) == 1
    np.testing.assert_array_equal(agent.infer_calls[0], np.asarray([0, 1]))

    assert len(env.step_calls) == 1
    np.testing.assert_array_equal(
        env.step_calls[0], np.zeros((num_envs, action_dim), dtype=np.float32)
    )

    assert len(agent.feedback_calls) == 1
    fb = agent.feedback_calls[0]
    np.testing.assert_array_equal(fb["env_indices"], np.asarray([0, 1]))
    np.testing.assert_allclose(fb["rewards"], np.asarray([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(
        fb["terminated"], np.asarray([True, False], dtype=np.bool_)
    )
    np.testing.assert_array_equal(
        fb["truncated"], np.asarray([False, False], dtype=np.bool_)
    )

    # Reset is called once at start, once for done envs.
    assert env.reset_calls[0] is None
    assert env.reset_calls[1] is not None
    reset_indices = env.reset_calls[1]["reset_indices"]
    np.testing.assert_array_equal(np.asarray(reset_indices), np.asarray([0]))


def test_rollout_dynamic_plan_capacity_when_replan_steps_none():
    num_envs = 1
    action_dim = 2

    # replan_steps=None -> initial plan capacity is 0, then it should grow to fit len(action_chunk).
    action_chunk = np.asarray(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ],
        dtype=np.float32,
    )

    env = _FakeVecEnv(
        num_envs=num_envs,
        action_dim=action_dim,
        terminated_mask=np.asarray([True], dtype=np.bool_),
    )
    agent = _FakeAgent(action_chunk)

    rollout(
        cast(gym.vector.VectorEnv, env),
        cast(WebSocketEnvClientAgent, agent),
        num_episodes=1,
        replan_steps=None,
        num_envs=num_envs,
    )

    assert len(env.step_calls) == 1
    np.testing.assert_array_equal(env.step_calls[0][0], action_chunk[0, 0])
