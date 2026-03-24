import sys
import json
from functools import partial
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np

from plugrl_env_client.recorder import Recorder, RecorderArgs
from plugrl_env_client.runner.args import RunnerArgs
from plugrl_env_client.runner.rollout import rollout
from plugrl_env_client.runner.run import _make_env
from plugrl_env_client.envs.base_env import Observation
from plugrl_env_client.agent.websocket_env_client_agent import WebSocketEnvClientAgent


class _FakeAgent:
    def __init__(self, action_chunk: np.ndarray):
        self._action_chunk = action_chunk
        self.infer_calls: list[np.ndarray] = []
        self.feedback_calls: list[dict] = []

    def infer(self, obs: dict, *, env_indices: np.ndarray, step_ids: np.ndarray):
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
        step_ids: np.ndarray,
    ) -> None:
        self.feedback_calls.append(
            {
                "env_indices": np.asarray(env_indices),
                "rewards": np.asarray(rewards),
                "terminated": np.asarray(terminated),
                "truncated": np.asarray(truncated),
                "info": info,
                "step_ids": np.asarray(step_ids),
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


def test_cli_main_parses_and_calls_run(monkeypatch, tmp_path):
    import plugrl_env_client.cli as cli

    called: dict[str, object] = {}

    def fake_run(args, agent_factory, **kwargs):
        called["args"] = args
        called["agent_factory"] = agent_factory
        called["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "dummy-v1",
            "--num-episodes",
            "2",
            "--num-envs",
            "3",
            "--server-host",
            "127.0.0.1",
            "--server-port",
            "9001",
            "--reconnect-on-server-stop",
        ],
    )

    cli.main()

    args = cast(RunnerArgs, called["args"])
    assert args.uid == "Dummy-v1"
    kwargs = cast(dict[str, object], called["kwargs"])
    assert kwargs["num_episodes"] == 2
    assert kwargs["num_envs"] == 3
    assert kwargs["env_config"] is not None
    assert isinstance(kwargs["exp_name"], str)
    assert cast(Path, kwargs["output_dir"]).name == cast(str, kwargs["exp_name"])
    config_path = cast(Path, kwargs["output_dir"]) / "client_config.json"
    assert config_path.exists()
    agent_factory = cast(partial, called["agent_factory"])
    assert agent_factory.keywords["host"] == "127.0.0.1"
    assert agent_factory.keywords["port"] == 9001
    assert agent_factory.keywords["reconnect_on_server_stop"] is True


def test_cli_main_keeps_top_level_env_for_runtime(monkeypatch, tmp_path):
    import plugrl_env_client.cli as cli

    called: dict[str, object] = {}

    def fake_run(args, _agent_factory, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "d4rl-v1",
            "--env.name",
            "walker2d-medium-v2",
            "--env.use-image",
        ],
    )

    cli.main()

    args = cast(RunnerArgs, called["args"])
    assert args.uid == "D4RL-v1"
    kwargs = cast(dict[str, object], called["kwargs"])
    env_config = kwargs["env_config"]
    assert env_config.name == "walker2d-medium-v2"
    assert env_config.use_image is True


def test_cli_main_num_procs_dispatches_to_multiprocess(monkeypatch, tmp_path):
    import plugrl_env_client.cli as cli

    called: dict[str, object] = {}

    def fake_run(_args, _agent_factory, **kwargs):
        called["run"] = True

    def fake_run_multiprocess(_args, agent_factory, **kwargs):
        called["mp"] = kwargs["num_procs"]
        called["agent_factory"] = agent_factory
        called["kwargs"] = kwargs

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "run_multiprocess", fake_run_multiprocess)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "dummy-v1", "--num-procs", "2", "--server-port", "8123"],
    )

    cli.main()

    assert "mp" in called
    assert called["mp"] == 2
    assert "run" not in called
    kwargs = cast(dict[str, object], called["kwargs"])
    assert isinstance(kwargs["recorder_args"], RecorderArgs)
    assert kwargs["num_procs"] == 2
    agent_factory = cast(partial, called["agent_factory"])
    assert agent_factory.keywords["host"] == "0.0.0.0"
    assert agent_factory.keywords["port"] == 8123


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


def test_make_env_uses_vector_entry_point_kwargs(monkeypatch):
    called: dict[str, object] = {}

    def fake_make_vec(env_id, **kwargs):
        called["env_id"] = env_id
        called["kwargs"] = kwargs
        return cast(gym.vector.VectorEnv, object())

    monkeypatch.setattr(gym, "make_vec", fake_make_vec)

    env_config = cast(object, type("Cfg", (), {})())
    runner_args = RunnerArgs(uid="Dummy-v1")
    runner_args.max_episode_steps = 17

    _make_env(
        runner_args,
        env_config,
        3,
        process_id=2,
        total_processes=4,
        env_lock=None,
    )

    assert called["env_id"] == "Dummy-v1"
    kwargs = cast(dict[str, object], called["kwargs"])
    assert kwargs["num_envs"] == 3
    assert kwargs["vectorization_mode"] == "vector_entry_point"
    assert kwargs["config"] is env_config
    assert kwargs["max_episode_steps"] == 17
    assert kwargs["process_id"] == 2
    assert kwargs["total_processes"] == 4
    assert "vector_kwargs" not in kwargs


def test_recorder_writes_fieldwise_obs_outputs(tmp_path):
    recorder = Recorder(
        RecorderArgs(
            episode_freq=1,
            record_obs_stats=True,
            record_episode_metrics=True,
            record_video=False,
        ),
        exp_name="exp",
        output_dir=tmp_path,
        num_envs=1,
        process_id=None,
        total_processes=None,
    )

    first_obs = Observation(
        images={"img": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
        states={"state": np.ones((1, 3), dtype=np.float32)},
        text=np.asarray(["hello"], dtype=np.str_),
    )
    last_obs = Observation(
        images={"img": np.full((1, 2, 2, 3), 255, dtype=np.uint8)},
        states={"state": np.full((1, 3), 2.0, dtype=np.float32)},
        text=np.asarray(["done"], dtype=np.str_),
    )

    recorder.on_reset(first_obs, {}, reset_indices=None)
    recorder.on_episode_done(
        np.asarray([0], dtype=np.int64),
        last_obs,
        {
            "episode": {
                "r": np.asarray([3.5], dtype=np.float32),
                "s": np.asarray([True]),
            },
            "bar": np.asarray([9], dtype=np.int32),
        },
    )
    recorder.close()

    sample_dir = (
        tmp_path / "rollout" / "proc_000" / "sampled" / "ep_000001" / "env_000" / "obs"
    )
    assert (sample_dir / "first" / "images" / "img.png").exists()
    assert (sample_dir / "last" / "images" / "img.png").exists()
    assert (sample_dir / "first" / "states" / "state.npy").exists()
    assert (sample_dir / "last" / "states" / "state.npy").exists()
    assert (sample_dir / "first" / "text.txt").read_text(
        encoding="utf-8"
    ).strip() == "hello"
    assert (sample_dir / "last" / "text.txt").read_text(
        encoding="utf-8"
    ).strip() == "done"
    last_info = json.loads(
        (sample_dir / "last" / "info.json").read_text(encoding="utf-8")
    )
    assert last_info["bar"] == 9
    assert last_info["episode"]["r"] == 3.5
    obs_stats_lines = (
        (tmp_path / "rollout" / "proc_000" / "metrics" / "obs_stats.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    obs_stats = json.loads(obs_stats_lines[-1])
    assert obs_stats["last_info"]["bar"] == 9
    assert (
        tmp_path / "rollout" / "proc_000" / "metrics" / "episode_metrics.jsonl"
    ).exists()
    assert (tmp_path / "rollout" / "proc_000" / "metrics" / "obs_stats.jsonl").exists()


def test_recorder_nonzero_process_writes_under_own_proc_dir(tmp_path):
    recorder = Recorder(
        RecorderArgs(
            episode_freq=1,
            thread0_only=False,
            record_obs_stats=False,
            record_episode_metrics=True,
            record_video=False,
        ),
        exp_name="exp",
        output_dir=tmp_path,
        num_envs=1,
        process_id=2,
        total_processes=4,
    )

    first_obs = Observation(
        images={"img": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
        states={"state": np.ones((1, 3), dtype=np.float32)},
        text=np.asarray(["hello"], dtype=np.str_),
    )

    recorder.on_reset(first_obs, {}, reset_indices=None)
    recorder.on_episode_done(
        np.asarray([0], dtype=np.int64),
        first_obs,
        {
            "episode": {
                "r": np.asarray([1.0], dtype=np.float32),
                "s": np.asarray([False]),
            }
        },
    )
    recorder.close()

    proc_dir = tmp_path / "rollout" / "proc_002"
    assert (proc_dir / "manifest.json").exists()
    assert (proc_dir / "metrics" / "episode_metrics.jsonl").exists()


def test_recorder_metric_window_uses_recent_episodes():
    recorder = Recorder(
        RecorderArgs(
            episode_freq=1,
            record_obs_stats=False,
            record_episode_metrics=False,
            record_video=False,
            metric_window=2,
        ),
        exp_name="exp",
        output_dir=Path("/tmp/recorder-metric-window"),
        num_envs=1,
        process_id=None,
        total_processes=None,
    )

    obs = Observation(
        images={"img": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
        states={"state": np.zeros((1, 1), dtype=np.float32)},
        text=np.asarray(["x"], dtype=np.str_),
    )

    recorder.on_reset(obs, {}, reset_indices=None)
    recorder.on_episode_done(
        np.asarray([0], dtype=np.int64),
        obs,
        {
            "episode": {
                "r": np.asarray([1.0], dtype=np.float32),
                "s": np.asarray([False]),
            }
        },
    )
    recorder.on_reset(obs, {}, reset_indices=np.asarray([0], dtype=np.int64))
    recorder.on_episode_done(
        np.asarray([0], dtype=np.int64),
        obs,
        {
            "episode": {
                "r": np.asarray([3.0], dtype=np.float32),
                "s": np.asarray([True]),
            }
        },
    )
    recorder.on_reset(obs, {}, reset_indices=np.asarray([0], dtype=np.int64))
    recorder.on_episode_done(
        np.asarray([0], dtype=np.int64),
        obs,
        {
            "episode": {
                "r": np.asarray([5.0], dtype=np.float32),
                "s": np.asarray([True]),
            }
        },
    )

    assert recorder._mean_return() == 4.0
    assert recorder._mean_success_rate() == 1.0
