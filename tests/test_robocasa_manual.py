from __future__ import annotations

import functools
import importlib
import http.client
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import plugrl_env_client.envs  # noqa: F401
from plugrl_env_client.agent.websocket_env_client_agent import WebSocketEnvClientAgent
from plugrl_env_client.envs.robocasa.robocasa_env import RobocasaConfig
from plugrl_env_client.recorder import RecorderArgs
from plugrl_env_client.runner.args import RunnerArgs
from plugrl_env_client.runner.run import run, run_multiprocess

MANUAL_TEST_MAX_EPISODE_STEPS = int(
    os.environ.get("PLUGRL_ROBOCASA_MANUAL_MAX_EPISODE_STEPS", "100")
)
MANUAL_TEST_NUM_EPISODES = int(
    os.environ.get("PLUGRL_ROBOCASA_MANUAL_NUM_EPISODES", "1")
)
MANUAL_TEST_NUM_ENVS = int(
    os.environ.get("PLUGRL_ROBOCASA_MANUAL_NUM_ENVS", "8")
)
MANUAL_TEST_NUM_PROCS = int(
    os.environ.get("PLUGRL_ROBOCASA_MANUAL_NUM_PROCS", str(MANUAL_TEST_NUM_ENVS))
)
MANUAL_TEST_NUM_ENVS_PER_PROC = int(
    os.environ.get("PLUGRL_ROBOCASA_MANUAL_NUM_ENVS_PER_PROC", "1")
)
MANUAL_TEST_SERVER_WAIT_SECONDS = float(
    os.environ.get("OPENPI_ROBOCASA_SERVER_WAIT_SECONDS", "60")
)
MANUAL_TEST_TASK_NAME = os.environ.get("OPENPI_ROBOCASA_TASK_NAME", "SearingMeat")
MANUAL_TEST_SPLIT = os.environ.get("OPENPI_ROBOCASA_SPLIT", "target")
MANUAL_TEST_THREAD0_ONLY = os.environ.get(
    "PLUGRL_ROBOCASA_MANUAL_THREAD0_ONLY", "0"
).lower() in {"1", "true", "yes", "on"}


def _have_real_robocasa() -> bool:
    try:
        importlib.import_module("robocasa")
        importlib.import_module("robocasa.wrappers.gym_wrapper")
    except Exception:
        return False
    return True


def _have_openpi_manual_inputs() -> bool:
    required = (
        "OPENPI_ROBOCASA_CONFIG_NAME",
        "OPENPI_ROBOCASA_CHECKPOINT_DIR",
        "OPENPI_ROBOCASA_DATASET_DIR",
    )
    return all(os.environ.get(key) for key in required)


def _openpi_server_python() -> str:
    return os.environ.get(
        "OPENPI_ROBOCASA_SERVER_PYTHON", "/mnt/openpi-base/.venv/bin/python"
    )


def _manual_robocasa_config() -> RobocasaConfig:
    return RobocasaConfig(task_name=MANUAL_TEST_TASK_NAME, split=MANUAL_TEST_SPLIT)


def _wait_for_server(host: str, port: int, *, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        conn = None
        try:
            conn = http.client.HTTPConnection(host, port, timeout=1.0)
            conn.request("GET", "/healthz")
            response = conn.getresponse()
            if response.status == 200:
                return
        except OSError:
            time.sleep(1.0)
        finally:
            if conn is not None:
                conn.close()

    raise TimeoutError(f"Timed out waiting for plugrl OpenPI server on {host}:{port}")


class _ZeroChunkAgent:
    def __init__(self, *, action_dim: int, horizon: int, num_envs: int):
        self._action = np.zeros(
            (horizon, num_envs, action_dim), dtype=np.float32
        )

    def infer(self, obs, *, env_indices, step_ids):
        del obs, env_indices, step_ids
        return {"action": self._action}

    def feedback(self, **kwargs):
        del kwargs
        return None


def _manual_runner_args() -> RunnerArgs:
    return RunnerArgs(
        uid="Robocasa-v1",
        max_episode_steps=MANUAL_TEST_MAX_EPISODE_STEPS,
        start_method="spawn",
    )


def _manual_recorder_args() -> RecorderArgs:
    return RecorderArgs(
        episode_freq=1,
        thread0_only=MANUAL_TEST_THREAD0_ONLY,
        record_video=True,
        record_full_rollout=True,
        record_debug_packets=True,
    )


def _run_manual_rollout(*, agent_factory, exp_name: str, output_dir: Path) -> None:
    runner_args = _manual_runner_args()
    if MANUAL_TEST_NUM_PROCS > 1:
        run_multiprocess(
            runner_args,
            agent_factory,
            env_config=_manual_robocasa_config(),
            num_envs=MANUAL_TEST_NUM_ENVS_PER_PROC,
            num_episodes=MANUAL_TEST_NUM_EPISODES,
            num_procs=MANUAL_TEST_NUM_PROCS,
            exp_name=exp_name,
            output_dir=output_dir,
            recorder_args=_manual_recorder_args(),
        )
        return

    run(
        runner_args,
        agent_factory,
        env_config=_manual_robocasa_config(),
        num_envs=MANUAL_TEST_NUM_ENVS_PER_PROC,
        num_episodes=MANUAL_TEST_NUM_EPISODES,
        exp_name=exp_name,
        output_dir=output_dir,
        recorder_args=_manual_recorder_args(),
    )


@pytest.mark.manual
@pytest.mark.skipif(not _have_real_robocasa(), reason="RoboCasa is not installed yet.")
def test_manual_robocasa_parallel_rollout_smoke(tmp_path):
    _run_manual_rollout(
        agent_factory=functools.partial(
            _ZeroChunkAgent,
            action_dim=12,
            horizon=30,
            num_envs=MANUAL_TEST_NUM_ENVS_PER_PROC,
        ),
        exp_name="manual-robocasa-smoke",
        output_dir=tmp_path,
    )


@pytest.mark.manual
@pytest.mark.skipif(
    not (_have_real_robocasa() and _have_openpi_manual_inputs()),
    reason="RoboCasa or OpenPI manual inputs are not available.",
)
def test_manual_robocasa_openpi_server_roundtrip(tmp_path):
    port = 8765
    script = Path("/mnt/openpi-base/scripts/serve_policy_robocasa_plugrl.py")
    server_log_path = tmp_path / "openpi_server.log"
    server_log = server_log_path.open("wb")
    server = subprocess.Popen(
        [
            _openpi_server_python(),
            str(script),
            "--config-name",
            os.environ["OPENPI_ROBOCASA_CONFIG_NAME"],
            "--checkpoint-dir",
            os.environ["OPENPI_ROBOCASA_CHECKPOINT_DIR"],
            "--dataset-dir",
            os.environ["OPENPI_ROBOCASA_DATASET_DIR"],
            "--port",
            str(port),
        ],
        cwd="/mnt/openpi-base",
        stdout=server_log,
        stderr=server_log,
    )
    try:
        try:
            _wait_for_server(
                "127.0.0.1",
                port,
                timeout_s=MANUAL_TEST_SERVER_WAIT_SECONDS,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}. Server log: {server_log_path}"
            ) from exc

        _run_manual_rollout(
            agent_factory=functools.partial(
                WebSocketEnvClientAgent,
                host="127.0.0.1",
                port=port,
            ),
            exp_name="manual-robocasa-openpi",
            output_dir=tmp_path,
        )
    finally:
        server.terminate()
        server.wait(timeout=10)
        server_log.close()
