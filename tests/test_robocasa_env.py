from __future__ import annotations

from typing import Any

import numpy as np

from plugrl_env_client.envs.robocasa.robocasa_env import (
    ROBOCASA_STATE_DIM,
    RobocasaConfig,
    RobocasaEnv,
    _build_state,
    _lerobot_to_hdf5_action,
)


def _make_raw_obs(token: int, *, prompt: str | None = None) -> dict[str, Any]:
    image_token = int(token) % 256
    return {
        "video.robot0_agentview_left": np.full(
            (8, 8, 3), image_token, dtype=np.uint8
        ),
        "video.robot0_agentview_right": np.full(
            (8, 8, 3), (image_token + 1) % 256, dtype=np.uint8
        ),
        "video.robot0_eye_in_hand": np.full(
            (8, 8, 3), (image_token + 2) % 256, dtype=np.uint8
        ),
        "state.end_effector_position_relative": np.asarray(
            [token + 0, token + 1, token + 2], dtype=np.float32
        ),
        "state.end_effector_rotation_relative": np.asarray(
            [token + 3, token + 4, token + 5, token + 6], dtype=np.float32
        ),
        "state.base_position": np.asarray(
            [token + 7, token + 8, token + 9], dtype=np.float32
        ),
        "state.base_rotation": np.asarray(
            [token + 10, token + 11, token + 12, token + 13], dtype=np.float32
        ),
        "state.gripper_qpos": np.asarray([token + 14, token + 15], dtype=np.float32),
        "annotation.human.task_description": prompt or f"task-{token}",
    }


class _FakeRoboCasaGymEnv:
    def __init__(self, env_idx: int):
        self.env_idx = env_idx
        self.reset_count = 0
        self.step_count = 0
        self.recorded_actions: list[dict[str, np.ndarray]] = []

    def reset(self, seed=None):
        del seed
        token = self.env_idx * 100 + self.reset_count
        self.reset_count += 1
        return _make_raw_obs(token), {"success": False}

    def step(self, action_dict):
        self.recorded_actions.append(action_dict)
        token = self.env_idx * 1000 + self.step_count + 1
        self.step_count += 1
        return (
            _make_raw_obs(token),
            float(self.env_idx + 1),
            bool(self.step_count >= 1 and self.env_idx == 0),
            False,
            {"success": self.env_idx == 0},
        )

    def close(self):
        return None


def test_build_state_matches_openpi_order():
    raw_obs = _make_raw_obs(10)
    state = _build_state(raw_obs)

    assert state.shape == (ROBOCASA_STATE_DIM,)
    np.testing.assert_array_equal(
        state,
        np.asarray(
            [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
            ],
            dtype=np.float32,
        ),
    )


def test_lerobot_to_hdf5_action_reorders_expected_fields():
    action = np.arange(12, dtype=np.float32)
    reordered = _lerobot_to_hdf5_action(action)
    np.testing.assert_array_equal(
        reordered,
        np.asarray([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], dtype=np.float32),
    )


def test_robocasa_env_partial_reset_and_passthrough_actions(monkeypatch):
    created_envs: list[_FakeRoboCasaGymEnv] = []

    def fake_make_env(*, task_name: str, split: str, seed: int):
        del task_name, split, seed
        env = _FakeRoboCasaGymEnv(len(created_envs))
        created_envs.append(env)
        return env

    monkeypatch.setattr(
        "plugrl_env_client.envs.robocasa.robocasa_env._import_robocasa_dependencies",
        lambda: (lambda action: {"action": np.asarray(action, dtype=np.float32)}),
    )
    monkeypatch.setattr(
        "plugrl_env_client.envs.robocasa.robocasa_env._make_robocasa_env",
        fake_make_env,
    )

    env = RobocasaEnv(RobocasaConfig(resize_size=8), num_envs=2)
    obs, _ = env.reset()
    first_env0 = obs.states["state"][0].copy()
    first_env1 = obs.states["state"][1].copy()

    next_obs, reward, terminated, truncated, info = env.step(
        np.asarray(
            [
                np.arange(12, dtype=np.float32),
                np.arange(12, dtype=np.float32) + 10.0,
            ]
        )
    )

    np.testing.assert_allclose(reward, np.asarray([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(
        terminated, np.asarray([True, False], dtype=np.bool_)
    )
    np.testing.assert_array_equal(truncated, np.asarray([False, False], dtype=np.bool_))
    assert "success" in info
    np.testing.assert_array_equal(
        created_envs[0].recorded_actions[0]["action"], np.arange(12, dtype=np.float32)
    )

    reset_obs, _ = env.reset(options={"reset_indices": np.asarray([1], dtype=np.int64)})
    np.testing.assert_array_equal(reset_obs.states["state"][0], next_obs.states["state"][0])
    assert not np.array_equal(reset_obs.states["state"][1], first_env1)
    assert not np.array_equal(reset_obs.states["state"][0], first_env0)


def test_robocasa_env_lerobot_action_encoding(monkeypatch):
    created_envs: list[_FakeRoboCasaGymEnv] = []

    def fake_make_env(*, task_name: str, split: str, seed: int):
        del task_name, split, seed
        env = _FakeRoboCasaGymEnv(len(created_envs))
        created_envs.append(env)
        return env

    monkeypatch.setattr(
        "plugrl_env_client.envs.robocasa.robocasa_env._import_robocasa_dependencies",
        lambda: (lambda action: {"action": np.asarray(action, dtype=np.float32)}),
    )
    monkeypatch.setattr(
        "plugrl_env_client.envs.robocasa.robocasa_env._make_robocasa_env",
        fake_make_env,
    )

    env = RobocasaEnv(
        RobocasaConfig(resize_size=8, action_encoding="lerobot_to_hdf5"),
        num_envs=1,
    )
    env.reset()
    env.step(np.arange(12, dtype=np.float32))
    np.testing.assert_array_equal(
        created_envs[0].recorded_actions[0]["action"],
        np.asarray([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4], dtype=np.float32),
    )
