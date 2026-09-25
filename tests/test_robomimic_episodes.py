"""robomimic episodes have to end.

robomimic's environments never end an episode on their own: the dataset
metadata the client ships sets `ignore_done`, `EnvRobosuite.is_done()` is
never true, and robomimic's own rollouts stop on success or at a fixed
horizon. The client did neither, so a robomimic run was one episode that
never ended - E27's pilot on NutAssemblySquare finished 0 episodes in 4,096
steps. These tests pin the two stops robomimic uses: success, and the
task's rollout horizon.
"""

import numpy as np
import pytest

robomimic_env = pytest.importorskip(
    "plugrl_env_client.envs.robomimic.robomimic_env",
    exc_type=ImportError,
    reason="requires the robomimic extra",
)
RobomimicConfig = robomimic_env.RobomimicConfig
RobomimicEnv = robomimic_env.RobomimicEnv


class _FakeRobosuite:
    hard_reset = True


class _FakeEnvRobosuite:
    """Stands in for robomimic's EnvRobosuite: never done, success on a step."""

    action_dimension = 7
    success_at: int | None = None

    def __init__(self) -> None:
        self.env = _FakeRobosuite()
        self.t = 0

    def _obs(self) -> dict:
        return {
            "object": np.zeros(14),
            "robot0_eye_in_hand_image": np.zeros((96, 96, 3), dtype=np.uint8),
        }

    def reset(self) -> dict:
        self.t = 0
        return self._obs()

    def step(self, action):
        self.t += 1
        return self._obs(), 0.0, False, {}

    def is_success(self) -> dict:
        return {"task": self.success_at is not None and self.t >= self.success_at}

    def render(self, **kwargs) -> np.ndarray:
        return np.zeros((kwargs["height"], kwargs["width"], 3), dtype=np.uint8)


@pytest.fixture
def make_env(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeEnvRobosuite()
    monkeypatch.setattr(
        robomimic_env._env_utils, "create_env_from_metadata", lambda **kw: fake
    )
    monkeypatch.setattr(
        robomimic_env._obs_utils,
        "initialize_obs_modality_mapping_from_dict",
        lambda mapping: None,
    )
    monkeypatch.setattr(
        robomimic_env.robomimic.envs.env_robosuite, "EnvRobosuite", _FakeEnvRobosuite
    )

    def make(success_at: int | None = None, **config):
        fake.success_at = success_at
        env = RobomimicEnv(
            RobomimicConfig(name="square-img", agentview_image_size=(8, 8), **config)
        )
        env.reset()
        return env

    return make


def _step(env) -> tuple[bool, bool]:
    _, _, terminated, truncated, _ = env.step(np.zeros((1, 7), dtype=np.float32))
    return bool(terminated[0]), bool(truncated[0])


def test_an_episode_is_cut_at_the_task_s_rollout_horizon(make_env):
    env = make_env()  # NutAssemblySquare: robomimic rolls out 400 steps

    ends = [_step(env) for _ in range(400)]

    assert ends[:399] == [(False, False)] * 399
    assert ends[399] == (False, True)


def test_success_ends_the_episode(make_env):
    env = make_env(success_at=5)

    ends = [_step(env) for _ in range(5)]

    assert ends[:4] == [(False, False)] * 4
    assert ends[4] == (True, False)


def test_success_can_be_left_running(make_env):
    env = make_env(success_at=5, terminate_on_success=False)

    ends = [_step(env) for _ in range(6)]

    assert ends == [(False, False)] * 6


def test_an_explicit_horizon_wins(make_env):
    env = make_env(horizon=10)

    ends = [_step(env) for _ in range(10)]

    assert ends[9] == (False, True)


def test_reset_restarts_the_count(make_env):
    env = make_env(horizon=3)
    for _ in range(3):
        _step(env)

    env.reset()

    assert [_step(env) for _ in range(3)] == [(False, False), (False, False), (False, True)]


def test_a_task_with_no_known_horizon_needs_one(make_env, monkeypatch):
    monkeypatch.setattr(robomimic_env, "_ROLLOUT_HORIZONS", {}, raising=False)

    with pytest.raises(ValueError, match="horizon"):
        make_env()
