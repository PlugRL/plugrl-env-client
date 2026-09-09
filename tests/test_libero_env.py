import numpy as np
import pytest

libero_env_module = pytest.importorskip(
    "plugrl_env_client.envs.libero.libero_env",
    exc_type=ImportError,
    reason="requires the libero extra",
)

LiberoConfig = libero_env_module.LiberoConfig
LiberoEnv = libero_env_module.LiberoEnv


class _FakeTaskSuite:
    def get_num_tasks(self) -> int:
        return 1

    def get_task(self, task_id: int) -> object:
        return object()

    def get_task_init_states(self, task_id: int) -> list[np.ndarray]:
        return [np.zeros(1, dtype=np.float32)]


class _FakeInnerEnv:
    action_spec = (
        np.full(7, -1.0, dtype=np.float64),
        np.full(7, 1.0, dtype=np.float64),
    )


class _FakeLiberoBaseEnv:
    def __init__(self) -> None:
        self.env = _FakeInnerEnv()

    def close(self) -> None:
        return None


def test_libero_env_exposes_single_action_space(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        libero_env_module.benchmark,
        "get_benchmark_dict",
        lambda: {"libero_spatial": lambda: _FakeTaskSuite()},
    )
    monkeypatch.setattr(
        libero_env_module,
        "_get_libero_env",
        lambda task, resolution, seed: (_FakeLiberoBaseEnv(), "fake task"),
    )

    env = LiberoEnv(LiberoConfig())

    assert env.single_action_space is env.action_space
    assert env.action_space.shape == (7,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_allclose(env.action_space.low, -1.0)
    np.testing.assert_allclose(env.action_space.high, 1.0)
