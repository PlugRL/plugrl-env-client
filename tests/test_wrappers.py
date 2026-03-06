import numpy as np
import gymnasium as gym

from plugrl_worker.utils.wrappers.real_time_wrapper import RealTimeWrapper
from plugrl_worker.utils.wrappers.success_record_wrapper import RecordSuccessByStep


class _OneStepEnv(gym.Env):
    metadata = {}

    def __init__(self, reward: float = 1.0):
        super().__init__()
        self._reward = reward
        self._stepped = False
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self._stepped = False
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        if self._stepped:
            raise RuntimeError("This env only supports a single step")
        self._stepped = True
        obs = np.zeros((1,), dtype=np.float32)
        info = {"episode": {}}
        return obs, float(self._reward), True, False, info


def test_record_success_by_step_updates_info():
    env = _OneStepEnv(reward=1.0)
    wrapped = RecordSuccessByStep(env, best_reward_threshold_for_success=0.5)

    wrapped.reset()
    _, _, terminated, truncated, info = wrapped.step(np.array([0.0], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert info["is_step_success"] is True
    assert info["episode"]["s"] is True
    assert 0.0 <= info["episode"]["mean_success_rate"] <= 1.0


def test_real_time_wrapper_sleeps_non_negative(monkeypatch):
    env = _OneStepEnv(reward=0.0)
    wrapped = RealTimeWrapper(env, fps=30.0)

    t = {"now": 0.0}

    def fake_time():
        return t["now"]

    slept = []

    def fake_sleep(dt):
        slept.append(dt)

    monkeypatch.setattr("time.time", fake_time)
    monkeypatch.setattr("time.sleep", fake_sleep)

    wrapped.reset()

    # First step: no previous timestamp -> should not sleep
    wrapped.step(np.array([0.0], dtype=np.float32))

    # Second env instance to force sleep path with last_step_time set
    env2 = _OneStepEnv(reward=0.0)
    wrapped2 = RealTimeWrapper(env2, fps=30.0)
    monkeypatch.setattr("time.time", fake_time)
    monkeypatch.setattr("time.sleep", fake_sleep)
    wrapped2.reset()

    wrapped2.last_step_time = 0.0
    t["now"] = 0.0
    wrapped2.step(np.array([0.0], dtype=np.float32))

    assert all(dt >= 0 for dt in slept)
