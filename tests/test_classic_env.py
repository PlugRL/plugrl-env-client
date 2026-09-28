"""The classic-control family runs its continuous tasks, and renders only on request.

It used to cast every action to an integer, which made Pendulum-v1 and
MountainCarContinuous-v0 unusable, and it rendered a frame every step whether
or not anything read it.
"""

import numpy as np
import pytest

classic_env = pytest.importorskip(
    "plugrl_env_client.envs.classic.classic_env", exc_type=ImportError
)

ClassicConfig = classic_env.ClassicConfig
ClassicEnv = classic_env.ClassicEnv


def make(name: str, render: bool = False) -> ClassicEnv:
    return ClassicEnv(config=ClassicConfig(name=name, render=render), num_envs=1)


class TestContinuousActions:
    def test_pendulum_takes_a_float_torque(self):
        env = make("Pendulum-v1")
        env.reset(seed=0)
        env.env.unwrapped.state = np.array([0.0, 0.0])
        # A torque of 1.5 from rest: an integer cast would have applied 1.0.
        obs, reward, terminated, truncated, _ = env.step(np.array([1.5], dtype=np.float32))
        assert obs.states["obs"].shape == (1, 3)
        assert obs.states["obs"][0, 2] == pytest.approx(3.0 * 1.5 * 0.05, rel=1e-6)
        assert reward.shape == terminated.shape == truncated.shape == (1,)

    def test_the_action_takes_the_space_shape(self):
        env = make("Pendulum-v1")
        env.reset(seed=0)
        # rollout() hands over one env's row of the plan, possibly with extra
        # leading axes; the env must reduce it to the space's own shape.
        env.step(np.array([[0.5]], dtype=np.float32))


class TestDiscreteActions:
    def test_cartpole_still_takes_an_integer(self):
        env = make("CartPole-v1")
        env.reset(seed=0)
        obs, *_ = env.step(np.array([1]))
        assert obs.states["obs"].shape == (1, 4)


class TestRendering:
    def test_no_frames_by_default(self):
        env = make("Pendulum-v1")
        obs, _ = env.reset(seed=0)
        assert obs.images == {}

    def test_frames_when_asked(self):
        env = make("Pendulum-v1", render=True)
        obs, _ = env.reset(seed=0)
        assert obs.images["env"].ndim == 4
