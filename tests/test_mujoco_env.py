"""The environment that has to work for a first learning curve to exist.

Every other family here is ruled out for that job: three need assets or a
Linux-only stack, one is discrete-action, and the dummy one has no reward to
learn from. So this file is less about MuJoCo than about the three properties
that make an environment usable at all, each of which has already been got
wrong once in this package:

  * it must expose an action space, or rollout() cannot size its action plan
    and the env fails at startup (this is what stopped atari, classic and
    robomimic running);
  * it must put its state under `states["obs"]`, because that is the single
    key FPOPolicy reads (fpo_policy.py:177);
  * it must not render when nobody asked for pixels, because rendering every
    step costs more than the physics and needs a display.
"""

import numpy as np
import pytest

mujoco_env = pytest.importorskip(
    "plugrl_env_client.envs.mujoco.mujoco_env", exc_type=ImportError
)

MuJoCoConfig = mujoco_env.MuJoCoConfig
MuJoCoEnv = mujoco_env.MuJoCoEnv


@pytest.fixture
def env():
    e = MuJoCoEnv(config=MuJoCoConfig(), num_envs=1)
    yield e
    e.env.close()


class TestItCanRunAtAll:
    def test_it_exposes_an_action_space(self, env):
        """rollout() reads this before the first step."""
        from plugrl_env_client.utils.rollout import _get_action_spec

        shape, dtype = _get_action_spec(env, num_envs=1)

        assert shape == (6,), "HalfCheetah-v5 has a 6-dimensional action"
        assert dtype == np.float32

    def test_it_is_registered(self):
        from plugrl_env_client.utils.registration import (
            REGISTERED_ENV_CONFIGS,
            REGISTERED_ENVS,
        )

        assert "MuJoCo-v1" in REGISTERED_ENVS
        assert "MuJoCo-v1" in REGISTERED_ENV_CONFIGS

    def test_more_than_one_env_is_refused_rather_than_mishandled(self):
        with pytest.raises(ValueError, match="num_envs=1"):
            MuJoCoEnv(config=MuJoCoConfig(), num_envs=4)


class TestTheShapeFPOExpects:
    def test_the_default_task_matches_the_shipped_policy_defaults(self, env):
        """HalfCheetah-v5 is obs 17 / act 6; FPOPolicyConfig defaults to both.

        If this ever stops being true, pointing FPO at this environment needs
        config surgery, and the "no setup required" claim in the README goes
        with it.
        """
        obs, _ = env.reset(seed=0)

        assert obs.states["obs"].shape == (1, 17)
        assert env.single_action_space.shape == (6,)

    def test_state_is_under_the_key_fpo_reads(self, env):
        obs, _ = env.reset(seed=0)

        assert set(obs.states) == {"obs"}, "fpo_policy.py:177 reads states['obs']"
        assert obs.states["obs"].dtype == np.float32


class TestRenderingIsOff:
    def test_no_images_by_default(self, env):
        """SPEC.md section 5.2 allows an observation with no images at all.

        A state-only policy never looks at the frames, and producing them
        costs more per step than the simulation does.
        """
        obs, _ = env.reset(seed=0)

        assert obs.images == {}
        assert env.env.render_mode is None

    def test_rendering_can_be_turned_on(self):
        e = MuJoCoEnv(config=MuJoCoConfig(render=True), num_envs=1)
        try:
            assert e.env.render_mode == "rgb_array"
        finally:
            e.env.close()


class TestSteppingAndSeeding:
    def test_a_step_returns_the_five_batched_pieces(self, env):
        env.reset(seed=0)
        action = np.zeros((1, 6), dtype=np.float32)

        obs, reward, terminated, truncated, _ = env.step(action)

        assert obs.states["obs"].shape == (1, 17)
        assert reward.shape == (1,) and reward.dtype == np.float32
        assert terminated.shape == (1,) and terminated.dtype == np.bool_
        assert truncated.shape == (1,) and truncated.dtype == np.bool_

    def test_the_same_seed_gives_the_same_first_observation(self):
        first = MuJoCoEnv(config=MuJoCoConfig(), num_envs=1)
        second = MuJoCoEnv(config=MuJoCoConfig(), num_envs=1)
        try:
            a, _ = first.reset(seed=7)
            b, _ = second.reset(seed=7)
            np.testing.assert_array_equal(a.states["obs"], b.states["obs"])
        finally:
            first.env.close()
            second.env.close()

    def test_different_seeds_give_different_observations(self):
        first = MuJoCoEnv(config=MuJoCoConfig(), num_envs=1)
        second = MuJoCoEnv(config=MuJoCoConfig(), num_envs=1)
        try:
            a, _ = first.reset(seed=7)
            b, _ = second.reset(seed=8)
            assert not np.array_equal(a.states["obs"], b.states["obs"])
        finally:
            first.env.close()
            second.env.close()

    def test_a_plugrl_reset_index_does_not_reach_the_mujoco_env(self, env):
        """rollout() passes options={"reset_indices": ...} on autoreset.

        That key is PlugRL's, not Gymnasium's; forwarding it would arrive at
        the MuJoCo env as an unrecognised reset option.
        """
        obs, _ = env.reset(options={"reset_indices": np.array([0])})

        assert obs.states["obs"].shape == (1, 17)
