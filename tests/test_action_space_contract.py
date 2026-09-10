"""Every environment has to expose an action space, or it cannot run at all.

rollout() calls _get_action_spec() before its first step, to size the action
plan. That helper reads single_action_space, falls back to action_space, and
raises ValueError when neither exists. Three of the six environment families
- atari, classic and robomimic - never defined either, so
`plugrl-run-env-client classic-v1` failed on startup with "Env must expose
action space via single_action_space/action_space". (d4rl defines it as a
property, which is easy to miss when grepping for an assignment.)

Nothing caught it because the only env exercised by the test suite was the
one that happened to define a space.
"""

import gymnasium as gym
import numpy as np
import pytest

import plugrl_env_client.envs  # noqa: F401  # registers the envs
from plugrl_env_client.utils.registration import (
    REGISTERED_ENV_CONFIGS,
    REGISTERED_ENVS,
)
from plugrl_env_client.utils.rollout import _get_action_spec


def _construct(uid):
    """Build a registered env with its default config, or skip if unavailable."""
    if uid not in REGISTERED_ENV_CONFIGS:
        pytest.skip(f"{uid} has no registered config")
    spec = REGISTERED_ENVS[uid]
    try:
        return spec.cls(config=REGISTERED_ENV_CONFIGS[uid], num_envs=1)
    except Exception as exc:  # missing extras, missing assets, missing display
        pytest.skip(f"{uid} cannot be constructed here: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("uid", sorted(REGISTERED_ENVS))
def test_env_exposes_an_action_space(uid):
    env = _construct(uid)
    space = getattr(env, "single_action_space", None)
    if space is None:
        space = getattr(env, "action_space", None)
    assert space is not None, (
        f"{uid} exposes neither single_action_space nor action_space, so "
        "rollout() cannot size its action plan and the env cannot run at all"
    )


@pytest.mark.parametrize("uid", sorted(REGISTERED_ENVS))
def test_action_spec_resolves(uid):
    """The call rollout() actually makes before its first step."""
    env = _construct(uid)
    shape, dtype = _get_action_spec(env, num_envs=1)
    assert isinstance(shape, tuple)
    assert dtype is not None


class _UnbatchedSpaceEnv:
    """Only what _get_action_spec reads: an unbatched single_action_space.

    Every env family in this package declares it this way - as the space of
    one environment, never of the vector.
    """

    def __init__(self, action_dim):
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.action_space = self.single_action_space


@pytest.mark.parametrize("action_dim", [1, 2, 4, 7])
def test_action_dim_equal_to_num_envs_is_not_mistaken_for_a_batch_axis(action_dim):
    """`single_action_space` is unbatched, so its leading axis is never a batch.

    Reading it as one drops a real dimension whenever the two numbers
    coincide, and rollout() then fails with "Expected action shape tail (),
    got (7,)" - or, for a one-dimensional action, sizes the action plan
    with an empty shape and never fails at all.
    """
    env = _UnbatchedSpaceEnv(action_dim)

    shape, _ = _get_action_spec(env, num_envs=action_dim)

    assert shape == (action_dim,)


def test_batched_action_space_fallback_still_strips_the_leading_axis():
    """Envs with no single_action_space keep the old behaviour.

    On a Gymnasium vector env, `action_space` really is the batched space.
    """

    class _BatchedOnly:
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3, 7), dtype=np.float32
        )

    shape, _ = _get_action_spec(_BatchedOnly(), num_envs=3)

    assert shape == (7,)


def test_missing_both_spaces_still_raises():
    class _Neither:
        pass

    with pytest.raises(ValueError, match="single_action_space/action_space"):
        _get_action_spec(_Neither(), num_envs=1)
