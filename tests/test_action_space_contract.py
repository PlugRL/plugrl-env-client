"""Every environment has to expose an action space, or it cannot run at all.

rollout() calls _get_action_spec() before its first step, to size the action
plan. That helper reads single_action_space, falls back to action_space, and
raises ValueError when neither exists. Four of the six environment families
never defined either, so `plugrl-run-env-client classic-v1` fails on startup
with "Env must expose action space via single_action_space/action_space".

Nothing caught it because the only env exercised by the test suite was the
one that happened to define a space.
"""

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
