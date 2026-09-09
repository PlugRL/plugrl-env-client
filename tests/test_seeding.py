"""Seeding is what makes a run reproducible, and reproducibility is what makes
a reported number mean anything.

Until these pass, two runs of the same configuration produce different
trajectories, and "3 seeds per configuration" is not something this package
can actually deliver.

Three separate things have to hold:

  * an environment given the same seed produces the same episode
  * different seeds produce different episodes (otherwise the first property
    is satisfied trivially by ignoring randomness)
  * separate client processes get *different* seeds, or running N processes
    just collects the same trajectory N times
"""

import numpy as np
import pytest

from plugrl_env_client.envs.dummy_env import DummyEnv, DummyEnvConfig
from plugrl_env_client.runner.args import RunnerArgs
from plugrl_env_client.runner.run import derive_process_seed


def _first_observation(seed):
    env = DummyEnv(DummyEnvConfig(), num_envs=2)
    obs, _ = env.reset(seed=seed)
    return obs.states["robot_state"].copy()


def test_same_seed_gives_the_same_reset():
    assert np.array_equal(_first_observation(1234), _first_observation(1234))


def test_different_seeds_give_different_resets():
    assert not np.array_equal(_first_observation(1234), _first_observation(4321))


def test_same_seed_gives_the_same_episode():
    def episode(seed):
        env = DummyEnv(DummyEnvConfig(), num_envs=2)
        env.reset(seed=seed)
        rewards = []
        for _ in range(5):
            _, reward, _, _, _ = env.step(env.fake_action())
            rewards.append(reward.copy())
        return np.stack(rewards)

    assert np.array_equal(episode(7), episode(7))
    assert not np.array_equal(episode(7), episode(8))


def test_unseeded_reset_still_works():
    """Seeding is opt-in; omitting it must not break anything."""
    env = DummyEnv(DummyEnvConfig(), num_envs=2)
    obs, _ = env.reset()
    assert obs.states["robot_state"].shape[0] == 2


class TestProcessSeeds:
    """Every client process needs its own slice of the random stream."""

    def test_processes_get_distinct_seeds(self):
        seeds = {derive_process_seed(100, process_id=i, num_envs=4) for i in range(8)}
        assert len(seeds) == 8

    def test_slices_do_not_overlap(self):
        # A vector env seeds its sub-envs from seed, seed+1, ... so consecutive
        # processes must be at least num_envs apart or their sub-envs collide.
        num_envs = 4
        used = set()
        for process_id in range(5):
            base = derive_process_seed(100, process_id=process_id, num_envs=num_envs)
            slice_ = set(range(base, base + num_envs))
            assert not (slice_ & used), f"process {process_id} overlaps an earlier one"
            used |= slice_

    def test_none_stays_none(self):
        assert derive_process_seed(None, process_id=3, num_envs=4) is None

    def test_single_process_gets_the_base_seed(self):
        assert derive_process_seed(100, process_id=None, num_envs=4) == 100


def test_runner_args_exposes_seed():
    assert RunnerArgs(uid="dummy-v1").seed is None
    assert RunnerArgs(uid="dummy-v1", seed=42).seed == 42


@pytest.mark.parametrize("seed", [0, 1, 2**31 - 1])
def test_seed_edge_values_are_accepted(seed):
    env = DummyEnv(DummyEnvConfig(), num_envs=1)
    obs, _ = env.reset(seed=seed)
    assert obs.states["robot_state"].shape[0] == 1
