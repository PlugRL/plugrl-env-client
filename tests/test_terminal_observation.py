"""What the training server is told about a terminal transition.

BaseEnv declares `autoreset_mode: AutoresetMode.NEXT_STEP`, which is a promise
to the caller: on the step that reports done, `step()` returns the *terminal*
observation, and the reset happens later. rollout() relies on that promise -
it sends feedback with the observation `step()` just returned, and only then
calls `env.reset(reset_indices=...)`.

An environment that resets inside its own `step()` breaks the promise
silently. Nothing raises; the terminal transition simply carries the first
observation of the *next* episode. That is corrupt training data, not a
cosmetic problem, and it is invisible unless someone looks.

None of the environments in this package do that today - each one delegates to
its backend and returns what it gets, and the Atari wrapper marks a lost life
terminal without resetting. These tests exist to keep it that way, and to make
the failure mode executable for whoever adds the next environment.
"""

import dataclasses

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

from plugrl_env_client.envs.base_env import BaseEnv, BaseEnvConfig, Observation
from plugrl_env_client.runner.rollout import rollout

TERMINAL_MARKER = 111.0
FRESH_MARKER = 999.0


class _RecordingAgent:
    """Captures what rollout would have sent to the training server."""

    def __init__(self, action_dim=2, horizon=1):
        self._chunk = np.zeros((horizon, 1, action_dim), dtype=np.float32)
        self.feedback_obs = []

    def infer(self, obs, *, env_indices, step_ids):
        return {"action": self._chunk}

    def feedback(
        self, *, obs, rewards, terminated, truncated, info, env_indices, step_ids
    ):
        self.feedback_obs.append(obs["states"]["marker"].copy())


class _WellBehavedEnv(BaseEnv):
    """Honours NEXT_STEP: the done step returns the terminal observation."""

    def __init__(self, config=None, num_envs=1, **kwargs):
        super().__init__(config=config or BaseEnvConfig(), num_envs=num_envs)
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = self.single_action_space
        self._steps = 0
        self._marker = FRESH_MARKER

    def _obs(self):
        return Observation(
            images={},
            states={"marker": np.full((self.num_envs, 1), self._marker, np.float32)},
            text="",
        )

    def reset(self, *, seed=None, options=None):
        self.seed_rngs(seed)
        self._steps = 0
        self._marker = FRESH_MARKER
        return self._obs(), {}

    def step(self, actions):
        self._steps += 1
        done = self._steps >= 2
        if done:
            self._marker = TERMINAL_MARKER
        return (
            self._obs(),
            np.zeros((self.num_envs,), np.float32),
            np.full((self.num_envs,), done, np.bool_),
            np.zeros((self.num_envs,), np.bool_),
            {},
        )


class _ResetsInsideStepEnv(_WellBehavedEnv):
    """Violates NEXT_STEP by resetting itself on the done step.

    This is the hazard. It declares the same metadata as every other env and
    raises nothing; the only symptom is that the terminal observation the
    server receives belongs to the next episode.
    """

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        if terminated.any():
            self._steps = 0
            self._marker = FRESH_MARKER  # the reset the caller has not asked for
            obs = self._obs()
        return obs, reward, terminated, truncated, info


def _run_one_episode(env_cls):
    env = env_cls()
    agent = _RecordingAgent()
    rollout(env, agent, num_episodes=1, replan_steps=1, num_envs=1, seed=0)
    return agent.feedback_obs


class TestDeclaredContract:
    def test_base_env_declares_next_step(self):
        """rollout()'s ordering is only correct under NEXT_STEP.

        If this ever changes, rollout must change with it: it sends feedback
        before resetting, which assumes the done step was not already a reset.
        """
        assert BaseEnv.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP

    @pytest.mark.parametrize(
        "module_name,class_name",
        [
            ("plugrl_env_client.envs.dummy_env", "DummyEnv"),
            ("plugrl_env_client.envs.classic.classic_env", "ClassicEnv"),
            ("plugrl_env_client.envs.atari.atari_env", "AtariEnv"),
            ("plugrl_env_client.envs.d4rl.d4rl_env", "D4RLEnv"),
            ("plugrl_env_client.envs.libero.libero_env", "LiberoEnv"),
            ("plugrl_env_client.envs.robomimic.robomimic_env", "RobomimicEnv"),
            ("plugrl_env_client.envs.mujoco.mujoco_env", "MuJoCoEnv"),
        ],
    )
    def test_no_env_overrides_the_mode(self, module_name, class_name):
        """An env quietly declaring SAME_STEP would make rollout wrong for it."""
        module = pytest.importorskip(module_name, exc_type=ImportError)
        env_cls = getattr(module, class_name)
        assert env_cls.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP


class TestTerminalObservationReachesTheServer:
    def test_well_behaved_env_reports_the_terminal_observation(self):
        markers = _run_one_episode(_WellBehavedEnv)
        assert markers, "rollout sent no feedback"
        assert markers[-1].item() == TERMINAL_MARKER, (
            "the last feedback should carry the terminal observation"
        )

    def test_an_env_that_resets_itself_corrupts_the_terminal_transition(self):
        """Documents the hazard rather than asserting any shipped env has it.

        If this test ever starts failing, rollout has grown a defence against
        misbehaving environments - which is a reasonable thing to add, and
        this test should then be inverted rather than deleted.
        """
        markers = _run_one_episode(_ResetsInsideStepEnv)
        assert markers, "rollout sent no feedback"
        assert markers[-1].item() == FRESH_MARKER, (
            "expected the known-bad env to leak its reset observation; if it no "
            "longer does, rollout now sanitises terminal observations"
        )


def test_observation_is_serialisable_as_sent():
    """feedback sends dataclasses.asdict(obs); a terminal obs must survive it."""
    env = _WellBehavedEnv()
    env.reset(seed=0)
    obs, *_ = env.step(np.zeros((1, 2), np.float32))
    as_sent = dataclasses.asdict(obs)
    assert set(as_sent) == {"images", "states", "text"}
