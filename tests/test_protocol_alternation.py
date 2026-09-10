"""Executable clauses of the wire protocol's exchange rules.

`plugrl-protocol/SPEC.md` makes three claims about the exchange that cannot
be read off any single function, because they are properties of the loop in
`rollout()` running against the loop in `WebSocketAgentServer._handler`:

  * section 4.2 - messages strictly alternate infer, action, feedback. The
    server's handler is straight-line code with no dispatcher, so a client
    that sends two infers in a row has the second one parsed as a feedback.
  * section 4.3 - the env set in the feedback need not match the env set in
    the infer it follows. The pairing is flow control, not association.
  * section 5.3 / 5.4 - the action chunk is time-major, and the reward
    reported for a chunk is the sum over the chunk.

These were derived by reading. This file makes them observable, by
recording every message rollout() would put on the wire and asserting the
sequence. If the rollout loop is ever restructured, these fail rather than
the protocol silently drifting away from its own specification.
"""

import gymnasium as gym
import numpy as np
import pytest

from plugrl_env_client.envs.base_env import BaseEnv, BaseEnvConfig, Observation
from plugrl_env_client.runner.rollout import rollout

# Deliberately not equal to the two environments below: an action dimension
# that matches num_envs exercises the _get_action_spec batch-axis ambiguity
# instead, which test_action_space_contract.py owns.
ACTION_DIM = 3


class _WireRecorder:
    """Stands in for the server, and remembers what it was told.

    Records the message sequence as (kind, env_indices) so a test can make
    statements about the whole exchange rather than one call at a time.
    """

    def __init__(self, horizon):
        self.horizon = horizon
        self.messages = []
        self.rewards = {}

    def infer(self, obs, *, env_indices, step_ids):
        env_indices = tuple(int(i) for i in env_indices)
        self.messages.append(("infer", env_indices))

        # Time-major, per SPEC section 5.3: [H, n, *da]. Each action encodes
        # the horizon position it belongs to, so the environment can report
        # back the order it actually executed them in.
        chunk = np.zeros((self.horizon, len(env_indices), ACTION_DIM), np.float32)
        for t in range(self.horizon):
            chunk[t, :, 0] = float(t)
        return {"action": chunk}

    def feedback(
        self, *, obs, rewards, terminated, truncated, info, env_indices, step_ids
    ):
        env_indices = tuple(int(i) for i in env_indices)
        self.messages.append(("feedback", env_indices))
        for position, env_id in enumerate(env_indices):
            self.rewards.setdefault(env_id, []).append(float(rewards[position]))

    @property
    def kinds(self):
        return [kind for kind, _ in self.messages]

    def cycles(self):
        """The (infer set, feedback set) pairs, in order."""
        pairs = []
        pending = None
        for kind, env_indices in self.messages:
            if kind == "infer":
                pending = env_indices
            elif pending is not None:
                pairs.append((pending, env_indices))
                pending = None
        return pairs


class _StaggeredEnv(BaseEnv):
    """Two environments with different episode lengths.

    That is the whole trick: with equal episode lengths the environments
    stay in lockstep and every message carries the same env set, which is
    exactly the case that hides the section 4.3 hazard. One short episode is
    enough to desynchronise them permanently.
    """

    def __init__(self, episode_lengths=(2, 6), num_envs=2, **kwargs):
        super().__init__(config=BaseEnvConfig(), num_envs=num_envs)
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        self.action_space = self.single_action_space
        self._episode_lengths = np.asarray(episode_lengths, dtype=np.int64)
        self._steps = np.zeros(num_envs, dtype=np.int64)
        self.actions_seen = [[] for _ in range(num_envs)]

    def _obs(self):
        return Observation(
            images={},
            states={"step": self._steps.reshape(-1, 1).astype(np.float32)},
            text="stay still",
        )

    def reset(self, *, seed=None, options=None):
        self.seed_rngs(seed)
        indices = (options or {}).get("reset_indices")
        if indices is None:
            self._steps[:] = 0
        else:
            self._steps[np.asarray(indices, dtype=np.int64)] = 0
        return self._obs(), {}

    def step(self, actions):
        actions = np.asarray(actions)
        for env_id in range(self.num_envs):
            self.actions_seen[env_id].append(float(actions[env_id, 0]))

        self._steps += 1
        terminated = self._steps >= self._episode_lengths
        return (
            self._obs(),
            np.ones((self.num_envs,), np.float32),  # one reward per env per step
            terminated.astype(np.bool_),
            np.zeros((self.num_envs,), np.bool_),
            {},
        )


def _run(horizon, episode_lengths=(2, 6), num_episodes=4):
    env = _StaggeredEnv(episode_lengths=episode_lengths)
    recorder = _WireRecorder(horizon=horizon)
    rollout(
        env,
        recorder,
        num_episodes=num_episodes,
        replan_steps=horizon,
        num_envs=2,
        seed=0,
    )
    return env, recorder


class TestAlternation:
    """SPEC section 4.2."""

    @pytest.mark.parametrize("horizon", [1, 2, 4])
    def test_messages_alternate_infer_then_feedback(self, horizon):
        """The server has no dispatcher; two infers in a row desynchronise it."""
        _, recorder = _run(horizon)

        kinds = recorder.kinds
        assert kinds, "rollout put nothing on the wire"
        assert kinds[0] == "infer", "the first client message is an infer"
        for position in range(1, len(kinds)):
            assert kinds[position] != kinds[position - 1], (
                f"two {kinds[position]} messages in a row at position {position}: "
                f"{kinds[: position + 1]}"
            )

    @pytest.mark.parametrize("horizon", [1, 2, 4])
    def test_at_most_one_infer_is_ever_outstanding(self, horizon):
        """Restates the invariant as a running balance, which is how the
        server experiences it: it blocks after each action until feedback."""
        _, recorder = _run(horizon)

        outstanding = 0
        for kind in recorder.kinds:
            outstanding += 1 if kind == "infer" else -1
            assert 0 <= outstanding <= 1, "an infer went unanswered by feedback"

    def test_a_long_chunk_delays_feedback_by_many_env_steps(self):
        """SPEC section 4.2: feedback may arrive H env steps after the action.

        This is why the server allows 60 s for it rather than expecting a
        prompt reply.
        """
        env, recorder = _run(horizon=4, episode_lengths=(6, 6), num_episodes=2)

        assert recorder.kinds[:2] == ["infer", "feedback"]
        # Four env steps of an environment that never terminated early.
        assert len(env.actions_seen[0]) >= 4


class TestEnvSetsDiverge:
    """SPEC section 4.3 - the claim most likely to be implemented wrong."""

    def test_lockstep_environments_hide_the_hazard(self):
        """Equal episode lengths make every set equal, which is the trap."""
        _, recorder = _run(horizon=4, episode_lengths=(4, 4), num_episodes=2)

        assert all(
            infer_set == feedback_set for infer_set, feedback_set in recorder.cycles()
        ), "this configuration is supposed to stay in lockstep"

    def test_one_early_termination_desynchronises_them_permanently(self):
        _, recorder = _run(horizon=4, episode_lengths=(2, 6))

        cycles = recorder.cycles()
        assert cycles, "no complete infer/feedback cycle was recorded"
        diverged = [
            (infer_set, feedback_set)
            for infer_set, feedback_set in cycles
            if infer_set != feedback_set
        ]
        assert diverged, (
            "expected at least one cycle whose feedback env set differs from "
            f"its infer env set; got {cycles}"
        )

    def test_feedback_can_name_an_env_the_infer_did_not(self):
        """The strongest form: not merely a subset, a different environment.

        A client implementation that indexes the feedback by position in the
        action it just received would corrupt exactly here.
        """
        _, recorder = _run(horizon=4, episode_lengths=(2, 6))

        disjoint = [
            (infer_set, feedback_set)
            for infer_set, feedback_set in recorder.cycles()
            if set(feedback_set) - set(infer_set)
        ]
        assert disjoint, (
            "expected a cycle whose feedback names an env absent from the "
            f"infer; got {recorder.cycles()}"
        )


class TestChunkSemantics:
    """SPEC sections 5.3 and 5.4."""

    def test_actions_are_consumed_in_horizon_order(self):
        """Time-major: chunk[t] is the action for horizon position t.

        The recorder encodes t in the action, so an environment that saw
        [0, 1, 2, 3] proves the client did not transpose the chunk.
        """
        env, _ = _run(horizon=4, episode_lengths=(8, 8), num_episodes=2)

        assert env.actions_seen[0][:4] == [0.0, 1.0, 2.0, 3.0]

    def test_reward_reported_for_a_chunk_is_the_sum_over_the_chunk(self):
        """Each env step pays 1.0, so a full four-step chunk must report 4.0.

        Reporting the last step's reward instead would send 1.0 and train a
        different MDP without failing anything.
        """
        _, recorder = _run(horizon=4, episode_lengths=(8, 8), num_episodes=2)

        assert recorder.rewards[0][0] == pytest.approx(4.0)

    def test_a_chunk_cut_short_by_termination_reports_only_what_it_earned(self):
        """Env 0 terminates on its second step, two into a four-step chunk."""
        _, recorder = _run(horizon=4, episode_lengths=(2, 6))

        assert recorder.rewards[0][0] == pytest.approx(2.0)

    def test_horizon_one_makes_chunk_reward_equal_step_reward(self):
        """The degenerate case, which is why H=1 hides the section 5.4 hazard."""
        _, recorder = _run(horizon=1, episode_lengths=(4, 4), num_episodes=2)

        assert all(reward == pytest.approx(1.0) for reward in recorder.rewards[0])
