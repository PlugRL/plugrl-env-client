"""After a reconnect, no chunk from the old connection is finished or fed back.

The rollout runs envs on chunks of different ages: an env whose episode just
ended gets a new chunk while the others are still executing theirs. When the
connection is replaced, every chunk still in flight was answered on the old
one, and the server's half of those transitions went with it (SPEC section
7.6). The rollout used to keep executing them and send their feedback on the
new connection; `plugrl-conformance --probe --scenario resync` caught it.
Now the agent raises ConnectionReplaced and the rollout drops them all, so
its first infer on the new connection asks for every env.
"""

from __future__ import annotations

import numpy as np

from plugrl_env_client.agent.websocket_env_client_agent import ConnectionReplaced
from plugrl_env_client.envs.probe_env import ProbeEnv, ProbeEnvConfig
from plugrl_env_client.runner.rollout import rollout

HORIZON = 4


class _ReplacingAgent:
    """Answers infers with zero chunks; replaces the connection on one feedback."""

    def __init__(self, replace_on_feedback: int) -> None:
        self.replace_on = replace_on_feedback
        self.connection = 1
        self.chunk_from: dict[int, int] = {}
        self.feedbacks = 0
        self.infers: list[tuple[int, list[int]]] = []
        self.stale: list[tuple[int, list[int]]] = []

    def infer(self, obs, *, env_indices, step_ids):
        envs = [int(e) for e in env_indices]
        self.infers.append((self.connection, envs))
        for env in envs:
            self.chunk_from[env] = self.connection
        return {"action": np.zeros((HORIZON, len(envs), 3))}

    def feedback(self, *, env_indices, **_):
        self.feedbacks += 1
        if self.feedbacks == self.replace_on:
            self.connection += 1
            raise ConnectionReplaced("the server closed for a resync")
        envs = [int(e) for e in env_indices]
        stale = [e for e in envs if self.chunk_from.get(e) != self.connection]
        if stale:
            self.stale.append((self.connection, stale))


def test_no_chunk_from_the_old_connection_is_fed_back_on_the_new_one():
    env = ProbeEnv(ProbeEnvConfig(), num_envs=3)  # episodes of 3, 5 and 7 steps
    # Feedback 1 is env 0 ending at t=3; env 0 then gets a new chunk. Feedback
    # 2, at t=4, is envs 1 and 2 finishing theirs - and the connection goes.
    agent = _ReplacingAgent(replace_on_feedback=2)

    rollout(env, agent, num_episodes=8, replan_steps=None, num_envs=3)

    assert agent.stale == [], "fed back a chunk answered on an earlier connection"
    after = [envs for connection, envs in agent.infers if connection == 2]
    assert after, "nothing was asked on the new connection"
    assert after[0] == [0, 1, 2], "the first infer after a reconnect re-plans every env"


def test_an_episode_that_ended_is_reset_even_when_its_feedback_is_dropped():
    env = ProbeEnv(ProbeEnvConfig(), num_envs=3)
    agent = _ReplacingAgent(replace_on_feedback=1)  # env 0's terminal feedback

    rollout(env, agent, num_episodes=6, replan_steps=None, num_envs=3)

    # Had env 0 not been reset, its episode of 3 would never end again and
    # the rollout would not finish six episodes.
    assert agent.stale == []
