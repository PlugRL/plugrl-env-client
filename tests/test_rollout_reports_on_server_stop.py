"""A run the server ended still has to report what it did.

`rollout()` logs a final timing summary when its loop finishes. The loop also
ends the other way: the server takes the last step it was asked for, closes
with `plugrl-server-stop`, and the agent raises `ServerStopped` out of an
infer. `run()` already treats that as the happy path rather than a failure.

It was not happy enough. The exception left `rollout()` before the summary,
so a run that ended exactly as intended reported no timings at all - and with
them went the only client-side record of how many environment steps it took,
which is what a server's `global_step` has to be reconciled against.

These drive the real rollout loop rather than a stand-in for it, because the
defect was in the loop's exit path and a mocked rollout cannot have one.
"""

from __future__ import annotations

import pytest
from loguru import logger

from plugrl_env_client.agent.websocket_env_client_agent import ServerStopped
from plugrl_env_client.runner.rollout import rollout

from test_protocol_alternation import _StaggeredEnv, _WireRecorder


class _StopsAfter(_WireRecorder):
    """A server that ends the run after `n` infers, the way a finished one does."""

    def __init__(self, horizon, stop_after):
        super().__init__(horizon=horizon)
        self.stop_after = stop_after
        self.infers = 0

    def infer(self, obs, *, env_indices, step_ids):
        self.infers += 1
        if self.infers > self.stop_after:
            raise ServerStopped("Server requested env client shutdown.")
        return super().infer(obs, env_indices=env_indices, step_ids=step_ids)


def _messages_from(stop_after):
    captured: list[str] = []
    sink = logger.add(lambda m: captured.append(m.record["message"]), level="INFO")
    try:
        with pytest.raises(ServerStopped):
            rollout(
                _StaggeredEnv(episode_lengths=(2, 6)),
                _StopsAfter(horizon=1, stop_after=stop_after),
                num_episodes=100,
                replan_steps=1,
                num_envs=2,
                seed=0,
            )
    finally:
        logger.remove(sink)
    return captured


def test_a_server_stop_still_reports_the_final_summary():
    messages = _messages_from(stop_after=5)

    finals = [m for m in messages if m.startswith("Final rollout timing summary")]
    assert finals, f"no final summary was logged; got {messages}"


def test_the_summary_carries_the_steps_actually_taken():
    """The count is the point: it is what reconciles against the server."""
    messages = _messages_from(stop_after=5)

    final = next(m for m in messages if m.startswith("Final rollout timing summary"))
    steps = int(final.split("env_steps=")[1].split()[0])
    assert steps > 0, f"summary reported {steps} steps: {final}"
