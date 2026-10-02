"""SPEC section 10.1: an infer leaves out the observations the server holds.

Every observation used to cross the link twice: in the feedback that ends a
chunk, and again in the next infer. When the server lists
`reuse-feedback-obs`, the agent marks in `reuse` each row whose observation
the server already has, and sends only the others. The server holds an env's
observation from its last feedback on this connection, unless that feedback
ended the episode, so these tests pin down when a row may be left out and
when it must not be.
"""

import numpy as np

from plugrl_env_client.agent.websocket_env_client_agent import (
    ConnectionReplaced,
    WebSocketEnvClientAgent,
)
from plugrl_protocol import msgpack_numpy
from plugrl_protocol.reuse import REUSE_FEEDBACK_OBS

OFFERED = {"features": [REUSE_FEEDBACK_OBS]}


class _Socket:
    """Records every message sent; answers each infer with an empty action."""

    def __init__(self):
        self.sent: list[dict] = []

    def send(self, data):
        self.sent.append(msgpack_numpy.unpackb(data))

    def recv(self):
        return msgpack_numpy.Packer().pack({"message_type": "action", "data": {}})

    def close(self):
        pass

    def infers(self) -> list[dict]:
        return [m for m in self.sent if m["message_type"] == "infer"]


def _agent(monkeypatch, metadata=OFFERED, **kwargs):
    sockets: list[_Socket] = []

    def fake_wait(self):
        sockets.append(_Socket())
        return sockets[-1], dict(metadata)

    monkeypatch.setattr(WebSocketEnvClientAgent, "_wait_for_server", fake_wait)
    agent = WebSocketEnvClientAgent(host="127.0.0.1", port=1, **kwargs)
    return agent, sockets


def _obs(*values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "images": {
            "cam": np.broadcast_to(values[:, None, None], (len(values), 2, 2)).copy()
        },
        "states": {"obs": values[:, None]},
        "text": np.asarray(["go"] * len(values)),
    }


def _infer(agent, obs, envs):
    agent.infer(obs, env_indices=np.asarray(envs), step_ids=np.zeros(len(envs)))


def _feedback(agent, obs, envs, done=None):
    done = np.asarray(done or [False] * len(envs))
    agent.feedback(
        obs,
        rewards=np.ones(len(envs), np.float32),
        terminated=done,
        truncated=np.zeros(len(envs), bool),
        info={},
        env_indices=np.asarray(envs),
        step_ids=np.zeros(len(envs)),
    )


def test_the_rows_the_server_holds_are_left_out(monkeypatch):
    agent, sockets = _agent(monkeypatch)
    _infer(agent, _obs(1, 2), [0, 1])
    _feedback(agent, _obs(10, 11), [0, 1])

    _infer(agent, _obs(10, 11), [0, 1])

    first, second = sockets[0].infers()
    assert "reuse" not in first  # nothing was held yet
    assert second["reuse"].tolist() == [True, True]
    assert second["data"]["states"]["obs"].shape == (0, 1)
    assert second["data"]["images"]["cam"].shape == (0, 2, 2)


def test_a_row_unlike_the_one_fed_back_is_sent(monkeypatch):
    agent, sockets = _agent(monkeypatch)
    _infer(agent, _obs(1, 2), [0, 1])
    _feedback(agent, _obs(10, 11), [0, 1])

    _infer(agent, _obs(10, 99), [0, 1])

    second = sockets[0].infers()[1]
    assert second["reuse"].tolist() == [True, False]
    assert second["data"]["states"]["obs"].tolist() == [[99.0]]
    assert second["data"]["text"].tolist() == ["go"]


def test_an_env_whose_feedback_ended_its_episode_sends_its_reset(monkeypatch):
    """That feedback carried the terminal observation, even if the reset looks the same."""
    agent, sockets = _agent(monkeypatch)
    _infer(agent, _obs(1, 2), [0, 1])
    _feedback(agent, _obs(10, 11), [0, 1], done=[True, False])

    _infer(agent, _obs(10, 11), [0, 1])  # env 0's reset equals its terminal obs

    second = sockets[0].infers()[1]
    assert second["reuse"].tolist() == [False, True]
    assert second["data"]["states"]["obs"].tolist() == [[10.0]]


def test_nothing_is_left_out_unless_the_server_offers_it(monkeypatch):
    agent, sockets = _agent(monkeypatch, metadata={"protocol_version": 1})
    _infer(agent, _obs(1), [0])
    _feedback(agent, _obs(10), [0])
    _infer(agent, _obs(10), [0])

    assert all("reuse" not in m for m in sockets[0].infers())


def test_it_can_be_turned_off(monkeypatch):
    agent, sockets = _agent(monkeypatch, reuse_feedback_obs=False)
    _infer(agent, _obs(1), [0])
    _feedback(agent, _obs(10), [0])
    _infer(agent, _obs(10), [0])

    assert all("reuse" not in m for m in sockets[0].infers())


def test_a_new_connection_holds_nothing(monkeypatch):
    """SPEC section 10.1: a client MUST NOT reuse just after a reconnect."""
    agent, sockets = _agent(monkeypatch)
    _infer(agent, _obs(1), [0])
    _feedback(agent, _obs(10), [0])
    agent._close_connection()  # the server went away
    try:
        _infer(agent, _obs(10), [0])
    except ConnectionReplaced:
        pass  # the caller drops its chunks and asks again
    _infer(agent, _obs(10), [0])

    assert len(sockets) == 2
    assert all("reuse" not in m for m in sockets[1].infers())


def test_an_observation_changed_in_place_after_its_feedback_is_sent(monkeypatch):
    """The agent keeps its own copy, so a caller reusing buffers cannot fool it."""
    agent, sockets = _agent(monkeypatch)
    obs = _obs(10)
    _infer(agent, _obs(1), [0])
    _feedback(agent, obs, [0])
    obs["states"]["obs"][0, 0] = 42.0  # the same buffer, a new step

    _infer(agent, obs, [0])

    second = sockets[0].infers()[1]
    assert "reuse" not in second
    assert second["data"]["states"]["obs"].tolist() == [[42.0]]
