"""A transition cannot survive the connection it was started on.

The server builds a transition across two messages: the infer it answered
tells it the previous observation and the policy step state, and the feedback
that follows completes it. Both halves of that state live in the connection
handler, so a new connection starts with neither.

That makes a resent feedback actively harmful rather than merely late. The
server has nothing to attach it to, builds the transition out of an empty
observation, and stores it - silently, because nothing about it is an error.
SPEC section 7.6 therefore says a client MUST drop held feedback across a
reconnect, and these tests hold the client to it.

The infer path is the opposite case and is covered here too: a fresh infer on
a fresh connection is exactly how the two sides get back in step, so that one
must still retry - but only once the caller has dropped the chunks it was
still executing, which came from the old connection. The agent says so by
raising ConnectionReplaced, from a failed feedback and from the first infer
after a reconnect.
"""

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from plugrl_env_client.agent.websocket_env_client_agent import (
    ConnectionReplaced,
    WebSocketEnvClientAgent,
)
from plugrl_protocol.websocket_protocol import SERVER_RESYNC_REASON, SERVER_STOP_REASON


class _Socket:
    """Fails the first send with `exc`, records every send attempt."""

    def __init__(self, exc):
        self.exc = exc
        self.sends = 0
        self.closed = False

    def send(self, _data):
        self.sends += 1
        if self.sends == 1 and self.exc is not None:
            raise self.exc
        return None

    def recv(self):
        raise AssertionError("no test here should get as far as a reply")


class _AnsweringSocket(_Socket):
    """Answers every infer with an empty action map."""

    def __init__(self):
        super().__init__(None)

    def recv(self):
        from plugrl_protocol import msgpack_numpy

        return msgpack_numpy.Packer().pack({"message_type": "action", "data": {}})

    def close(self):
        self.closed = True


def _agent(monkeypatch, socket, *, fresh=_Socket, **kwargs):
    """An agent wired to `socket`, whose reconnect is observable, not real.

    It is in the state a feedback is sent from: the infer it completes was
    answered on this connection.
    """
    reconnects = []

    def fake_wait(self):
        reconnects.append(True)
        return (fresh() if fresh is _AnsweringSocket else fresh(None)), {}

    monkeypatch.setattr(WebSocketEnvClientAgent, "_wait_for_server", fake_wait)
    agent = WebSocketEnvClientAgent(host="127.0.0.1", port=1, **kwargs)
    agent._ws = socket
    agent._actions_from = agent._connection_id
    reconnects.clear()  # the constructor's own connect is not a reconnect
    return agent, reconnects


def _feedback(agent):
    agent.feedback(
        {},
        rewards=[0.0],
        terminated=[False],
        truncated=[False],
        info={},
        env_indices=[0],
        step_ids=[0],
    )


KEEPALIVE = ConnectionClosedError(Close(1011, "keepalive ping timeout"), None)
PLAIN_CLOSE = ConnectionClosedOK(Close(1000, ""), None)
RESYNC = ConnectionClosedOK(Close(1001, SERVER_RESYNC_REASON), None)
STOP = ConnectionClosedOK(Close(1001, SERVER_STOP_REASON), None)


@pytest.mark.parametrize(
    "exc, label",
    [
        (KEEPALIVE, "a keepalive timeout, which is the case that found this"),
        (PLAIN_CLOSE, "an ordinary close with no reason given"),
        (RESYNC, "an explicit resync request"),
    ],
)
def test_feedback_is_dropped_not_resent(monkeypatch, exc, label):
    socket = _Socket(exc)
    agent, reconnects = _agent(monkeypatch, socket)

    with pytest.raises(ConnectionReplaced):  # the caller drops its chunks too
        _feedback(agent)

    assert socket.sends == 1, f"feedback was resent after {label}"
    assert reconnects == [], f"the feedback path reconnected after {label}"


def test_a_stop_the_client_waits_out_drops_the_feedback_too(monkeypatch):
    """This path resent it, the one break of SPEC section 7.6 left."""
    socket = _Socket(STOP)
    agent, reconnects = _agent(monkeypatch, socket, reconnect_on_server_stop=True)

    with pytest.raises(ConnectionReplaced):
        _feedback(agent)

    assert socket.sends == 1, "feedback was resent after a stop"
    assert reconnects == [], "the feedback path reconnected after a stop"


def test_feedback_never_reconnects_to_be_sent(monkeypatch):
    """With the connection already gone, there is nothing to send it on."""
    agent, reconnects = _agent(monkeypatch, _Socket(None))
    agent._close_connection()

    with pytest.raises(ConnectionReplaced):
        _feedback(agent)

    assert reconnects == []


def test_infer_still_retries_after_a_drop(monkeypatch):
    """The mirror image: a fresh infer is how the two sides resynchronise."""
    socket = _Socket(KEEPALIVE)
    agent, reconnects = _agent(monkeypatch, socket)
    agent._actions_from = None  # nothing in flight yet

    with pytest.raises(AssertionError):  # the retry reaches recv() and stops
        agent.infer({}, env_indices=[0], step_ids=[0])

    assert reconnects == [True], "infer should have reconnected and retried"


def test_the_first_infer_after_a_reconnect_tells_the_caller(monkeypatch):
    """Chunks answered on the old connection must not be finished on the new one.

    The caller is told before anything is sent, so it can drop them and ask
    again for every env; that second infer goes out.
    """
    socket = _Socket(KEEPALIVE)
    agent, reconnects = _agent(monkeypatch, socket, fresh=_AnsweringSocket)

    with pytest.raises(ConnectionReplaced):
        agent.infer({}, env_indices=[0], step_ids=[0])
    assert reconnects == [True]
    assert agent._ws.sends == 0, "an infer went out before the caller re-planned"

    agent.infer({}, env_indices=[0, 1], step_ids=[0, 0])
    assert agent._ws.sends == 1
