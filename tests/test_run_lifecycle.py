"""How a run ends.

There are two ways for collection to stop, and only one of them is a
failure. The server closing with `plugrl-server-stop` (SPEC.md section 7.1)
means the algorithm took every step it was asked for - the documented happy
path of "run a server for N steps, point a client at it". That used to
escape `run()` as an unhandled `ServerStopped`, so a successful run ended in
a traceback and exit 1, and under `run_multiprocess` the non-zero child exit
tore its siblings down mid-episode.

Anything else still has to propagate, or a broken client looks like a
finished one.
"""

import importlib
import types

import pytest

from plugrl_env_client.agent.websocket_env_client_agent import ServerStopped

# `plugrl_env_client.runner.run` is shadowed by the function of the same name
# that the package re-exports, so the module has to be asked for by name.
run_module = importlib.import_module("plugrl_env_client.runner.run")


class _Env:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Recorder:
    total_episode_count = 3

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Args:
    pass_proc_id = False
    use_env_lock = False
    replan_steps = 4
    seed = 0


@pytest.fixture
def harness(monkeypatch):
    """run() with its environment, recorder and rollout replaced."""
    env, recorder = _Env(), _Recorder()
    monkeypatch.setattr(run_module, "_make_env", lambda *a, **k: env)
    monkeypatch.setattr(run_module, "Recorder", lambda *a, **k: recorder)
    monkeypatch.setattr(run_module, "report_server_metadata", lambda *a, **k: None)

    def invoke(rollout_impl):
        monkeypatch.setattr(run_module, "rollout", rollout_impl)
        run_module.run(
            _Args(),
            lambda: types.SimpleNamespace(),
            env_config=None,
            num_envs=1,
            num_episodes=5,
            exp_name="t",
            output_dir="runs",
            recorder_args=None,
        )

    return invoke, env, recorder


def test_a_server_stop_ends_the_run_without_raising(harness):
    invoke, env, recorder = harness

    def stops(*_a, **_k):
        raise ServerStopped("Server requested env client shutdown.")

    invoke(stops)  # must not raise

    assert env.closed and recorder.closed, "cleanup still has to happen"


def test_any_other_failure_still_propagates(harness):
    """A client that died has to look different from one that finished."""
    invoke, env, recorder = harness

    def breaks(*_a, **_k):
        raise ValueError("Expected action shape tail (), got (7,)")

    with pytest.raises(ValueError, match="Expected action shape"):
        invoke(breaks)

    assert env.closed and recorder.closed


def test_a_normal_return_closes_everything_too(harness):
    invoke, env, recorder = harness

    invoke(lambda *_a, **_k: None)

    assert env.closed and recorder.closed


def test_the_stop_message_says_how_far_it_got(harness, capsys):
    """`0/5 episodes` is the useful part: the run ended early, and by how much."""
    invoke, _, _ = harness
    from loguru import logger

    messages = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    try:

        def stops(*_a, **_k):
            raise ServerStopped("stop")

        invoke(stops)
    finally:
        logger.remove(sink)

    assert any("3/5 episodes" in message for message in messages), messages
