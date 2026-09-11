"""What the client does with the server's metadata message.

SPEC.md section 5.1 says a client must not require any key in it. That cuts
both ways: the client may use what is there, and must keep working when
none of it is. These tests pin both halves, because the useful half - a
warning about a `replan_steps` that the policy cannot satisfy - is only
worth having if it cannot itself become a reason the client refuses to
start.
"""

import pytest
from loguru import logger

from plugrl_env_client.runner.run import report_server_metadata


class _Agent:
    def __init__(self, metadata):
        self._metadata = metadata

    def get_server_metadata(self):
        return self._metadata


class _AgentWithoutMetadata:
    """An agent predating get_server_metadata, or a test double."""


@pytest.fixture
def captured():
    messages = []
    sink_id = logger.add(lambda m: messages.append(m.record), level="DEBUG")
    yield messages
    logger.remove(sink_id)


def _levels(records):
    return [r["level"].name for r in records]


def _text(records):
    return " ".join(r["message"] for r in records)


class TestItNeverBlocksStartup:
    @pytest.mark.parametrize(
        "agent",
        [
            pytest.param(_Agent({}), id="empty-metadata"),
            pytest.param(_Agent(None), id="none-metadata"),
            pytest.param(_AgentWithoutMetadata(), id="no-such-method"),
        ],
    )
    def test_a_server_that_says_nothing_is_fine(self, agent, captured):
        report_server_metadata(agent, replan_steps=4)

        assert "WARNING" not in _levels(captured)

    def test_a_horizon_of_the_wrong_type_is_ignored(self, captured):
        """Descriptive keys are not validated, so they may be anything."""
        report_server_metadata(_Agent({"action_horizon": "four"}), replan_steps=8)

        assert "WARNING" not in _levels(captured)

    def test_no_replan_steps_means_nothing_to_check(self, captured):
        report_server_metadata(_Agent({"action_horizon": 4}), replan_steps=None)

        assert "WARNING" not in _levels(captured)


class TestTheOneCheckItMakes:
    def test_asking_for_more_steps_than_the_policy_plans_warns(self, captured):
        report_server_metadata(
            _Agent({"action_horizon": 4, "policy": "DummyPolicy"}), replan_steps=8
        )

        assert "WARNING" in _levels(captured)
        message = _text(captured)
        assert "8" in message and "4" in message and "DummyPolicy" in message

    @pytest.mark.parametrize("replan_steps", [1, 3, 4])
    def test_a_horizon_that_fits_is_silent(self, replan_steps, captured):
        report_server_metadata(_Agent({"action_horizon": 4}), replan_steps=replan_steps)

        assert "WARNING" not in _levels(captured)

    def test_the_metadata_is_logged_so_a_run_records_what_it_talked_to(self, captured):
        report_server_metadata(
            _Agent({"policy": "DummyPolicy", "action_dim": 7}), replan_steps=1
        )

        assert "DummyPolicy" in _text(captured)
