"""The rollout loop already measures where its time goes. Until now it only
printed that as a log line, which is fine for watching one run and useless for
comparing ninety.

These cover the structured form: a timing record that survives into
summary.json, so a sweep can be aggregated without parsing logs.
"""

import json

import pytest

from plugrl_env_client.recorder import Recorder, RecorderArgs
from plugrl_env_client.runner.rollout import RolloutTiming


def test_timing_starts_at_zero():
    t = RolloutTiming()
    assert t.env_steps == 0
    assert t.as_dict()["effective_fps"] == 0.0


def test_timing_accumulates():
    t = RolloutTiming()
    t.infer_wait += 1.0
    t.env_step += 3.0
    t.env_steps += 12
    t.infer_calls += 4

    d = t.as_dict()
    assert d["infer_wait_s"] == 1.0
    assert d["env_step_s"] == 3.0
    assert d["env_steps"] == 12
    assert d["infer_calls"] == 4


def test_effective_fps_divides_by_collect_time():
    t = RolloutTiming()
    t.env_steps = 100
    t.infer_wait = 1.0
    t.env_step = 1.0  # collect time 2.0s -> 50 steps/s
    assert t.as_dict()["effective_fps"] == pytest.approx(50.0)


def test_fractions_sum_to_one_and_show_the_bottleneck():
    """The point of the whole exercise: which stage dominates."""
    t = RolloutTiming()
    t.infer_wait = 6.0
    t.env_step = 3.0
    t.infer_obs_pack = 1.0
    d = t.as_dict()

    assert d["infer_wait_frac"] == pytest.approx(0.6)
    assert d["env_step_frac"] == pytest.approx(0.3)
    assert d["infer_obs_pack_frac"] == pytest.approx(0.1)
    total = sum(v for k, v in d.items() if k.endswith("_frac"))
    assert total == pytest.approx(1.0)


def test_fractions_are_zero_when_nothing_ran():
    """No division by zero on an empty rollout."""
    d = RolloutTiming().as_dict()
    assert all(v == 0.0 for k, v in d.items() if k.endswith("_frac"))


def test_timing_reaches_summary_json(tmp_path):
    recorder = Recorder(
        RecorderArgs(),
        exp_name="test-exp",
        output_dir=tmp_path,
        num_envs=1,
        process_id=0,
        total_processes=1,
    )
    timing = RolloutTiming()
    timing.env_steps = 20
    timing.env_step = 2.0
    recorder.record_timing(timing)
    recorder.close()

    summary = json.loads(recorder.summary_path.read_text())
    assert summary["timing"]["env_steps"] == 20
    assert summary["timing"]["env_step_s"] == 2.0
    assert summary["timing"]["effective_fps"] == pytest.approx(10.0)


def test_summary_written_without_timing(tmp_path):
    """Recording timing is optional; a recorder that never sees one still works."""
    recorder = Recorder(
        RecorderArgs(),
        exp_name="test-exp",
        output_dir=tmp_path,
        num_envs=1,
        process_id=0,
        total_processes=1,
    )
    recorder.close()
    summary = json.loads(recorder.summary_path.read_text())
    assert summary["timing"] is None
