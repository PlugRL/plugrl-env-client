import dataclasses
import time

import gymnasium as gym
import numpy as np
from loguru import logger

from plugrl_env_client.agent.websocket_env_client_agent import WebSocketEnvClientAgent
from plugrl_env_client.recorder import Recorder
from plugrl_env_client.utils.rollout import _get_action_spec, _select_info, _select_obs


@dataclasses.dataclass
class RolloutTiming:
    """Where a rollout's wall clock actually went.

    The interesting output is the fractions: whether a configuration is
    bounded by the environment, by waiting on the server, or by packing
    observations differs by orders of magnitude across environment families,
    and that is not visible from throughput alone.
    """

    infer_wait: float = 0.0
    infer_obs_pack: float = 0.0
    env_step: float = 0.0
    feedback: float = 0.0
    feedback_obs_pack: float = 0.0
    feedback_info_pack: float = 0.0
    infer_calls: int = 0
    feedback_calls: int = 0
    env_steps: int = 0

    # The stages that make up one collection cycle. feedback_obs_pack and
    # feedback_info_pack are already inside feedback, so counting them here
    # too would double-count them.
    _STAGES = ("infer_wait", "infer_obs_pack", "env_step", "feedback")

    @property
    def collect_time(self) -> float:
        return sum(getattr(self, stage) for stage in self._STAGES)

    def as_dict(self) -> dict:
        total = self.collect_time
        out = {
            "env_steps": self.env_steps,
            "infer_calls": self.infer_calls,
            "feedback_calls": self.feedback_calls,
            "infer_wait_s": self.infer_wait,
            "infer_obs_pack_s": self.infer_obs_pack,
            "env_step_s": self.env_step,
            "feedback_s": self.feedback,
            "feedback_obs_pack_s": self.feedback_obs_pack,
            "feedback_info_pack_s": self.feedback_info_pack,
            "collect_time_s": total,
            "effective_fps": (self.env_steps / total) if total > 0 else 0.0,
        }
        for stage in self._STAGES:
            out[f"{stage}_frac"] = (getattr(self, stage) / total) if total > 0 else 0.0
        return out


def rollout(
    env: gym.vector.VectorEnv,
    agent: WebSocketEnvClientAgent,
    *,
    num_episodes: int,
    replan_steps: int | None,
    num_envs: int,
    recorder: Recorder | None = None,
    seed: int | None = None,
) -> None:
    # Only the first reset carries the seed; later resets continue the stream
    # the env already established, which is what makes the whole rollout - not
    # just its first episode - reproducible.
    obs, info = env.reset(seed=seed)
    if recorder is not None:
        recorder.on_reset(obs, info, reset_indices=None)

    plan_capacity = int(replan_steps or 0)
    expected_action_shape, expected_action_dtype = _get_action_spec(
        env, num_envs=num_envs
    )

    action_plan = np.empty(
        (num_envs, plan_capacity) + expected_action_shape,
        dtype=expected_action_dtype,
    )
    plan_pos = np.zeros((num_envs,), dtype=np.int32)
    plan_len = np.zeros((num_envs,), dtype=np.int32)
    chunk_reward = np.zeros((num_envs,), dtype=np.float32)

    step_id = np.zeros((num_envs,), dtype=np.int64)

    timing = RolloutTiming()
    last_timing_log_at = time.perf_counter()

    def log_timing_summary(final: bool = False) -> None:
        logger.info(
            "{} rollout timing summary: env_steps={} infer_calls={} feedback_calls={} infer_wait={:.3f}s infer_obs_pack={:.3f}s env_step={:.3f}s feedback_total={:.3f}s feedback_obs_pack={:.3f}s feedback_info_pack={:.3f}s effective_fps={:.2f}",
            "Final" if final else "Intermediate",
            timing.env_steps,
            timing.infer_calls,
            timing.feedback_calls,
            timing.infer_wait,
            timing.infer_obs_pack,
            timing.env_step,
            timing.feedback,
            timing.feedback_obs_pack,
            timing.feedback_info_pack,
            timing.as_dict()["effective_fps"],
        )

    finished_episodes = 0
    # The loop also ends when the server says it is done, which raises
    # ServerStopped out of an agent call. That is the happy path - the
    # algorithm took every step it was asked for - and it used to skip the
    # summary below, so a run that finished normally reported no timings at
    # all and left nothing to reconcile the client's step count against the
    # server's. The summary belongs to the rollout either way.
    try:
        while finished_episodes < num_episodes:
            need_infer = np.nonzero(plan_pos >= plan_len)[0]
            if need_infer.size:
                infer_obs_pack_started_at = time.perf_counter()
                obs_msg = dataclasses.asdict(_select_obs(obs, need_infer))
                timing.infer_obs_pack += time.perf_counter() - infer_obs_pack_started_at

                infer_wait_started_at = time.perf_counter()
                action_chunk = agent.infer(
                    obs_msg,
                    env_indices=need_infer,
                    step_ids=step_id[need_infer],
                )["action"]
                timing.infer_wait += time.perf_counter() - infer_wait_started_at
                timing.infer_calls += 1

                steps = replan_steps or len(action_chunk)
                if len(action_chunk) < steps:
                    raise ValueError(
                        f"replan_steps={replan_steps} exceeds predicted steps={len(action_chunk)}"
                    )

                a = np.asarray(action_chunk[:steps], dtype=expected_action_dtype)
                if a.shape[:2] != (steps, need_infer.size):
                    raise ValueError(
                        f"Expected action shape ({steps}, {need_infer.size}, da), got {a.shape}"
                    )
                if a.shape[2:] != expected_action_shape:
                    raise ValueError(
                        f"Expected action shape tail {expected_action_shape}, got {a.shape[2:]}"
                    )

                need_capacity = max(int(action_plan.shape[1]), int(steps))
                need_shape = (num_envs, need_capacity) + expected_action_shape
                if action_plan.shape != need_shape:
                    new_plan = np.empty(need_shape, dtype=expected_action_dtype)
                    cap = min(int(action_plan.shape[1]), need_capacity)
                    if cap:
                        new_plan[:, :cap, ...] = action_plan[:, :cap, ...]
                    action_plan = new_plan

                action_plan[need_infer, :steps, ...] = a.swapaxes(0, 1)
                plan_pos[need_infer] = 0
                plan_len[need_infer] = steps

            if np.any(plan_pos >= plan_len):
                raise RuntimeError("Action plan is not ready for all envs")

            actions = action_plan[np.arange(num_envs), plan_pos]
            env_step_started_at = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(actions)
            timing.env_step += time.perf_counter() - env_step_started_at
            timing.env_steps += num_envs
            if recorder is not None:
                recorder.on_step(obs, reward, terminated, truncated, info)

            plan_pos += 1

            reward = np.asarray(reward, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=np.bool_)
            truncated = np.asarray(truncated, dtype=np.bool_)
            done = np.logical_or(terminated, truncated)

            chunk_reward += reward

            done_indices = np.nonzero(done)[0]
            if done_indices.size:
                plan_pos[done_indices] = plan_len[done_indices]

            feedback_indices = np.nonzero(plan_pos >= plan_len)[0]
            if feedback_indices.size:
                feedback_started_at = time.perf_counter()
                feedback_obs_pack_started_at = time.perf_counter()
                feedback_obs = dataclasses.asdict(_select_obs(obs, feedback_indices))
                timing.feedback_obs_pack += (
                    time.perf_counter() - feedback_obs_pack_started_at
                )
                feedback_info_pack_started_at = time.perf_counter()
                feedback_info = _select_info(info, feedback_indices, num_envs=num_envs)
                timing.feedback_info_pack += (
                    time.perf_counter() - feedback_info_pack_started_at
                )
                agent.feedback(
                    obs=feedback_obs,
                    rewards=chunk_reward[feedback_indices],
                    terminated=terminated[feedback_indices],
                    truncated=truncated[feedback_indices],
                    info=feedback_info,
                    env_indices=feedback_indices,
                    step_ids=step_id[feedback_indices],
                )
                timing.feedback += time.perf_counter() - feedback_started_at
                timing.feedback_calls += 1
                chunk_reward[feedback_indices] = 0.0
                step_id[feedback_indices] += 1

            if done_indices.size:
                if recorder is not None:
                    recorder.on_episode_done(done_indices, obs, info)
                finished_episodes += done_indices.size
                # Only reset if the loop is going to use what comes back. A reset
                # after the last episode costs a simulator a wasted rollout, and
                # costs a real robot a pointless move back to its home pose - for
                # an observation nothing will ever read.
                if finished_episodes < num_episodes:
                    obs, info = env.reset(options={"reset_indices": done_indices})
                    if recorder is not None:
                        recorder.on_reset(obs, info, reset_indices=done_indices)
                    step_id[done_indices] = 0
            now = time.perf_counter()
            if now - last_timing_log_at >= 30.0:
                log_timing_summary(final=False)
                last_timing_log_at = now

    finally:
        log_timing_summary(final=True)
        if recorder is not None:
            recorder.record_timing(timing)
