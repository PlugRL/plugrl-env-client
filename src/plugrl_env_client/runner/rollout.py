import dataclasses
import time

import gymnasium as gym
import numpy as np
from loguru import logger

from plugrl_env_client.agent.websocket_env_client_agent import WebSocketEnvClientAgent
from plugrl_env_client.recorder import Recorder
from plugrl_env_client.utils.rollout import _get_action_spec, _select_info, _select_obs


def rollout(
    env: gym.vector.VectorEnv,
    agent: WebSocketEnvClientAgent,
    *,
    num_episodes: int,
    replan_steps: int | None,
    num_envs: int,
    recorder: Recorder | None = None,
) -> None:
    obs, info = env.reset()
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

    infer_wait_total = 0.0
    infer_obs_pack_total = 0.0
    env_step_total = 0.0
    feedback_total = 0.0
    feedback_obs_pack_total = 0.0
    feedback_info_pack_total = 0.0
    infer_call_count = 0
    feedback_call_count = 0
    total_env_steps = 0
    last_timing_log_at = time.perf_counter()

    def log_timing_summary(final: bool = False) -> None:
        total_collect_time = (
            infer_wait_total + infer_obs_pack_total + env_step_total + feedback_total
        )
        logger.info(
            "{} rollout timing summary: env_steps={} infer_calls={} feedback_calls={} infer_wait={:.3f}s infer_obs_pack={:.3f}s env_step={:.3f}s feedback_total={:.3f}s feedback_obs_pack={:.3f}s feedback_info_pack={:.3f}s effective_fps={:.2f}",
            "Final" if final else "Intermediate",
            total_env_steps,
            infer_call_count,
            feedback_call_count,
            infer_wait_total,
            infer_obs_pack_total,
            env_step_total,
            feedback_total,
            feedback_obs_pack_total,
            feedback_info_pack_total,
            0.0 if total_collect_time <= 0 else total_env_steps / total_collect_time,
        )

    finished_episodes = 0
    while finished_episodes < num_episodes:
        need_infer = np.nonzero(plan_pos >= plan_len)[0]
        if need_infer.size:
            infer_obs_pack_started_at = time.perf_counter()
            obs_msg = dataclasses.asdict(_select_obs(obs, need_infer))
            infer_obs_pack_total += time.perf_counter() - infer_obs_pack_started_at

            infer_wait_started_at = time.perf_counter()
            action_chunk = agent.infer(
                obs_msg,
                env_indices=need_infer,
                step_ids=step_id[need_infer],
            )["action"]
            infer_wait_total += time.perf_counter() - infer_wait_started_at
            infer_call_count += 1

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
        env_step_total += time.perf_counter() - env_step_started_at
        total_env_steps += num_envs
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
            feedback_obs_pack_total += (
                time.perf_counter() - feedback_obs_pack_started_at
            )
            feedback_info_pack_started_at = time.perf_counter()
            feedback_info = _select_info(info, feedback_indices, num_envs=num_envs)
            feedback_info_pack_total += (
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
            feedback_total += time.perf_counter() - feedback_started_at
            feedback_call_count += 1
            chunk_reward[feedback_indices] = 0.0
            step_id[feedback_indices] += 1

        if done_indices.size:
            if recorder is not None:
                recorder.on_episode_done(done_indices, obs, info)
            finished_episodes += done_indices.size
            obs, info = env.reset(options={"reset_indices": done_indices})
            if recorder is not None:
                recorder.on_reset(obs, info, reset_indices=done_indices)
            step_id[done_indices] = 0
        now = time.perf_counter()
        if now - last_timing_log_at >= 30.0:
            log_timing_summary(final=False)
            last_timing_log_at = now

    log_timing_summary(final=True)
