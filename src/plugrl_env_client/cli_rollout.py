import dataclasses

import gymnasium as gym
import numpy as np

from plugrl_env_client.cli_utils import _get_action_spec, _select_info, _select_obs
from plugrl_env_client.websocket_env_client_agent import WebSocketEnvClientAgent


def rollout(
    env: gym.vector.VectorEnv,
    agent: WebSocketEnvClientAgent,
    *,
    num_episodes: int,
    replan_steps: int | None,
    num_envs: int,
) -> None:
    obs, info = env.reset()

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

    finished_episodes = 0
    while finished_episodes < num_episodes:
        need_infer = np.nonzero(plan_pos >= plan_len)[0]
        if need_infer.size:
            obs_msg = dataclasses.asdict(_select_obs(obs, need_infer))
            action_chunk = agent.infer(obs_msg, need_infer)["action"]

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
        obs, reward, terminated, truncated, info = env.step(actions)

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
            agent.feedback(
                obs=dataclasses.asdict(_select_obs(obs, feedback_indices)),
                rewards=chunk_reward[feedback_indices],
                terminated=terminated[feedback_indices],
                truncated=truncated[feedback_indices],
                info=_select_info(info, feedback_indices, num_envs=num_envs),
                env_indices=feedback_indices,
            )
            chunk_reward[feedback_indices] = 0.0

        if done_indices.size:
            finished_episodes += done_indices.size
            obs, info = env.reset(options={"reset_indices": done_indices})
