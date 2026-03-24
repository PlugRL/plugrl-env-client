from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from loguru import logger

from plugrl_env_client.envs.base_env import Observation
from plugrl_env_client.recorder.args import RecorderArgs
from plugrl_env_client.utils.recorder import (
    append_jsonl,
    ensure_rgb,
    make_grid,
    phase_stats,
    to_builtin,
    write_json,
)
from plugrl_env_client.utils.rollout import _select_obs


class Recorder:
    def __init__(
        self,
        args: RecorderArgs,
        *,
        exp_name: str,
        output_dir: Path,
        num_envs: int,
        process_id: int | None,
        total_processes: int | None,
    ) -> None:
        self.args = args
        self.exp_name = exp_name
        self.output_dir = Path(output_dir)
        self.num_envs = int(num_envs)
        self.process_id = process_id
        self.total_processes = total_processes
        self.proc_index = 0 if process_id is None else int(process_id)
        self.should_write = (not args.thread0_only) or self.proc_index == 0
        self.enabled = self.should_write and int(args.episode_freq) > 0
        self.effective_episode_freq = int(args.episode_freq)

        self.full_rollout_video = False
        if self.enabled and args.record_video and args.record_full_rollout:
            if int(args.episode_freq) == 1:
                self.full_rollout_video = True
            else:
                logger.warning(
                    "Ignoring record_full_rollout because recorder.episode_freq != 1."
                )

        self.root_dir = self.output_dir / "rollout" / f"proc_{self.proc_index:03d}"

        self.metrics_dir = self.root_dir / "metrics"
        self.sampled_dir = self.root_dir / "sampled"
        self.full_videos_dir = self.root_dir / "videos" / "full" / "images"

        self.episode_metrics_path = self.metrics_dir / "episode_metrics.jsonl"
        self.obs_stats_path = self.metrics_dir / "obs_stats.jsonl"
        self.summary_path = self.root_dir / "summary.json"

        self.completed_episodes = 0
        self.total_episode_count = 0
        self.metric_window = max(1, int(args.metric_window))
        self.return_window: deque[float] = deque(maxlen=self.metric_window)
        self.success_window: deque[float] = deque(maxlen=self.metric_window)

        self.current_episode_ids = np.zeros((self.num_envs,), dtype=np.int64)
        self.first_obs_by_env: list[Observation | None] = [None] * self.num_envs
        self.env0_image_frames: dict[str, list[np.ndarray]] = {}
        self.full_video_writers: dict[str, Any] = {}

        if not self.should_write:
            return

        self.root_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            self.root_dir / "manifest.json",
            {
                "exp_name": exp_name,
                "process_id": self.proc_index,
                "total_processes": total_processes,
                "num_envs": self.num_envs,
                "args": asdict(args),
                "metric_window": self.metric_window,
                "effective_episode_freq": self.effective_episode_freq,
                "full_rollout_video": self.full_rollout_video,
            },
        )

    def on_reset(
        self,
        obs: Observation,
        info: dict[str, Any],
        *,
        reset_indices: np.ndarray | None,
    ) -> None:
        del info
        indices = (
            np.arange(self.num_envs, dtype=np.int64)
            if reset_indices is None
            else np.asarray(reset_indices, dtype=np.int64)
        )
        for env_idx in indices.tolist():
            self.first_obs_by_env[env_idx] = _select_obs(obs, [env_idx])
            self.current_episode_ids[env_idx] += 1

        if not self.should_write:
            return

        if reset_indices is None and int(0) < self.num_envs:
            self._reset_env0_video(obs)
        elif reset_indices is not None and np.any(indices == 0):
            self._reset_env0_video(obs)

        if self.full_rollout_video:
            self._write_full_rollout_frame(obs)

    def on_step(
        self,
        next_obs: Observation,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: dict[str, Any],
    ) -> None:
        del reward, terminated, truncated, info
        if not self.should_write:
            return
        if self.args.record_video and not self.full_rollout_video and self.num_envs > 0:
            self._append_env0_video_frame(next_obs)
        if self.full_rollout_video:
            self._write_full_rollout_frame(next_obs)

    def on_episode_done(
        self,
        done_indices: np.ndarray,
        obs: Observation,
        info: dict[str, Any],
    ) -> None:
        if done_indices.size == 0:
            return

        episode_info = self._extract_episode_info(info, done_indices.size)
        returns = episode_info["r"]
        successes = episode_info["s"]

        for offset, env_idx in enumerate(done_indices.tolist()):
            self.completed_episodes += 1
            episode_return = float(returns[offset])
            episode_success = bool(successes[offset])
            self.total_episode_count += 1
            self.return_window.append(episode_return)
            self.success_window.append(float(episode_success))

            if (
                self.enabled
                and self.completed_episodes % self.effective_episode_freq == 0
            ):
                self._record_sample(
                    env_idx=env_idx,
                    done_obs=_select_obs(obs, [env_idx]),
                    done_info=info,
                    done_offset=offset,
                    episode_return=episode_return,
                    episode_success=episode_success,
                )

    def close(self) -> None:
        for writer in self.full_video_writers.values():
            writer.close()
        self.full_video_writers.clear()

        if not self.should_write:
            return

        write_json(
            self.summary_path,
            {
                "exp_name": self.exp_name,
                "process_id": self.proc_index,
                "total_processes": self.total_processes,
                "completed_episodes": self.completed_episodes,
                "total_episode_count": self.total_episode_count,
                "metric_window": self.metric_window,
                "mean_return": self._mean_return(),
                "mean_success_rate": self._mean_success_rate(),
            },
        )

    def _mean_return(self) -> float:
        if not self.return_window:
            return 0.0
        return float(np.mean(np.asarray(self.return_window, dtype=np.float32)))

    def _mean_success_rate(self) -> float:
        if not self.success_window:
            return 0.0
        return float(np.mean(np.asarray(self.success_window, dtype=np.float32)))

    def _extract_episode_info(
        self, info: dict[str, Any], expected_count: int
    ) -> dict[str, np.ndarray]:
        episode_info = info.get("episode", {}) if isinstance(info, dict) else {}
        returns = np.asarray(
            episode_info.get("r", np.zeros((expected_count,))), dtype=np.float32
        )
        successes = np.asarray(
            episode_info.get("s", np.zeros((expected_count,), dtype=np.bool_)),
            dtype=np.bool_,
        )
        if returns.ndim == 0:
            returns = returns.reshape(1)
        if successes.ndim == 0:
            successes = successes.reshape(1)
        if returns.shape[0] != expected_count:
            returns = np.resize(returns, (expected_count,))
        if successes.shape[0] != expected_count:
            successes = np.resize(successes, (expected_count,))
        return {"r": returns, "s": successes}

    def _record_sample(
        self,
        *,
        env_idx: int,
        done_obs: Observation,
        done_info: dict[str, Any],
        done_offset: int,
        episode_return: float,
        episode_success: bool,
    ) -> None:
        if not self.should_write:
            return

        episode_id = int(self.current_episode_ids[env_idx])
        sample_dir = (
            self.sampled_dir
            / f"ep_{self.completed_episodes:06d}"
            / f"env_{env_idx:03d}"
        )
        first_obs = self.first_obs_by_env[env_idx]
        if first_obs is None:
            first_obs = done_obs

        if self.args.record_obs_stats:
            first_dir = sample_dir / "obs" / "first"
            last_dir = sample_dir / "obs" / "last"
            self._write_observation_artifacts(first_dir, first_obs)
            self._write_observation_artifacts(
                last_dir,
                done_obs,
                info=self._select_episode_info(
                    done_info, env_idx=env_idx, done_offset=done_offset
                ),
            )
            append_jsonl(
                self.obs_stats_path,
                {
                    "completed_episode": self.completed_episodes,
                    "process_id": self.proc_index,
                    "env_id": env_idx,
                    "env_episode_id": episode_id,
                    "first": phase_stats(first_obs),
                    "last": phase_stats(done_obs),
                    "last_info": self._select_episode_info(
                        done_info, env_idx=env_idx, done_offset=done_offset
                    ),
                },
            )

        if self.args.record_episode_metrics:
            append_jsonl(
                self.episode_metrics_path,
                {
                    "completed_episode": self.completed_episodes,
                    "process_id": self.proc_index,
                    "env_id": env_idx,
                    "env_episode_id": episode_id,
                    "episode_return": episode_return,
                    "episode_success": episode_success,
                    "mean_return": self._mean_return(),
                    "mean_success_rate": self._mean_success_rate(),
                },
            )

        if self.args.record_video and not self.full_rollout_video and env_idx == 0:
            self._write_env0_episode_videos(sample_dir / "images")

    def _write_observation_artifacts(
        self,
        phase_dir: Path,
        obs: Observation,
        *,
        info: dict[str, Any] | None = None,
    ) -> None:
        for key, value in obs.images.items():
            image_path = phase_dir / "images" / f"{key}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(image_path, ensure_rgb(value[0]))

        for key, value in obs.states.items():
            state_path = phase_dir / "states" / f"{key}.npy"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with state_path.open("wb") as f:
                np.save(f, value[0], allow_pickle=False)

        text_path = phase_dir / "text.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("\n".join(obs.text.tolist()), encoding="utf-8")

        if info is not None:
            write_json(phase_dir / "info.json", to_builtin(info))

    def _select_episode_info(
        self,
        info: dict[str, Any],
        *,
        env_idx: int,
        done_offset: int,
    ) -> dict[str, Any]:
        def _slice(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _slice(v) for k, v in value.items()}
            if isinstance(value, np.ndarray):
                if value.ndim >= 1 and value.shape[0] == self.num_envs:
                    return value[env_idx]
                if value.ndim >= 1 and value.shape[0] > done_offset:
                    return value[done_offset]
                return value
            if isinstance(value, list):
                if len(value) == self.num_envs:
                    return value[env_idx]
                if len(value) > done_offset:
                    return value[done_offset]
                return value
            if isinstance(value, tuple):
                if len(value) == self.num_envs:
                    return value[env_idx]
                if len(value) > done_offset:
                    return value[done_offset]
                return value
            return value

        return {k: _slice(v) for k, v in info.items()}

    def _reset_env0_video(self, obs: Observation) -> None:
        self.env0_image_frames = {
            key: [ensure_rgb(value[0]).copy()] for key, value in obs.images.items()
        }

    def _append_env0_video_frame(self, obs: Observation) -> None:
        if not self.env0_image_frames:
            self._reset_env0_video(obs)
            return
        for key, value in obs.images.items():
            self.env0_image_frames.setdefault(key, []).append(
                ensure_rgb(value[0]).copy()
            )

    def _write_env0_episode_videos(self, images_dir: Path) -> None:
        for key, frames in self.env0_image_frames.items():
            if not frames:
                continue
            path = images_dir / f"{key}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(path, fps=8)
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()

    def _write_full_rollout_frame(self, obs: Observation) -> None:
        for key, value in obs.images.items():
            writer = self.full_video_writers.get(key)
            if writer is None:
                path = self.full_videos_dir / f"{key}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(path, fps=8)
                self.full_video_writers[key] = writer
            frames = np.asarray(value)
            grid = make_grid(np.asarray([ensure_rgb(frame) for frame in frames]))
            writer.append_data(grid)
