from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from plugrl_env_client.envs.base_env import Observation
from plugrl_env_client.recorder.args import RecorderArgs
from plugrl_env_client.recorder.events import EpisodeSampleEvent, FullRolloutFrameEvent
from plugrl_env_client.recorder.sink import AsyncRecorderSink
from plugrl_env_client.recorder.writers import (
    EpisodeMetricsWriter,
    EpisodeVideoBuffer,
    ObservationArtifactWriter,
    VideoArtifactWriter,
)
from plugrl_env_client.utils.recorder import write_json
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

        self._obs_writer = ObservationArtifactWriter(obs_stats_path=self.obs_stats_path)
        self._metrics_writer = EpisodeMetricsWriter(
            episode_metrics_path=self.episode_metrics_path
        )
        self._video_buffer = EpisodeVideoBuffer()
        self._video_writer = VideoArtifactWriter(
            full_videos_dir=self.full_videos_dir,
            video_fps=args.video_fps,
        )
        self._sink: AsyncRecorderSink | None = None

        if not self.should_write:
            return

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._sink = AsyncRecorderSink(
            name=f"recorder-writer-{self.proc_index}",
            handler=self._handle_event,
            on_close=self._video_writer.close,
        )
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
            self.first_obs_by_env[env_idx] = self._clone_observation(
                _select_obs(obs, [env_idx])
            )
            self.current_episode_ids[env_idx] += 1

        if not self.should_write:
            return

        if reset_indices is None and int(0) < self.num_envs:
            self._video_buffer.reset(obs)
        elif reset_indices is not None and np.any(indices == 0):
            self._video_buffer.reset(obs)

        if self.full_rollout_video:
            self._submit(FullRolloutFrameEvent(obs=self._snapshot_images(obs)))

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
            self._video_buffer.append(next_obs)
        if self.full_rollout_video:
            self._submit(FullRolloutFrameEvent(obs=self._snapshot_images(next_obs)))

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
        if not self.should_write:
            return

        if self._sink is not None:
            self._sink.close()

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
        episode_id = int(self.current_episode_ids[env_idx])
        sample_dir = (
            self.sampled_dir
            / f"ep_{self.completed_episodes:06d}"
            / f"env_{env_idx:03d}"
        )
        first_obs = self.first_obs_by_env[env_idx]
        if first_obs is None:
            first_obs = done_obs

        self._submit(
            EpisodeSampleEvent(
                sample_dir=sample_dir,
                completed_episode=self.completed_episodes,
                process_id=self.proc_index,
                env_id=env_idx,
                env_episode_id=episode_id,
                episode_return=episode_return,
                episode_success=episode_success,
                mean_return=self._mean_return(),
                mean_success_rate=self._mean_success_rate(),
                first_obs=self._clone_observation(first_obs),
                done_obs=self._clone_observation(done_obs),
                last_info=self._clone_value(
                    self._select_episode_info(
                        done_info, env_idx=env_idx, done_offset=done_offset
                    )
                ),
                env0_image_frames=self._video_buffer.snapshot()
                if self.args.record_video
                and not self.full_rollout_video
                and env_idx == 0
                else {},
            )
        )

    def _handle_event(self, event: EpisodeSampleEvent | FullRolloutFrameEvent) -> None:
        if isinstance(event, EpisodeSampleEvent):
            if self.args.record_obs_stats:
                self._obs_writer.write_sample(event)
            if self.args.record_episode_metrics:
                self._metrics_writer.write_sample(event)
            if event.env0_image_frames:
                self._video_writer.write_sample(event)
            return

        if isinstance(event, FullRolloutFrameEvent):
            self._video_writer.write_full_rollout_frame(event)
            return

        raise TypeError(f"Unsupported recorder event type: {type(event)!r}")

    def _submit(self, event: EpisodeSampleEvent | FullRolloutFrameEvent) -> None:
        if self._sink is None:
            raise RuntimeError("Recorder sink is not initialized")
        self._sink.submit(event)

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

    def _write_observation_artifacts(
        self,
        phase_dir: Path,
        obs: Observation,
        *,
        info: dict[str, Any] | None = None,
    ) -> None:
        self._obs_writer.write_observation_artifacts(phase_dir, obs, info=info)

    def _clone_observation(self, obs: Observation) -> Observation:
        return Observation(
            images={key: value.copy() for key, value in obs.images.items()},
            states={key: value.copy() for key, value in obs.states.items()},
            text=np.array(obs.text, dtype=np.str_, copy=True),
        )

    def _snapshot_images(self, obs: Observation) -> Observation:
        batch_size = 1
        if obs.images:
            batch_size = int(next(iter(obs.images.values())).shape[0])
        return Observation(
            images={key: value.copy() for key, value in obs.images.items()},
            states={},
            text=np.asarray([""] * batch_size, dtype=np.str_),
        )

    def _clone_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {k: self._clone_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clone_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._clone_value(v) for v in value)
        return value
