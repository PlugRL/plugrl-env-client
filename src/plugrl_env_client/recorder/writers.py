from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio
import numpy as np
from plugrl_protocol import msgpack_numpy

from plugrl_env_client.envs.base_env import Observation
from plugrl_env_client.recorder.events import (
    DebugPacketEvent,
    EpisodeSampleEvent,
    FullRolloutFrameEvent,
)
from plugrl_env_client.utils.recorder import (
    append_jsonl,
    ensure_rgb,
    make_grid,
    phase_stats,
    summarize_value,
    to_builtin,
    write_json,
)


class ObservationArtifactWriter:
    def __init__(self, *, obs_stats_path: Path) -> None:
        self.obs_stats_path = obs_stats_path

    def write_sample(self, event: EpisodeSampleEvent) -> None:
        first_dir = event.sample_dir / "obs" / "first"
        last_dir = event.sample_dir / "obs" / "last"
        self.write_observation_artifacts(first_dir, event.first_obs)
        self.write_observation_artifacts(last_dir, event.done_obs, info=event.last_info)
        append_jsonl(
            self.obs_stats_path,
            {
                "completed_episode": event.completed_episode,
                "process_id": event.process_id,
                "env_id": event.env_id,
                "env_episode_id": event.env_episode_id,
                "first": phase_stats(event.first_obs),
                "last": phase_stats(event.done_obs),
                "last_info": event.last_info,
            },
        )

    def write_observation_artifacts(
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


class EpisodeMetricsWriter:
    def __init__(self, *, episode_metrics_path: Path) -> None:
        self.episode_metrics_path = episode_metrics_path

    def write_sample(self, event: EpisodeSampleEvent) -> None:
        append_jsonl(
            self.episode_metrics_path,
            {
                "completed_episode": event.completed_episode,
                "process_id": event.process_id,
                "env_id": event.env_id,
                "env_episode_id": event.env_episode_id,
                "episode_return": event.episode_return,
                "episode_success": event.episode_success,
                "mean_return": event.mean_return,
                "mean_success_rate": event.mean_success_rate,
            },
        )


class EpisodeVideoBuffer:
    def __init__(self) -> None:
        self._image_frames: dict[str, list[np.ndarray]] = {}

    def reset(self, obs: Observation) -> None:
        self._image_frames = {
            key: [ensure_rgb(value[0]).copy()] for key, value in obs.images.items()
        }

    def append(self, obs: Observation) -> None:
        if not self._image_frames:
            self.reset(obs)
            return
        for key, value in obs.images.items():
            self._image_frames.setdefault(key, []).append(ensure_rgb(value[0]).copy())

    def snapshot(self) -> dict[str, list[np.ndarray]]:
        return {
            key: [frame.copy() for frame in frames]
            for key, frames in self._image_frames.items()
        }


class VideoArtifactWriter:
    def __init__(self, *, full_videos_dir: Path, video_fps: float) -> None:
        self.full_videos_dir = full_videos_dir
        self.video_fps = float(video_fps)
        if self.video_fps <= 0:
            raise ValueError("video_fps must be > 0")
        self._full_video_writers: dict[str, Any] = {}

    def write_sample(self, event: EpisodeSampleEvent) -> None:
        self.write_episode_videos(
            images_dir=event.sample_dir / "images",
            image_frames=event.env0_image_frames,
        )

    def write_episode_videos(
        self,
        *,
        images_dir: Path,
        image_frames: dict[str, list[np.ndarray]],
    ) -> None:
        for key, frames in image_frames.items():
            if not frames:
                continue
            path = images_dir / f"{key}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(path, fps=self.video_fps)
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()

    def write_full_rollout_frame(self, event: FullRolloutFrameEvent) -> None:
        for key, value in event.obs.images.items():
            writer = self._full_video_writers.get(key)
            if writer is None:
                path = self.full_videos_dir / f"{key}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(path, fps=self.video_fps)
                self._full_video_writers[key] = writer
            frames = np.asarray(value)
            grid = make_grid(np.asarray([ensure_rgb(frame) for frame in frames]))
            writer.append_data(grid)

    def close(self) -> None:
        for key, writer in list(self._full_video_writers.items()):
            writer.close()
        self._full_video_writers.clear()


class DebugPacketWriter:
    def write_packet(self, event: DebugPacketEvent) -> None:
        packet_dir = event.packet_dir / event.name
        packet_dir.mkdir(parents=True, exist_ok=True)
        write_json(packet_dir / "summary.json", {"payload": summarize_value(event.payload)})
        (packet_dir / "raw.msgpack").write_bytes(msgpack_numpy.packb(event.payload))
