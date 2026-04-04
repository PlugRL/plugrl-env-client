from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plugrl_env_client.envs.base_env import Observation


@dataclass(slots=True)
class EpisodeSampleEvent:
    sample_dir: Path
    completed_episode: int
    process_id: int
    env_id: int
    env_episode_id: int
    episode_return: float
    episode_success: bool
    mean_return: float
    mean_success_rate: float
    first_obs: Observation
    done_obs: Observation
    last_info: dict[str, Any]
    env0_image_frames: dict[str, list[np.ndarray]]


@dataclass(slots=True)
class FullRolloutFrameEvent:
    obs: Observation


@dataclass(slots=True)
class DebugPacketEvent:
    packet_dir: Path
    name: str
    payload: dict[str, Any]
