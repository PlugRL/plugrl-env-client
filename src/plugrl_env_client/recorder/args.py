import dataclasses


@dataclasses.dataclass
class RecorderArgs:
    """Controls optional rollout recording artifacts and summaries."""

    # Record every N completed episodes for the current process. Set to 0 to disable.
    episode_freq: int = 0
    # Only allow process 0 to emit recorder outputs.
    thread0_only: bool = True
    # Save per-field video artifacts when a sampled episode is recorded.
    record_video: bool = False
    # Output fps for recorded mp4 artifacts.
    video_fps: float = 30.0
    # When enabled with episode_freq=1, record the whole rollout as videos.
    record_full_rollout: bool = False
    # Save first/last observation stats and raw per-field obs artifacts.
    record_obs_stats: bool = True
    # Save running return and success summaries for sampled episodes.
    record_episode_metrics: bool = True
    # Window size for moving-average return and success metrics.
    metric_window: int = 100
