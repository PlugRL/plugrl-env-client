import imageio
import pandas as pd
import pathlib
from plugrl_worker.envs.base_env import Observation


class Recorder:
    def __init__(
        self,
        save_dir: pathlib.Path,
        record_trajectory: bool = True,
        record_video: bool = True,
        trajectory_save_interval: int = 10,
        video_record_interval: int = 10,
    ):
        self.save_dir = save_dir
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.record_trajectory = record_trajectory
        self.record_video = record_video
        self.trajectory_save_interval = trajectory_save_interval
        self.video_record_interval = video_record_interval
        self.trajectory_df = pd.DataFrame(
            columns=["episode", "reward", "success", "length"]
        )
        self.writers = None

    def init_writer(self, episode: int, obs: Observation):
        if not self.record_video:
            return
        self.writers = {}
        for cam_name, img_array in obs.images.items():
            video_path = self.save_dir / f"episode_{episode:06d}_{cam_name}.mp4"
            self.writers[cam_name] = imageio.get_writer(video_path, fps=30)

    def record_frame(self, episode: int, obs: Observation, force: bool = False):
        if not self.record_video or (
            episode % self.video_record_interval != 0 and not force
        ):
            return
        if self.writers is None:
            self.init_writer(episode, obs)
        if self.writers is not None:
            for cam_name, img_array in obs.images.items():
                self.writers[cam_name].append_data(
                    img_array[0]
                )  # Assuming batch size of 1

    def finish_episode(
        self,
        episode: int,
        reward: float,
        success: bool,
        length: int,
        force: bool = False,
    ):
        if self.record_trajectory:
            self.trajectory_df.loc[len(self.trajectory_df)] = [
                episode,
                reward,
                success,
                length,
            ]
        if self.record_video and self.writers is not None:
            for writer in self.writers.values():
                writer.close()
            self.writers = None
        if episode % self.trajectory_save_interval == 0 or force:
            self.trajectory_df.to_csv(self.save_dir / "trajectory.csv", index=False)

    def close(self) -> None:
        if self.record_video and self.writers is not None:
            for writer in self.writers.values():
                writer.close()
            self.writers = None
        if self.record_trajectory:
            self.trajectory_df.to_csv(self.save_dir / "trajectory.csv", index=False)
