import dataclasses
from typing import Literal

import tyro

from plugrl_env_client.envs.base_env import BaseEnvConfig


@dataclasses.dataclass
class Args:
    uid: tyro.conf._markers.Suppress[str]
    env: BaseEnvConfig

    num_procs: int = 1

    pass_worker_id: bool = False
    use_env_lock: bool = False

    start_method: Literal["spawn", "fork", "forkserver"] = "spawn"
    set_start_method: bool = True

    num_episodes: int = 1

    server_host: str = "0.0.0.0"
    server_port: int = 8000
    reconnect_on_server_stop: bool = False

    replan_steps: int | None = None
    max_episode_steps: int | None = None
