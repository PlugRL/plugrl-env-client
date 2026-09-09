import dataclasses
from typing import Literal

import tyro


@dataclasses.dataclass
class RunnerArgs:
    """Execution settings for the env client runner."""

    uid: tyro.conf.Suppress[str]

    # Whether to forward process id metadata into env construction.
    pass_proc_id: bool = False
    # Serialize env creation when the underlying backend is not fork-safe.
    use_env_lock: bool = False

    # Multiprocessing start method used by the client.
    start_method: Literal["spawn", "fork", "forkserver"] = "spawn"

    # If set, request a fresh action chunk every fixed number of steps.
    replan_steps: int | None = None
    # Optional max episode horizon override passed into env creation.
    max_episode_steps: int | None = None

    # Base seed for environment randomness. Left unset, episodes differ between
    # runs and results cannot be reproduced. With multiple client processes each
    # one takes a disjoint slice of the seed space; see derive_process_seed.
    seed: int | None = None
