import abc
import dataclasses

from typing import Dict, Annotated, TypeVar
import numpy.typing as npt

import numpy as np
import gymnasium as gym

DType = TypeVar("DType", bound=np.generic)

ImageArray = Annotated[npt.NDArray[np.uint8], ("b", "h", "w", "c")]
StateArray = Annotated[npt.NDArray[DType], ("b", "d")]


@dataclasses.dataclass
class Observation:
    images: Dict[str, ImageArray]
    states: Dict[str, StateArray]
    text: str


Action = Annotated[npt.NDArray[DType], ("b", "da")]


@dataclasses.dataclass
class BaseEnvConfig:
    max_episode_steps: int | None = None


class BaseEnv(gym.Env, abc.ABC):
    def __init__(
        self,
        config: BaseEnvConfig,
        worker_id: int | None = None,
        total_workers: int | None = None,
    ): ...

    @abc.abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation | None, dict]: ...

    @abc.abstractmethod
    def step(
        self, action: Action
    ) -> tuple[Observation | None, float, bool, bool, dict]: ...

    def fake_action(self) -> Action: ...

    def fake_obs(self) -> Observation: ...
