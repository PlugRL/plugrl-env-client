import abc
import dataclasses

from typing import Any, Dict, Annotated, TypeVar
import numpy.typing as npt

import numpy as np
from gymnasium.utils import seeding
from gymnasium.vector import VectorEnv, AutoresetMode

DType = TypeVar("DType", bound=np.generic)

ImageArray = Annotated[npt.NDArray[np.uint8], ("b", "h", "w", "c")]
StateArray = Annotated[npt.NDArray[DType], ("b", "d")]
TextArray = Annotated[npt.NDArray[np.str_], ("b",)]


@dataclasses.dataclass
class Observation:
    images: Dict[str, ImageArray]
    states: Dict[str, StateArray]
    text: TextArray

    def __init__(
        self,
        images: Dict[str, ImageArray],
        states: Dict[str, StateArray],
        text: str | list[str] | npt.NDArray[np.str_] | None = None,
    ):
        self.images = images
        self.states = states

        batch_sizes: list[int] = []
        for arr in images.values():
            batch_sizes.append(int(arr.shape[0]))
        for arr in states.values():
            batch_sizes.append(int(arr.shape[0]))

        inferred_batch: int | None = None
        if batch_sizes:
            inferred_batch = batch_sizes[0]
            if any(b != inferred_batch for b in batch_sizes):
                raise ValueError(
                    "Inconsistent batch size across images/states in Observation"
                )

        if text is None:
            b = inferred_batch if inferred_batch is not None else 1
            text_arr = np.asarray([""] * b, dtype=np.str_)
        elif isinstance(text, str):
            b = inferred_batch if inferred_batch is not None else 1
            text_arr = np.asarray([text] * b, dtype=np.str_)
        else:
            text_arr = np.asarray(text, dtype=np.str_)
            if text_arr.ndim == 0:
                b = inferred_batch if inferred_batch is not None else 1
                text_arr = np.asarray([str(text_arr.item())] * b, dtype=np.str_)
            elif text_arr.ndim != 1:
                raise ValueError(
                    "Observation.text must be a 1D array with shape (batch,)"
                )
            else:
                if inferred_batch is None:
                    inferred_batch = int(text_arr.shape[0])
                if text_arr.shape[0] == 1 and inferred_batch != 1:
                    text_arr = np.asarray(
                        [str(text_arr[0])] * inferred_batch, dtype=np.str_
                    )
                elif text_arr.shape[0] != inferred_batch:
                    raise ValueError(
                        "Observation.text batch dimension must match images/states"
                    )

        self.text = text_arr


Action = Annotated[npt.NDArray[DType], ("b", "da")]

RewardArray = Annotated[npt.NDArray[np.float32], ("b",)]
BoolArray = Annotated[npt.NDArray[np.bool_], ("b",)]


@dataclasses.dataclass
class BaseEnvConfig:
    max_episode_steps: int | None = None


class BaseEnv(VectorEnv[Observation, Action, np.ndarray], abc.ABC):
    metadata: dict[str, Any] = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(
        self,
        config: BaseEnvConfig,
        num_envs: int = 1,
        process_id: int | None = None,
        total_processes: int | None = None,
    ):
        self.config = config
        self.process_id = process_id
        self.total_processes = total_processes
        self.num_envs = int(num_envs)

    def seed_rngs(self, seed: int | None) -> None:
        """Seed this env's generator. Call first thing in reset().

        Draw randomness from `self.np_random` rather than the module-level
        `np.random`, or the seed will have no effect: `np.random` is global
        state that nothing here owns.
        """
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

    @abc.abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, dict]: ...

    @abc.abstractmethod
    def step(
        self, actions: Action
    ) -> tuple[Observation, RewardArray, BoolArray, BoolArray, dict]: ...

    # Not abstract, because most envs have no use for them - but the bodies
    # raise rather than return None. Returning None sends the failure to
    # whoever eventually indexes the result, which is a long way from the
    # class that did not implement it.
    def fake_action(self) -> Action:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fake_action(). Sample "
            "from self.action_space instead, or implement it."
        )

    def fake_obs(self) -> Observation:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fake_obs()."
        )
