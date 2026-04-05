from __future__ import annotations

import dataclasses
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np

from plugrl_env_client.envs.base_env import Action, BaseEnv, BaseEnvConfig, Observation
from plugrl_env_client.envs.libero import image_tools
from plugrl_env_client.utils.registration import register_env, register_env_config

UID = "Robocasa-v1"

ROBOCASA_IMAGE_KEYS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
ROBOCASA_ACTION_DIM = 12
ROBOCASA_STATE_DIM = 16

_ACTION_KEY_ORDERING_HDF5 = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}

_ACTION_KEY_ORDERING_LEROBOT = {
    "base_motion": (0, 4),
    "control_mode": (4, 5),
    "end_effector_position": (5, 8),
    "end_effector_rotation": (8, 11),
    "gripper_close": (11, 12),
}


def _import_robocasa_dependencies() -> Callable[[np.ndarray], dict[str, np.ndarray]]:
    try:
        import robocasa  # noqa: F401
        import robocasa.wrappers.gym_wrapper  # noqa: F401
        from robocasa.utils.env_utils import convert_action
    except ImportError as exc:
        raise ImportError(
            "RoboCasa is not installed for plugrl-env-client. "
            "Please follow the manual RoboCasa installation steps documented in README.md "
            "for this repository before constructing Robocasa-v1."
        ) from exc
    return convert_action


def _make_robocasa_env(
    *,
    task_name: str,
    split: str,
    seed: int,
) -> gym.Env:
    return gym.make(f"robocasa/{task_name}", split=split, seed=seed)


def _prepare_image(img: np.ndarray, resize_size: int) -> np.ndarray:
    img = np.ascontiguousarray(img)
    img = image_tools.resize_with_pad(img, resize_size, resize_size)
    return image_tools.convert_to_uint8(img)


def _build_state(raw_obs: dict[str, Any]) -> np.ndarray:
    state = np.concatenate(
        (
            raw_obs["state.end_effector_position_relative"],
            raw_obs["state.end_effector_rotation_relative"],
            raw_obs["state.base_position"],
            raw_obs["state.base_rotation"],
            raw_obs["state.gripper_qpos"],
        ),
        axis=0,
    )
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != ROBOCASA_STATE_DIM:
        raise ValueError(
            f"Expected RoboCasa state dim {ROBOCASA_STATE_DIM}, got {state.shape[-1]}"
        )
    return state


def _get_prompt(raw_obs: dict[str, Any]) -> str:
    prompt = raw_obs.get("annotation.human.task_description", "")
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    return str(prompt)


def _prepare_single_observation(
    raw_obs: dict[str, Any], *, resize_size: int
) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    images = {
        key: _prepare_image(raw_obs[f"video.{key}"], resize_size)
        for key in ROBOCASA_IMAGE_KEYS
    }
    return images, _build_state(raw_obs), _get_prompt(raw_obs)


def _lerobot_to_hdf5_action(action_lerobot: np.ndarray) -> np.ndarray:
    reordered = np.zeros_like(action_lerobot, dtype=np.float32)
    for key, (hdf5_start, hdf5_end) in _ACTION_KEY_ORDERING_HDF5.items():
        lerobot_start, lerobot_end = _ACTION_KEY_ORDERING_LEROBOT[key]
        reordered[hdf5_start:hdf5_end] = action_lerobot[lerobot_start:lerobot_end]
    return reordered


def _collate_infos(info_by_env: list[dict[str, Any]]) -> dict[str, Any]:
    if not info_by_env:
        return {}
    keys = set().union(*(info.keys() for info in info_by_env))
    collated: dict[str, Any] = {}
    for key in keys:
        values = [info.get(key) for info in info_by_env]
        if all(isinstance(v, dict) for v in values):
            collated[key] = _collate_infos(values)
            continue
        if all(isinstance(v, np.ndarray) for v in values):
            try:
                collated[key] = np.stack(values)
                continue
            except ValueError:
                pass
        if all(
            isinstance(v, (bool, int, float, str, np.generic)) or v is None
            for v in values
        ):
            if any(v is None for v in values):
                collated[key] = list(values)
            else:
                collated[key] = np.asarray(values)
            continue
        collated[key] = list(values)
    return collated


def _extract_success_flag(info: dict[str, Any]) -> bool:
    success = info.get("success", False)
    if isinstance(success, np.ndarray):
        if success.size == 0:
            return False
        return bool(np.any(success))
    return bool(success)


@register_env_config(UID)
@dataclasses.dataclass
class RobocasaConfig(BaseEnvConfig):
    task_name: str = "SearingMeat"
    split: str = "target"
    seed: int = 7
    resize_size: int = 224
    action_encoding: Literal["passthrough", "lerobot_to_hdf5"] = "passthrough"


@register_env(UID, max_episode_steps=2000, best_reward_threshold_for_success=1.0)
class RobocasaEnv(BaseEnv):
    def __init__(
        self,
        config: RobocasaConfig,
        num_envs: int = 1,
        process_id: int | None = None,
        total_processes: int | None = None,
    ):
        super().__init__(
            config=config,
            num_envs=num_envs,
            process_id=process_id,
            total_processes=total_processes,
        )
        self.config = config
        self._convert_action = _import_robocasa_dependencies()
        process_offset = 0 if process_id is None else int(process_id) * self.num_envs
        self._envs: list[gym.Env] = []
        for env_idx in range(self.num_envs):
            self._envs.append(
                _make_robocasa_env(
                    task_name=config.task_name,
                    split=config.split,
                    seed=int(config.seed) + process_offset + env_idx,
                )
            )
        self._raw_obs_by_env: list[dict[str, Any] | None] = [None] * self.num_envs

        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ROBOCASA_ACTION_DIM,),
            dtype=np.float32,
        )
        self.action_space = self.single_action_space

    def _collate_observation(self) -> Observation:
        raw_obs_by_env = [obs for obs in self._raw_obs_by_env if obs is not None]
        if len(raw_obs_by_env) != self.num_envs:
            raise RuntimeError("RoboCasa observation cache is incomplete")

        image_batches = {key: [] for key in ROBOCASA_IMAGE_KEYS}
        states: list[np.ndarray] = []
        prompts: list[str] = []
        for raw_obs in raw_obs_by_env:
            images, state, prompt = _prepare_single_observation(
                raw_obs, resize_size=self.config.resize_size
            )
            for key, value in images.items():
                image_batches[key].append(value)
            states.append(state)
            prompts.append(prompt)

        return Observation(
            images={
                key: np.stack(values).astype(np.uint8, copy=False)
                for key, values in image_batches.items()
            },
            states={"state": np.stack(states).astype(np.float32, copy=False)},
            text=np.asarray(prompts, dtype=np.str_),
        )

    def _encode_action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.shape != (ROBOCASA_ACTION_DIM,):
            raise ValueError(
                f"Expected RoboCasa action shape ({ROBOCASA_ACTION_DIM},), got {action_arr.shape}"
            )
        if self.config.action_encoding == "lerobot_to_hdf5":
            action_arr = _lerobot_to_hdf5_action(action_arr)
        elif self.config.action_encoding != "passthrough":
            raise ValueError(
                f"Unsupported action_encoding: {self.config.action_encoding}"
            )
        return self._convert_action(action_arr)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, dict]:
        reset_indices = None if options is None else options.get("reset_indices")
        if reset_indices is None:
            indices = np.arange(self.num_envs, dtype=np.int64)
        else:
            indices = np.asarray(reset_indices, dtype=np.int64)

        for env_idx in indices.tolist():
            reset_seed = None
            if seed is not None:
                reset_seed = int(seed) + env_idx
            raw_obs, _ = self._envs[env_idx].reset(seed=reset_seed)
            self._raw_obs_by_env[env_idx] = raw_obs

        return self._collate_observation(), {}

    def step(
        self, actions: Action
    ) -> tuple[Observation, np.ndarray, np.ndarray, np.ndarray, dict]:
        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr[None, :]
        if actions_arr.shape != (self.num_envs, ROBOCASA_ACTION_DIM):
            raise ValueError(
                "Expected actions with shape "
                f"({self.num_envs}, {ROBOCASA_ACTION_DIM}), got {actions_arr.shape}"
            )

        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=np.bool_)
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)
        infos: list[dict[str, Any]] = []

        for env_idx, (env, action) in enumerate(zip(self._envs, actions_arr, strict=True)):
            raw_obs, reward, done, env_truncated, info = env.step(
                self._encode_action(action)
            )
            self._raw_obs_by_env[env_idx] = raw_obs
            rewards[env_idx] = float(reward)
            terminated[env_idx] = bool(done) or _extract_success_flag(info)
            truncated[env_idx] = bool(env_truncated)
            infos.append(info)

        return self._collate_observation(), rewards, terminated, truncated, _collate_infos(infos)

    def close(self) -> None:
        for env in getattr(self, "_envs", ()):
            env.close()
