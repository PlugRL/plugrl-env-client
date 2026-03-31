# ruff: noqa: E402

import dataclasses
import os
from pathlib import Path
from collections.abc import Mapping
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")


def _configure_jax_cuda_toolchain() -> None:
    if sys.platform != "linux":
        return

    # Old MuJoCo / conda library paths can interfere with JAX CUDA detection.
    ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    if ld_library_path:
        entries = [p for p in ld_library_path.split(os.pathsep) if "mujoco210" not in p]
        if entries:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(entries)
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)

    for base in sys.path:
        if not base:
            continue
        base_path = Path(base)
        ptxas = base_path / "nvidia" / "cuda_nvcc" / "bin" / "ptxas"
        if not ptxas.is_file():
            continue

        ptxas_dir = str(ptxas.parent)
        path_entries = (
            os.environ.get("PATH", "").split(os.pathsep)
            if os.environ.get("PATH")
            else []
        )
        if ptxas_dir not in path_entries:
            os.environ["PATH"] = ptxas_dir + os.pathsep + os.environ.get("PATH", "")

        cuda_data_dir = str(base_path / "nvidia" / "cuda_nvcc")
        xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_gpu_cuda_data_dir=" not in xla_flags:
            os.environ["XLA_FLAGS"] = (
                xla_flags + f" --xla_gpu_cuda_data_dir={cuda_data_dir}"
            ).strip()
        return


_configure_jax_cuda_toolchain()

import jax
import numpy as np
import gymnasium as gym

from mujoco_playground import registry

from plugrl_env_client.envs.base_env import (
    Action,
    BaseEnv,
    BaseEnvConfig,
    BoolArray,
    Observation,
    RewardArray,
)
from plugrl_env_client.utils.registration import register_env, register_env_config

UID = "Playground-v1"


def _flatten_obs(obs: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(obs, Mapping):
        flat: dict[str, np.ndarray] = {}
        for key, value in obs.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            flat.update(_flatten_obs(value, prefix=next_prefix))
        return flat
    return {prefix or "obs": np.asarray(jax.device_get(obs))}


def _tree_take(tree: Any, index: int) -> Any:
    return jax.tree.map(lambda x: x[index], tree)


def _tree_assign_rows(base: Any, update: Any, indices: np.ndarray) -> Any:
    idx = np.asarray(indices, dtype=np.int64)
    return jax.tree.map(lambda a, b: a.at[idx].set(b), base, update)


@register_env_config(UID)
@dataclasses.dataclass
class PlaygroundConfig(BaseEnvConfig):
    name: str = "CheetahRun"
    seed: int = 0
    width: int = 320
    height: int = 240
    camera: str | None = None
    render_images: bool = False


@register_env(UID)
class PlaygroundEnv(BaseEnv):
    def __init__(
        self,
        config: PlaygroundConfig,
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

        env_config = registry.get_default_config(config.name)
        self.env = registry.load(config.name, config=env_config)
        self.rng = jax.random.PRNGKey(config.seed)
        self.state = None
        self.task_name = config.name
        self.action_dim = int(self.env.action_size)
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.action_space = self.single_action_space
        self._reset_fn = jax.jit(jax.vmap(self.env.reset))
        self._step_fn = jax.jit(jax.vmap(self.env.step))

    def _next_reset_keys(self, num_keys: int) -> jax.Array:
        self.rng, split_key = jax.random.split(self.rng)
        return jax.random.split(split_key, num_keys)

    def _render_frames(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError(
                "Environment state is not initialized. Call reset first."
            )

        frames = []
        for i in range(self.num_envs):
            frame = self.env.render(
                _tree_take(self.state, i),
                height=self.config.height,
                width=self.config.width,
                camera=self.config.camera,
            )
            frames.append(np.asarray(frame, dtype=np.uint8))
        return np.stack(frames, axis=0)

    def _prepare_obs(self) -> Observation:
        if self.state is None:
            raise RuntimeError(
                "Environment state is not initialized. Call reset first."
            )

        state_tensors = {
            key.replace("/", "_"): value
            for key, value in _flatten_obs(self.state.obs).items()
        }
        images = dict(env=self._render_frames()) if self.config.render_images else {}
        return Observation(
            images=images,
            states=state_tensors,
            text=self.task_name,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, dict]:
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)

        reset_indices = None if options is None else options.get("reset_indices")
        if reset_indices is None:
            reset_keys = self._next_reset_keys(self.num_envs)
            self.state = self._reset_fn(reset_keys)
        else:
            if self.state is None:
                raise RuntimeError(
                    "Partial reset requires an initialized environment state."
                )
            indices = np.asarray(reset_indices, dtype=np.int64)
            if indices.size == 0:
                return self._prepare_obs(), {}
            reset_keys = self._next_reset_keys(int(indices.size))
            reset_state = self._reset_fn(reset_keys)
            self.state = _tree_assign_rows(self.state, reset_state, indices)
        return self._prepare_obs(), {}

    def step(
        self, actions: Action
    ) -> tuple[Observation, RewardArray, BoolArray, BoolArray, dict]:
        if self.state is None:
            raise RuntimeError(
                "Environment state is not initialized. Call reset first."
            )

        action = np.asarray(actions, dtype=np.float32)
        if action.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected actions with leading batch {self.num_envs}, got {action.shape}"
            )
        self.state = self._step_fn(self.state, action)

        reward = np.asarray(jax.device_get(self.state.reward), dtype=np.float32)
        done = np.asarray(jax.device_get(self.state.done), dtype=np.bool_)
        metrics = {
            key: np.asarray(jax.device_get(value))
            for key, value in self.state.metrics.items()
        }
        return (
            self._prepare_obs(),
            reward,
            done,
            np.zeros((self.num_envs,), dtype=np.bool_),
            {"metrics": metrics},
        )


if __name__ == "__main__":
    from plugrl_env_client.cli import main

    main()
