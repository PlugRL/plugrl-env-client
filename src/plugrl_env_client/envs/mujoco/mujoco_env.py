"""Gymnasium's MuJoCo control suite as a PlugRL environment.

This is the smallest environment in this package that can show a policy
actually learning. It is continuous-control with a dense reward, it needs no
downloadable assets, no display and no GPU, and it runs on Windows - which
between them rule out every other family here for the purpose of producing a
first learning curve.

`HalfCheetah-v5` in particular has a 17-dimensional observation and a
6-dimensional action, which are exactly `FPOPolicyConfig`'s defaults
(`obs_dim=17`, `action_dim=6`). That is a coincidence worth relying on: no
config surgery is needed to point the shipped flow policy at it.

Rendering is **off by default**, and that is the difference between a
feasible CPU run and an infeasible one. A state-only policy never looks at
the frames, and rendering every step to produce them costs more than the
physics does. `SPEC.md` section 5.2 allows an observation with no images at
all, so the frames simply are not sent. Turn them on with `--env.render`
when something downstream actually wants pixels.
"""

import dataclasses
import importlib.util

import gymnasium as gym
import numpy as np

if importlib.util.find_spec("mujoco") is None:
    raise ImportError(
        "MuJoCo is not installed. Please install it with "
        'pip install "plugrl-env-client[mujoco]".'
    )

from plugrl_env_client.envs.base_env import BaseEnv, BaseEnvConfig, Action, Observation
from plugrl_env_client.utils.registration import register_env, register_env_config

UID = "MuJoCo-v1"


@register_env_config(UID)
@dataclasses.dataclass
class MuJoCoConfig(BaseEnvConfig):
    name: str = "HalfCheetah-v5"
    # Off by default; see the module docstring. Rendering needs a working
    # OpenGL context, which is one more thing to go wrong on a headless box.
    render: bool = False


@register_env(UID)
class MuJoCoEnv(BaseEnv):
    env: gym.Env

    def __init__(
        self,
        config: MuJoCoConfig,
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
        if self.num_envs != 1:
            raise ValueError(
                "MuJoCoEnv only supports num_envs=1; run more client processes "
                "instead, which is what --num-procs is for."
            )

        self.render_frames = bool(config.render)
        self.env = gym.make(
            config.name, render_mode="rgb_array" if self.render_frames else None
        )
        self.task_name = config.name

        # rollout() sizes its action plan from this before the first step, so
        # an env without it cannot run at all.
        self.single_action_space = self.env.action_space
        self.action_space = self.env.action_space

    def prepare_obs(self, obs: np.ndarray) -> Observation:
        images = {}
        if self.render_frames:
            images = {"env": np.asarray(self.env.render())[None, ...]}
        return Observation(
            images=images,
            # MuJoCo hands back float64; the policy consumes float32, and
            # halving the payload costs nothing that survives the network.
            states={"obs": np.asarray(obs, dtype=np.float32)[None, ...]},
            text=self.task_name,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation, dict]:
        self.seed_rngs(seed)
        # `reset_indices` is PlugRL's, not Gymnasium's - rollout() sends it to
        # say which sub-environments to restart, and with num_envs=1 there is
        # only ever one. Forwarding it would reach the MuJoCo env as an
        # unrecognised reset option.
        if options is not None:
            options = {k: v for k, v in options.items() if k != "reset_indices"} or None
        obs, info = self.env.reset(seed=seed, options=options)
        return self.prepare_obs(obs), info

    def step(
        self, action: Action
    ) -> tuple[Observation, np.ndarray, np.ndarray, np.ndarray, dict]:
        obs, reward, terminated, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32).reshape(-1)
        )
        return (
            self.prepare_obs(obs),
            np.asarray([float(reward)], dtype=np.float32),
            np.asarray([bool(terminated)], dtype=np.bool_),
            np.asarray([bool(truncated)], dtype=np.bool_),
            info,
        )
