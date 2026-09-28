"""Gymnasium's classic-control environments as a PlugRL environment.

Both kinds of action space work. A discrete one (CartPole-v1) takes the
action as an integer; a continuous one (Pendulum-v1, MountainCarContinuous-v0)
as a float array of the space's shape and dtype. This env used to cast every
action to an integer, so the continuous tasks could not run at all.

Rendering is off by default, as in the MuJoCo family: a state-only policy
never reads the frames, and rendering every step costs more than the physics.
Turn it on with `--env.render` when something downstream wants pixels.
"""

import dataclasses
import importlib.util
import numpy as np
import gymnasium as gym

if importlib.util.find_spec("pygame") is None:
    raise ImportError(
        'pygame is not installed. Please install it with pip install "plugrl-env-client[classic]".'
    )

from plugrl_env_client.envs.base_env import BaseEnv, BaseEnvConfig, Action, Observation
from plugrl_env_client.utils.registration import register_env, register_env_config

UID = "Classic-v1"


@register_env_config(UID)
@dataclasses.dataclass
class ClassicConfig(BaseEnvConfig):
    name: str = "CartPole-v1"
    render: bool = False


@register_env(UID)
class ClassicEnv(BaseEnv):
    env: gym.Env

    def __init__(
        self,
        config: ClassicConfig,
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
            raise ValueError("ClassicEnv only supports num_envs=1")
        self.render_frames = bool(config.render)
        env = gym.make(config.name, render_mode="rgb_array" if self.render_frames else None)
        self.env = env
        self.game_name = config.name
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        # rollout() sizes its action plan from this before the first step, so
        # an env without it cannot run at all.
        self.single_action_space = env.action_space
        self.action_space = env.action_space

    def prepare_obs(self, obs: np.ndarray) -> Observation:
        frames = {}
        if self.render_frames:
            frames = {"env": np.array(self.env.render())[None, ...]}
        states = {"obs": obs[None, ...]}
        return Observation(
            images=frames,
            states=states,
            text=self.game_name,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Observation | None, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.prepare_obs(obs), info

    def step(
        self, action: Action
    ) -> tuple[Observation | None, np.ndarray, np.ndarray, np.ndarray, dict]:
        if self.discrete:
            action = int(np.asarray(action).item())
        else:
            space = self.env.action_space
            action = np.asarray(action, dtype=space.dtype).reshape(space.shape)
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = np.array([float(reward)], dtype=np.float32)
        terminated = np.array([bool(terminated)], dtype=np.bool_)
        truncated = np.array([bool(truncated)], dtype=np.bool_)
        return self.prepare_obs(obs), reward, terminated, truncated, info
