"""The probe environment of SPEC.md section 8.1, for `plugrl-conformance --probe`.

Env i counts the steps of its episode in `states["t"]`, keeps the action it
applied last in `states["a"]`, pays a reward of 1 per step, and ends its
episode, terminated, when t reaches 3 + 2 * (i % 3). It never truncates.

The conformance checker sends actions whose values encode where in the chunk
they are, so from each feedback it can tell what this client did with them:
whether it summed the chunk's reward, sent the terminal observation, and
applied the actions time-major and in order. It needs no extra:

    plugrl-conformance --probe --scenario all --action-dim 3 --client \
        "plugrl-run-env-client probe-v1 --server-host 127.0.0.1 --server-port 8000 \
         --num-envs 3 --num-episodes 1000000"

`--env.action-dim` must match the checker's `--action-dim`.
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import numpy as np

from plugrl_env_client.utils.registration import register_env, register_env_config

from .base_env import (
    Action,
    BaseEnv,
    BaseEnvConfig,
    BoolArray,
    Observation,
    RewardArray,
)

UID = "Probe-v1"


@register_env_config(UID)
@dataclasses.dataclass
class ProbeEnvConfig(BaseEnvConfig):
    action_dim: int = 3


# No time limit: the probe env never truncates, and its episodes are 3 to 7
# steps long.
@register_env(UID, max_episode_steps=None)
class ProbeEnv(BaseEnv):
    def __init__(
        self,
        config: ProbeEnvConfig,
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
        self.action_dim = config.action_dim
        # float64, so the checker's values arrive exactly whatever they are.
        self.single_action_space = gym.spaces.Box(
            -np.inf, np.inf, (self.action_dim,), np.float64
        )
        self.action_space = self.single_action_space
        # Env indices on the wire are the positions in this vector env.
        self.lengths = 3 + 2 * (np.arange(num_envs) % 3)
        self.t = np.zeros(num_envs, dtype=np.int64)
        self.a = np.zeros((num_envs, self.action_dim), dtype=np.float64)

    def _observation(self) -> Observation:
        return Observation(
            images={},
            states={
                "t": self.t[:, None].astype(np.float64),
                "a": self.a.copy(),
            },
            text="probe",
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        self.seed_rngs(seed)
        indices = np.arange(self.num_envs)
        if options is not None and options.get("reset_indices") is not None:
            indices = np.asarray(options["reset_indices"], dtype=np.int64)
        self.t[indices] = 0
        self.a[indices] = 0.0
        return self._observation(), {}

    def step(
        self, actions: Action
    ) -> tuple[Observation, RewardArray, BoolArray, BoolArray, dict]:
        actions = np.asarray(actions, dtype=np.float64).reshape(
            self.num_envs, self.action_dim
        )
        self.t += 1
        self.a = actions.copy()
        reward = np.ones(self.num_envs, dtype=np.float32)
        terminated = self.t == self.lengths
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        # No reset here: the client sends this terminal observation, then
        # resets the env itself (SPEC.md section 5.4).
        return self._observation(), reward, terminated, truncated, {}
