from typing import Type, Dict
from copy import deepcopy
from functools import partial
import json
import difflib

import gymnasium as gym
from loguru import logger

from plugrl_worker.envs.base_env import BaseEnv, BaseEnvConfig
from plugrl_worker.utils.wrappers.success_record_wrapper import RecordSuccessByStep


class EnvSpec:
    def __init__(
        self,
        uid: str,
        cls: Type[BaseEnv],
        max_episode_steps: int | None = None,
        default_kwargs: dict | None = None,
    ):
        self.uid = uid
        self.cls = cls
        self.max_episode_steps = max_episode_steps
        self.default_kwargs = default_kwargs if default_kwargs is not None else {}

    def make(self, **kwargs):
        _kwargs = self.default_kwargs.copy()
        _kwargs.update(kwargs)
        return self.cls(**_kwargs)


REGISTERED_ENVS: Dict[str, EnvSpec] = {}


def register(
    name: str,
    cls: Type[BaseEnv],
    max_episode_steps: int | None = None,
    default_kwargs: dict | None = None,
):
    if name in REGISTERED_ENVS:
        logger.warning(f"Env {name} already registered")
    if not issubclass(cls, BaseEnv):
        raise TypeError(f"Env {name} must inherit from BaseEnv")

    REGISTERED_ENVS[name] = EnvSpec(
        name,
        cls,
        max_episode_steps=max_episode_steps,
        default_kwargs=default_kwargs,
    )


def make(env_id, **kwargs):
    if env_id not in REGISTERED_ENVS:
        raise KeyError("Env {} not found in registry".format(env_id))
    env_spec = REGISTERED_ENVS[env_id]
    env = env_spec.make(**kwargs)
    return env


def register_env(
    uid: str,
    max_episode_steps: int | None = None,
    best_reward_threshold_for_success: float | None = None,
    override: bool = False,
    **kwargs,
):
    """A decorator to register ManiSkill environments.

    Args:
        uid (str): unique id of the environment.
        max_episode_steps (int): maximum number of steps in an episode.
        asset_download_ids (List[str]): asset download ids the environment depends on. When environments are created
            this list is checked to see if the user has all assets downloaded and if not, prompt the user if they wish to download them.
        override (bool): whether to override the environment if it is already registered.

    Notes:
        - `max_episode_steps` is processed differently from other keyword arguments in gym.
          `gym.make` wraps the env with `gym.wrappers.TimeLimit` to limit the maximum number of steps.
        - `gym.EnvSpec` uses kwargs instead of **kwargs!
    """
    try:
        json.dumps(kwargs)
    except TypeError:
        raise RuntimeError(
            "You cannot register_env with non json dumpable kwargs, e.g. classes or types. If you really need to do this, it is recommended to create a mapping of string to the unjsonable data and to pass the string in the kwarg and during env creation find the data you need"
        )

    def _register_env(cls):
        if uid in REGISTERED_ENVS:
            if override:
                from gymnasium.envs.registration import registry

                logger.warning(f"Override registered env {uid}")
                REGISTERED_ENVS.pop(uid)
                registry.pop(uid)
            else:
                logger.warning(f"Env {uid} is already registered. Skip registration.")
                return cls

        register(
            uid,
            cls,
            max_episode_steps=max_episode_steps,
            default_kwargs=deepcopy(kwargs),
        )

        # Register for gym
        gym.register(
            uid,
            entry_point=partial(make, env_id=uid),
            max_episode_steps=max_episode_steps,
            disable_env_checker=True,  # Temporary solution as we allow empty observation spaces
            kwargs=deepcopy(kwargs),
            additional_wrappers=(
                gym.wrappers.RecordEpisodeStatistics.wrapper_spec(),
                RecordSuccessByStep.wrapper_spec(
                    best_reward_threshold_for_success=best_reward_threshold_for_success
                ),
            ),
        )

        return cls

    return _register_env


REGISTERED_ENV_CONFIGS: Dict[str, BaseEnvConfig] = {}


def register_env_config(uid: str):
    def _register_env_config(cls):
        if uid in REGISTERED_ENV_CONFIGS:
            raise KeyError(f"Env config {uid} is already registered.")
        REGISTERED_ENV_CONFIGS[uid] = cls()
        return cls

    return _register_env_config


def get_env_config(uid: str) -> BaseEnvConfig:
    if uid not in REGISTERED_ENV_CONFIGS:
        close_matches = difflib.get_close_matches(uid, REGISTERED_ENV_CONFIGS.keys())
        msg = f"Env config {uid} not found in registry."
        if close_matches:
            msg += f" Did you mean {close_matches}?"
        raise KeyError(msg)
    return REGISTERED_ENV_CONFIGS[uid]
