import dataclasses
import uuid

import numpy as np
import pytest
import gymnasium as gym

from plugrl_env_client.envs.base_env import BaseEnv, BaseEnvConfig, Observation
from plugrl_env_client.utils.registration import (
    get_env_config,
    register_env,
    register_env_config,
)


def test_register_env_config_and_get_env_config_with_close_match():
    uid = f"UnitTestConfig-{uuid.uuid4()}"

    @register_env_config(uid)
    @dataclasses.dataclass
    class Cfg(BaseEnvConfig):
        foo: int = 123

    cfg = get_env_config(uid)
    assert isinstance(cfg, Cfg)
    assert cfg.foo == 123

    with pytest.raises(KeyError) as exc:
        get_env_config(uid.replace("Config", "Cnfig"))
    assert "Did you mean" in str(exc.value)


def test_register_env_config_duplicate_raises():
    uid = f"UnitTestConfigDup-{uuid.uuid4()}"

    @register_env_config(uid)
    @dataclasses.dataclass
    class Cfg1(BaseEnvConfig):
        pass

    with pytest.raises(KeyError):

        @register_env_config(uid)
        @dataclasses.dataclass
        class Cfg2(BaseEnvConfig):
            pass


def test_register_env_rejects_non_json_kwargs():
    uid = f"UnitTestEnvBadKwargs-{uuid.uuid4()}"

    with pytest.raises(RuntimeError):
        register_env(uid, not_json=object())


def test_register_env_and_gym_make_works():
    uid = f"UnitTestEnv-{uuid.uuid4()}"

    @register_env_config(uid)
    @dataclasses.dataclass
    class Cfg(BaseEnvConfig):
        pass

    @register_env(uid, max_episode_steps=3)
    class Env(BaseEnv):
        def __init__(
            self,
            config: Cfg,
            process_id: int | None = None,
            total_processes: int | None = None,
        ):
            super().__init__(
                config=config,
                process_id=process_id,
                total_processes=total_processes,
            )
            self._done = False
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict({})

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            self._done = False
            obs = Observation(images={}, states={}, text="")
            return obs, {}

        def step(self, actions):
            self._done = True
            obs = Observation(images={}, states={}, text="")
            reward = np.array([1.0], dtype=np.float32)
            terminated = np.array([True], dtype=np.bool_)
            truncated = np.array([False], dtype=np.bool_)
            return obs, reward, terminated, truncated, {}

    env = gym.make_vec(uid, num_envs=2, config=Cfg())
    obs, info = env.reset()
    assert isinstance(obs, Observation)

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0], dtype=np.float32)
    )
    assert bool(np.asarray(terminated)[0]) is True
    assert float(np.asarray(reward)[0]) == 1.0
    env.close()
