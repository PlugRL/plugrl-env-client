import numpy as np

from plugrl_env_client.envs.dummy_env import DummyEnv, DummyEnvConfig
from plugrl_env_client.envs.base_env import Observation


def test_dummy_env_fake_obs_and_action_shapes():
    cfg = DummyEnvConfig(num_envs=2)
    env = DummyEnv(cfg)

    action = env.fake_action()
    assert isinstance(action, np.ndarray)
    assert action.shape == (cfg.num_envs, cfg.action_dim)

    obs = env.fake_obs()
    assert isinstance(obs, Observation)
    assert set(obs.images.keys()) == {"base", "wrist"}

    base = obs.images["base"]
    wrist = obs.images["wrist"]
    assert base.shape == (cfg.num_envs, cfg.img_height, cfg.img_width, 3)
    assert wrist.shape == (cfg.num_envs, cfg.img_height // 2, cfg.img_width // 2, 3)

    assert isinstance(obs.text, np.ndarray)
    assert obs.text.shape == (cfg.num_envs,)
    assert obs.text.tolist() == [cfg.text for _ in range(cfg.num_envs)]
