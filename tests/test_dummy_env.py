import numpy as np

from plugrl_env_client.envs.dummy_env import DummyEnv, DummyEnvConfig
from plugrl_env_client.envs.base_env import Observation


def test_dummy_env_fake_obs_and_action_shapes():
    cfg = DummyEnvConfig()
    env = DummyEnv(cfg)

    action = env.fake_action()
    assert isinstance(action, np.ndarray)
    assert action.shape == (1, cfg.action_dim)

    obs = env.fake_obs()
    assert isinstance(obs, Observation)
    assert set(obs.images.keys()) == {"base", "wrist"}

    base = obs.images["base"]
    wrist = obs.images["wrist"]
    assert base.shape == (1, cfg.img_height, cfg.img_width, 3)
    assert wrist.shape == (1, cfg.img_height // 2, cfg.img_width // 2, 3)

    assert obs.text == cfg.text
