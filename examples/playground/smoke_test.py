import numpy as np

from playground_env import PlaygroundConfig, PlaygroundEnv
import os
import jax


def main() -> None:
    print("jax backend:", jax.default_backend())
    print("jax devices:", jax.devices())
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
    print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
    env = PlaygroundEnv(PlaygroundConfig(), num_envs=4)
    obs, _ = env.reset(seed=0)
    action = np.zeros((env.num_envs, env.action_dim), dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    print("task:", env.task_name)
    print("action_dim:", env.action_dim)
    print("reset image shapes:", {k: v.shape for k, v in obs.images.items()})
    print("reset state shapes:", {k: v.shape for k, v in obs.states.items()})
    print("step image shapes:", {k: v.shape for k, v in next_obs.images.items()})
    print("reward:", reward.tolist())
    print("terminated:", terminated.tolist())
    print("truncated:", truncated.tolist())
    print("metric_keys:", sorted(info["metrics"].keys()))


if __name__ == "__main__":
    main()
