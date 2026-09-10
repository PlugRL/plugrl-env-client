import gymnasium as gym
import numpy as np

from plugrl_env_client.envs.base_env import Observation


def _select_obs(obs: Observation, env_indices: np.ndarray | list[int]) -> Observation:
    return Observation(
        images={k: v[env_indices] for k, v in obs.images.items()},
        states={k: v[env_indices] for k, v in obs.states.items()},
        text=obs.text[env_indices],
    )


def _select_info(
    info: dict, env_indices: np.ndarray | list[int], *, num_envs: int
) -> dict:
    def _slice(v):
        if isinstance(v, np.ndarray) and v.shape[:1] == (num_envs,):
            return v[env_indices]
        if isinstance(v, list) and len(v) == num_envs:
            return [v[i] for i in env_indices]
        if isinstance(v, tuple) and len(v) == num_envs:
            return tuple(v[i] for i in env_indices)
        if isinstance(v, dict):
            return {kk: _slice(vv) for kk, vv in v.items()}
        return v

    return {k: _slice(v) for k, v in info.items()}


def _get_action_spec(
    env: gym.vector.VectorEnv, *, num_envs: int
) -> tuple[tuple[int, ...], np.dtype]:
    """The shape and dtype of one environment's action.

    `single_action_space` is unbatched by Gymnasium's own definition, so its
    shape is taken as-is. Guessing at a batch dimension there - by asking
    whether the leading axis happens to equal `num_envs` - can only ever be
    wrong: it silently drops a real action dimension whenever an environment
    with a `num_envs`-dimensional action is run with that many environments,
    which for a 7-DoF arm means `--num-envs 7`.

    The `action_space` fallback is different. On a Gymnasium vector env that
    attribute *is* the batched space, so the heuristic stays there, for envs
    that expose no `single_action_space` at all.
    """
    try:
        return tuple(map(int, env.single_action_space.shape)), np.dtype(
            env.single_action_space.dtype
        )
    except AttributeError:
        pass

    try:
        space = env.action_space
    except AttributeError as exc:
        raise ValueError(
            "Env must expose action space via single_action_space/action_space"
        ) from exc

    shape = tuple(map(int, space.shape))
    return (shape[1:] if shape[:1] == (num_envs,) else shape), np.dtype(space.dtype)
