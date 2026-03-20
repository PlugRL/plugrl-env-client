import gymnasium as gym

from plugrl_env_client.cli_rollout import rollout
from plugrl_env_client.websocket_env_client_agent import WebSocketEnvClientAgent


def _make_env(args) -> gym.vector.VectorEnv:
    return gym.make_vec(
        args.uid,
        num_envs=args.env.num_envs,
        config=args.env,
        max_episode_steps=args.max_episode_steps,
    )


def _connect_agent(args) -> WebSocketEnvClientAgent:
    return WebSocketEnvClientAgent(
        host=args.server_host,
        port=args.server_port,
        reconnect_on_server_stop=args.reconnect_on_server_stop,
    )


def run(args) -> None:
    env = _make_env(args)
    agent = _connect_agent(args)
    try:
        rollout(
            env,
            agent,
            num_episodes=args.num_episodes,
            replan_steps=args.replan_steps,
            num_envs=args.env.num_envs,
        )
    finally:
        env.close()
