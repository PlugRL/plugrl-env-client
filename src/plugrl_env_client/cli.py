import dataclasses
import functools

import tyro

import plugrl_env_client.envs  # noqa: F401  # register envs
from plugrl_env_client.cli_runner import run
from plugrl_env_client.envs.base_env import BaseEnvConfig
from plugrl_env_client.utils.registration import REGISTERED_ENV_CONFIGS


@dataclasses.dataclass
class Args:
    uid: tyro.conf._markers.Suppress[str]
    env: BaseEnvConfig

    num_episodes: int = 1

    server_host: str = "0.0.0.0"
    server_port: int = 8000
    reconnect_on_server_stop: bool = False

    replan_steps: int | None = None
    max_episode_steps: int | None = None


_CONFIGS_DICT = {
    k.lower(): functools.partial(Args, uid=k, env=v)
    for k, v in REGISTERED_ENV_CONFIGS.items()
}


def cli() -> Args:
    return tyro.extras.subcommand_cli_from_dict(
        {k: v for k, v in _CONFIGS_DICT.items()}
    )


def main() -> None:
    args = cli()
    run(args)


if __name__ == "__main__":
    main()
