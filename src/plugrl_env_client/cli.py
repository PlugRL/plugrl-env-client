import functools
import multiprocessing as mp

import tyro

import plugrl_env_client.envs  # noqa: F401  # register envs
from plugrl_env_client.cli_runner import run, run_multiprocess
from plugrl_env_client.cli_args import Args
from plugrl_env_client.utils.registration import REGISTERED_ENV_CONFIGS


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

    if args.set_start_method:
        mp.set_start_method(args.start_method, force=True)

    if args.num_procs > 1:
        run_multiprocess(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
