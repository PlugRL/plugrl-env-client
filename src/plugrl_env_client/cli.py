import dataclasses
import functools
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import tyro
from loguru import logger

import plugrl_env_client.envs  # noqa: F401  # register envs
from plugrl_env_client.agent.websocket_env_client_agent import WebSocketEnvClientAgent
from plugrl_env_client.envs.base_env import BaseEnvConfig
from plugrl_env_client.recorder import RecorderArgs
from plugrl_env_client.runner import RunnerArgs, run, run_multiprocess
from plugrl_env_client.utils.registration import REGISTERED_ENV_CONFIGS


@dataclasses.dataclass
class Args:
    uid: tyro.conf.Suppress[str]
    env: BaseEnvConfig
    num_envs: int = 1
    num_procs: int = 1
    num_episodes: int = 1
    runner: RunnerArgs = dataclasses.field(default_factory=lambda: RunnerArgs(uid=""))
    recorder: RecorderArgs = dataclasses.field(default_factory=RecorderArgs)
    exp_name: str | None = None
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    reconnect_on_server_stop: bool = False


_CONFIGS_DICT = {
    k.lower(): functools.partial(
        Args,
        uid=k,
        env=v,
        runner=RunnerArgs(uid=k),
        recorder=RecorderArgs(),
    )
    for k, v in REGISTERED_ENV_CONFIGS.items()
}


def cli() -> Args:
    return tyro.extras.subcommand_cli_from_dict(_CONFIGS_DICT)


def make_agent_factory(args: Args):
    return functools.partial(
        WebSocketEnvClientAgent,
        host=args.server_host,
        port=args.server_port,
        reconnect_on_server_stop=args.reconnect_on_server_stop,
    )


def _generate_exp_name(args: Args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{args.uid.lower()}-nenv{int(args.num_envs)}-{timestamp}-{uuid4().hex[:8]}"


def _save_client_config(args: Args, *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(args)
    config_path = output_dir / "client_config.json"
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_single_process_entry(
    args: RunnerArgs,
    agent_factory,
    env_config: BaseEnvConfig,
    num_envs: int,
    num_episodes: int,
    exp_name: str,
    output_dir: Path,
    recorder_args: RecorderArgs,
) -> None:
    run(
        args,
        agent_factory,
        env_config=env_config,
        num_envs=num_envs,
        num_episodes=num_episodes,
        exp_name=exp_name,
        output_dir=output_dir,
        recorder_args=recorder_args,
        process_id=0,
        total_processes=1,
    )


def main() -> None:
    args = cli()
    args.runner.uid = args.uid
    args.exp_name = args.exp_name or _generate_exp_name(args)
    output_dir = Path("runs") / args.exp_name
    _save_client_config(args, output_dir=output_dir)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger.add(output_dir / "logs" / "client.log", enqueue=True)
    logger.info(
        "Starting env client exp_name={} output_dir={} recorder={}",
        args.exp_name,
        output_dir,
        dataclasses.asdict(args.recorder),
    )
    agent_factory = make_agent_factory(args)

    mp.set_start_method(args.runner.start_method, force=True)

    if args.num_procs > 1:
        run_multiprocess(
            args.runner,
            agent_factory,
            env_config=args.env,
            num_envs=args.num_envs,
            num_episodes=args.num_episodes,
            num_procs=args.num_procs,
            exp_name=args.exp_name,
            output_dir=output_dir,
            recorder_args=args.recorder,
        )
    else:
        ctx = mp.get_context(args.runner.start_method)
        p = cast(
            mp.Process,
            ctx.Process(
                target=_run_single_process_entry,
                args=(
                    args.runner,
                    agent_factory,
                    args.env,
                    args.num_envs,
                    args.num_episodes,
                    args.exp_name,
                    output_dir,
                    args.recorder,
                ),
                daemon=False,
            ),
        )
        p.start()
        p.join()
        if p.exitcode not in (0, None):
            raise RuntimeError(
                f"Env client process exited abnormally: pid={p.pid}, exitcode={p.exitcode}"
            )


if __name__ == "__main__":
    main()
