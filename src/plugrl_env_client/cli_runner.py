import multiprocessing as mp
from multiprocessing.process import BaseProcess
from typing import Any, cast

import gymnasium as gym

from plugrl_env_client.cli_args import Args
from plugrl_env_client.cli_rollout import rollout
from plugrl_env_client.websocket_env_client_agent import WebSocketEnvClientAgent


def _make_env(
    args: Args,
    *,
    process_id: int | None,
    total_processes: int | None,
    env_lock,
) -> gym.vector.VectorEnv:
    def _create() -> gym.vector.VectorEnv:
        return gym.make_vec(
            args.uid,
            num_envs=args.env.num_envs,
            config=args.env,
            max_episode_steps=args.max_episode_steps,
            process_id=process_id,
            total_processes=total_processes,
        )

    if args.use_env_lock and env_lock is not None:
        with env_lock:
            return _create()
    return _create()


def _connect_agent(args) -> WebSocketEnvClientAgent:
    return WebSocketEnvClientAgent(
        host=args.server_host,
        port=args.server_port,
        reconnect_on_server_stop=args.reconnect_on_server_stop,
    )


def run(
    args: Args,
    *,
    process_id: int | None = None,
    total_processes: int | None = None,
    env_lock=None,
) -> None:
    if not args.pass_worker_id:
        process_id = None
        total_processes = None

    if env_lock is None and args.use_env_lock:
        env_lock = mp.Lock()

    env = _make_env(
        args,
        process_id=process_id,
        total_processes=total_processes,
        env_lock=env_lock,
    )
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


def _run_process_entry(
    args: Args, process_id: int, total_processes: int, env_lock
) -> None:
    run(
        args,
        process_id=process_id,
        total_processes=total_processes,
        env_lock=env_lock,
    )


def run_multiprocess(args: Args) -> None:
    num_procs = int(args.num_procs)
    if num_procs <= 1:
        run(args)
        return

    start_method = args.start_method
    ctx = mp.get_context(start_method)
    env_lock = ctx.Lock() if args.use_env_lock else None
    processes: list[BaseProcess] = []
    for process_id in range(num_procs):
        p = cast(Any, ctx).Process(
            target=_run_process_entry,
            args=(args, process_id, num_procs, env_lock),
            daemon=False,
        )
        p.start()
        processes.append(cast(BaseProcess, p))

    try:
        for p in processes:
            p.join()
            if p.exitcode not in (0, None):
                for other in processes:
                    if other.is_alive():
                        other.terminate()
                for other in processes:
                    other.join()
                raise RuntimeError(
                    f"Env client process exited abnormally: pid={p.pid}, exitcode={p.exitcode}"
                )
    except KeyboardInterrupt:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()
        raise
