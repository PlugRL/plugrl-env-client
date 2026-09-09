import multiprocessing as mp
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Callable, cast

import gymnasium as gym

from plugrl_env_client.agent.base_agent import BaseAgent
from plugrl_env_client.envs.base_env import BaseEnvConfig
from plugrl_env_client.recorder import Recorder, RecorderArgs
from plugrl_env_client.runner.args import RunnerArgs
from plugrl_env_client.runner.rollout import rollout

AgentFactory = Callable[[], BaseAgent]


def derive_process_seed(
    seed: int | None, *, process_id: int | None, num_envs: int
) -> int | None:
    """Give each client process its own slice of the seed space.

    A vector env seeds its sub-envs from seed, seed+1, ... so processes have to
    be at least num_envs apart. Handing every process the same base seed would
    have them all replay the same trajectory, which looks like data collection
    and is not.
    """
    if seed is None:
        return None
    if process_id is None:
        return int(seed)
    return int(seed) + int(process_id) * max(1, int(num_envs))


def _make_env(
    args: RunnerArgs,
    env_config: BaseEnvConfig,
    num_envs: int,
    *,
    process_id: int | None,
    total_processes: int | None,
    env_lock,
) -> gym.vector.VectorEnv:
    def _create() -> gym.vector.VectorEnv:
        return gym.make_vec(
            args.uid,
            num_envs=num_envs,
            vectorization_mode="vector_entry_point",
            config=env_config,
            max_episode_steps=args.max_episode_steps,
            process_id=process_id,
            total_processes=total_processes,
        )

    if args.use_env_lock and env_lock is not None:
        with env_lock:
            return _create()
    return _create()


def run(
    args: RunnerArgs,
    agent_factory: AgentFactory,
    *,
    env_config: BaseEnvConfig,
    num_envs: int,
    num_episodes: int,
    exp_name: str,
    output_dir: Path,
    recorder_args: RecorderArgs,
    process_id: int | None = None,
    total_processes: int | None = None,
    env_lock=None,
) -> None:
    env_process_id = process_id if args.pass_proc_id else None
    env_total_processes = total_processes if args.pass_proc_id else None

    if env_lock is None and args.use_env_lock:
        env_lock = mp.Lock()

    env = _make_env(
        args,
        env_config,
        num_envs,
        process_id=env_process_id,
        total_processes=env_total_processes,
        env_lock=env_lock,
    )
    agent = agent_factory()
    recorder = Recorder(
        recorder_args,
        exp_name=exp_name,
        output_dir=output_dir,
        num_envs=num_envs,
        process_id=process_id,
        total_processes=total_processes,
    )
    seed = derive_process_seed(args.seed, process_id=process_id, num_envs=num_envs)
    try:
        rollout(
            env,
            agent,
            num_episodes=num_episodes,
            replan_steps=args.replan_steps,
            num_envs=num_envs,
            recorder=recorder,
            seed=seed,
        )
    finally:
        recorder.close()
        env.close()


def _run_process_entry(
    args: RunnerArgs,
    agent_factory: AgentFactory,
    env_config: BaseEnvConfig,
    num_envs: int,
    num_episodes: int,
    exp_name: str,
    output_dir: Path,
    recorder_args: RecorderArgs,
    process_id: int,
    total_processes: int,
    env_lock,
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
        process_id=process_id,
        total_processes=total_processes,
        env_lock=env_lock,
    )


def run_multiprocess(
    args: RunnerArgs,
    agent_factory: AgentFactory,
    *,
    env_config: BaseEnvConfig,
    num_envs: int,
    num_episodes: int,
    num_procs: int,
    exp_name: str,
    output_dir: Path,
    recorder_args: RecorderArgs,
) -> None:
    num_procs = int(num_procs)
    if num_procs <= 1:
        run(
            args,
            agent_factory,
            env_config=env_config,
            num_envs=num_envs,
            num_episodes=num_episodes,
            exp_name=exp_name,
            output_dir=output_dir,
            recorder_args=recorder_args,
        )
        return

    ctx = mp.get_context(args.start_method)
    env_lock = ctx.Lock() if args.use_env_lock else None
    processes: list[BaseProcess] = []
    for process_id in range(num_procs):
        p = cast(Any, ctx).Process(
            target=_run_process_entry,
            args=(
                args,
                agent_factory,
                env_config,
                num_envs,
                num_episodes,
                exp_name,
                output_dir,
                recorder_args,
                process_id,
                num_procs,
                env_lock,
            ),
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
