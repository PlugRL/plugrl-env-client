import multiprocessing as mp
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Callable, cast

import gymnasium as gym
from loguru import logger

from plugrl_env_client.agent.base_agent import BaseAgent
from plugrl_env_client.agent.websocket_env_client_agent import ServerStopped
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


def report_server_metadata(agent: BaseAgent, *, replan_steps: int | None) -> None:
    """Say what the server said about itself, and flag one mismatch.

    The `metadata` message (SPEC.md section 5.1) is the only thing a client
    receives before it has to commit to a configuration, and until recently
    it was always empty. Now that it describes the policy, the one setting
    it can check is `replan_steps`: asking for more steps than the policy
    plans fails inside rollout, but only after a full inference round trip,
    and the message there does not mention the policy or the horizon.

    This warns rather than raises. The keys are descriptive and a client
    must not require them, so an absent or stale `action_horizon` has to
    stay survivable - the real check is still the one in rollout, against
    the chunk that actually arrives.
    """
    metadata = getattr(agent, "get_server_metadata", dict)() or {}
    if not metadata:
        logger.info("Server sent no metadata; action shape is whatever arrives.")
        return

    logger.info("Server metadata: {}", metadata)

    horizon = metadata.get("action_horizon")
    if replan_steps and isinstance(horizon, int) and replan_steps > horizon:
        logger.warning(
            "replan_steps={} exceeds the action_horizon={} that {} reports. "
            "The first inference will fail unless the server is lying about "
            "its horizon.",
            replan_steps,
            horizon,
            metadata.get("policy", "the server"),
        )


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
    report_server_metadata(agent, replan_steps=args.replan_steps)
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
    except ServerStopped:
        # The end of training, not a failure. The server closes with
        # `plugrl-server-stop` (SPEC.md section 7.1) once the algorithm has
        # taken every step it was asked for, and a client with
        # reconnect_on_server_stop off has nothing left to do.
        #
        # This used to escape as an unhandled exception, so the documented
        # happy path - run a server for N steps, point a client at it - ended
        # in a traceback and exit 1 on both sides of a successful run. Under
        # run_multiprocess the non-zero child exit also tore down its
        # siblings mid-episode.
        logger.info(
            "Server signalled the end of the run; collection stopped after "
            "{}/{} episodes.",
            recorder.total_episode_count,
            num_episodes,
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
