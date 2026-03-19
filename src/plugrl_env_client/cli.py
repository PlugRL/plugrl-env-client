import dataclasses
import sys
import collections
from typing import Literal, Any, cast
import gymnasium as gym
import numpy as np
import tyro
from loguru import logger
import multiprocessing
import gc
import datetime
import dateutil
import uuid
import pathlib

import plugrl_env_client.envs
import functools
from plugrl_env_client.utils.registration import REGISTERED_ENV_CONFIGS
from plugrl_env_client.envs.base_env import BaseEnvConfig
from plugrl_env_client.utils.trajectory import Recorder
from plugrl_env_client.websocket_env_client_agent import (
    ServerStopped,
    WebSocketEnvClientAgent,
)
import plugrl_env_client.utils.wrappers as _wrappers

_env_lock = multiprocessing.Lock()


@dataclasses.dataclass
class RecordArgs:
    traj: bool = True
    traj_interval: int = 10
    video: bool = True
    video_interval: int = 100


@dataclasses.dataclass
class Args:
    record: RecordArgs
    uid: tyro.conf._markers.Suppress[str]
    env: BaseEnvConfig
    num_episodes: int = 1
    log_level: Literal["debug", "info"] = "info"

    num_workers: int = 1

    server_host: str = "0.0.0.0"
    server_port: int = 8000
    reconnect_on_server_stop: bool = False

    use_remote_viewer: bool = False

    viewer_host: str = "0.0.0.0"
    viewer_port: int = 9000

    prefix: str | None = None
    suffix: str | None = None
    exp_name: str | None = None
    log_base_dir: pathlib.Path = pathlib.Path("./logs")

    use_real_time: bool = False
    fps: float = 30.0

    replan_steps: int | None = None
    max_episode_steps: int | None = None

    pass_worker_id: bool = False
    use_env_lock: bool = False

    def __post_init__(self):
        if self.exp_name is None:
            if self.prefix is None:
                self.prefix = str(uuid.uuid4().fields[-1])[:5]

            self.exp_name = create_exp_name(self.prefix)
            if self.suffix is not None:
                self.exp_name = f"{self.exp_name}_{self.suffix}"

    @property
    def log_dir(self):
        assert self.exp_name is not None
        return self.log_base_dir / self.exp_name


def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return "%s_%s_%04d--s-%d" % (timestamp, exp_prefix, exp_id, seed)


_CONFIGS_DICT = {
    k.lower(): functools.partial(Args, uid=k, env=v)
    for k, v in REGISTERED_ENV_CONFIGS.items()
}


def cli() -> Args:
    return tyro.extras.subcommand_cli_from_dict(
        {k: v for k, v in _CONFIGS_DICT.items()}
    )


def _run_worker(worker_id: int, args: Args):
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "level": args.log_level.upper(),
                "format": (
                    "<green>{time:HH:mm:ss}</green>|"
                    "<level>{level}</level>|"
                    "{file}:{line}|"
                    "<yellow>{extra[prefix]}</yellow>"
                    "<level>{message}</level>"
                ),
            }
        ],
        extra={"prefix": f"[W{worker_id}] "},
    )
    logger.info(f"Starting worker process for env: {args.uid}")

    try:
        if args.use_env_lock:
            with _env_lock:
                if args.max_episode_steps is None:
                    env = gym.make_vec(
                        args.uid,
                        num_envs=args.env.num_envs,
                        config=args.env,
                        worker_id=worker_id if args.pass_worker_id else None,
                        total_workers=args.num_workers if args.pass_worker_id else None,
                    )
                else:
                    env = gym.make_vec(
                        args.uid,
                        num_envs=args.env.num_envs,
                        config=args.env,
                        max_episode_steps=args.max_episode_steps,
                        worker_id=worker_id if args.pass_worker_id else None,
                        total_workers=args.num_workers if args.pass_worker_id else None,
                    )
        else:
            if args.max_episode_steps is None:
                env = gym.make_vec(
                    args.uid,
                    num_envs=args.env.num_envs,
                    config=args.env,
                    worker_id=worker_id if args.pass_worker_id else None,
                    total_workers=args.num_workers if args.pass_worker_id else None,
                )
            else:
                env = gym.make_vec(
                    args.uid,
                    num_envs=args.env.num_envs,
                    config=args.env,
                    max_episode_steps=args.max_episode_steps,
                    worker_id=worker_id if args.pass_worker_id else None,
                    total_workers=args.num_workers if args.pass_worker_id else None,
                )
    except Exception as e:
        logger.error(f"Failed to create environment {args.uid}: {e}")
        gc.collect()
        return

    try:
        env_client_agent = WebSocketEnvClientAgent(
            host=args.server_host,
            port=args.server_port,
            reconnect_on_server_stop=args.reconnect_on_server_stop,
        )
        logger.info(
            f"Connected to server with metadata: {env_client_agent.get_server_metadata()}"
        )
    except ServerStopped as e:
        logger.info(f"Server stopped normally before worker started running: {e}")
        return
    except Exception as e:
        logger.error(f"Failed to connect to server: {e}")
        return

    if args.use_remote_viewer:
        if worker_id == 0:
            logger.info(
                f"Remote viewer enabled at {args.viewer_host}:{args.viewer_port}. Other workers share this port."
            )
            env = _wrappers.RemoteViewerWrapper(
                cast(Any, env),
                websocket_uri=f"ws://{args.viewer_host}:{args.viewer_port}/ws/env",
            )

    if args.use_real_time:
        env = _wrappers.RealTimeWrapper(cast(Any, env), fps=args.fps)
        logger.info(f"Real-time mode enabled at {args.fps} FPS")

    recorder = Recorder(
        save_dir=args.log_dir / f"worker_{worker_id}",
        record_trajectory=args.record.traj,
        trajectory_save_interval=args.record.traj_interval,
        record_video=args.record.video,
        video_record_interval=args.record.video_interval,
    )

    try:
        for ep in range(args.num_episodes):
            obs, info = env.reset()
            recorder.record_frame(ep, obs)
            action_plan = collections.deque()

            num_envs = int(getattr(env, "num_envs", 1))
            reward = np.zeros((num_envs,), dtype=np.float32)
            terminated = np.zeros((num_envs,), dtype=np.bool_)
            truncated = np.zeros((num_envs,), dtype=np.bool_)
            sum_reward = np.zeros((num_envs,), dtype=np.float32)
            step_count, total_reward = 0, 0.0

            while not (bool(np.any(terminated)) or bool(np.any(truncated))):
                if not action_plan:
                    sum_reward[...] = 0.0
                    obs_msg = dataclasses.asdict(obs)
                    action_data = env_client_agent.infer(obs_msg)
                    action_chunk = action_data["action"]
                    replan_steps = args.replan_steps or len(action_chunk)
                    assert len(action_chunk) >= replan_steps, (
                        f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    )
                    action_plan.extend(action_chunk)

                action = action_plan.popleft()
                obs, reward, terminated, truncated, info = env.step(action)
                reward = np.asarray(reward, dtype=np.float32)
                terminated = np.asarray(terminated, dtype=np.bool_)
                truncated = np.asarray(truncated, dtype=np.bool_)
                recorder.record_frame(ep, obs)
                step_count += 1
                total_reward += float(np.sum(reward))
                sum_reward += reward

                if (
                    not action_plan
                    or bool(np.any(terminated))
                    or bool(np.any(truncated))
                ):
                    obs_msg = dataclasses.asdict(obs)
                    env_client_agent.feedback(
                        obs_msg,
                        sum_reward,
                        terminated,
                        truncated,
                        info,
                    )

            success = False
            if "episode" in info and isinstance(info["episode"], dict):
                success = info["episode"].get("s", False)
            recorder.finish_episode(
                ep, total_reward, success, step_count, force=ep == args.num_episodes - 1
            )
            logger.debug(
                f"Episode {ep} finished after {step_count} steps with total reward {total_reward} and info {info}"
            )
    except ServerStopped as exc:
        logger.info(f"Server stopped normally. Exiting worker loop: {exc}")
    finally:
        recorder.close()
        env.close()


def _main(args: Args):
    multiprocessing.set_start_method("spawn", force=True)
    logger.configure(handlers=[{"sink": sys.stdout, "level": args.log_level.upper()}])

    logger.info(f"plugrl_env_client version: {plugrl_env_client.__version__}")
    logger.info(f"Selected env: {args.uid}")
    logger.info(f"Env config: {args.env}")

    if args.num_workers < 1:
        logger.error("num_workers must be at least 1.")
        return

    logger.info(
        f"Starting {args.num_workers} worker processes to run {args.num_episodes} episodes each."
    )
    processes = []
    if args.num_workers == 1:
        _run_worker(0, args)
    else:
        for i in range(args.num_workers):
            proc = multiprocessing.Process(
                target=_run_worker, args=(i, args), daemon=False
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

    logger.info("All worker processes finished.")


def main():
    _main(cli())


if __name__ == "__main__":
    main()
