import dataclasses
import numpy as np
import pathlib
import math

try:
    from libero import benchmark
    from libero import get_libero_path
    from libero.envs import OffScreenRenderEnv
except ImportError:
    raise ImportError('libero is not installed. Please install it with pip install "plugrl-worker[libero]".')
except Exception as e:
    raise ImportError(f"An error occurred while importing libero: {e}")

from plugrl_worker.envs.base_env import BaseEnv, BaseEnvConfig, Action, Observation
from plugrl_worker.utils.registration import register_env, register_env_config

from plugrl_worker.envs.libero import image_tools
from loguru import logger

UID = "Libero-v1"

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

@register_env_config(UID)
@dataclasses.dataclass
class LiberoConfig(BaseEnvConfig):
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    task_id: int = 0
    seed: int = 7
    resize_size: int = 224
    randomize_initial_state: bool = True

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

@register_env(UID, best_reward_threshold_for_success=1.)
class LiberoEnv(BaseEnv):
    env: OffScreenRenderEnv
    
    def __init__(self, config: LiberoConfig, worker_id: int | None = None, total_workers: int | None = None):
        super().__init__(config=config)
        self.config = config
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[config.task_suite_name]()
        self.task_suite = task_suite
        self.num_tasks_in_suite = task_suite.get_num_tasks()
        self.worker_id = worker_id
        self.total_workers = total_workers
        if worker_id is not None:
            self.task_id = worker_id % self.num_tasks_in_suite
        else:
            self.task_id = config.task_id % self.num_tasks_in_suite
        task = task_suite.get_task(self.task_id)
        initial_states = task_suite.get_task_init_states(self.task_id)
        
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, config.seed)
        
        self.env = env
        self.task_description = task_description
        self.initial_states = initial_states

        if config.task_suite_name == "libero_spatial":
            max_steps = 220  # longest training demo has 193 steps
        elif config.task_suite_name == "libero_object":
            max_steps = 280  # longest training demo has 254 steps
        elif config.task_suite_name == "libero_goal":
            max_steps = 300  # longest training demo has 270 steps
        elif config.task_suite_name == "libero_10":
            max_steps = 520  # longest training demo has 505 steps
        elif config.task_suite_name == "libero_90":
            max_steps = 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task suite: {config.task_suite_name}")
        
        self.max_steps = max_steps
        self.total_episodes = 0
        self.resize_size = config.resize_size
        self.num_steps_wait = config.num_steps_wait
        self.current_step = 0

    def prepare_obs(self, obs: dict) -> Observation:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, self.resize_size, self.resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, self.resize_size, self.resize_size)
        )

        eef_state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
        frames = {
            "base": img[None, ...],
            "wrist": wrist_img[None, ...],
        }
        states = {
            "eef": eef_state[None, ...],
        }
        return Observation(
            images=frames,
            states=states,
            text=self.task_description,
        )
        
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Observation | None, dict]:
        if options is None:
            options = {}

        if "initial_state" not in options:
            if self.config.randomize_initial_state:
                intial_states_idx = np.random.randint(len(self.initial_states))
            else:
                if self.worker_id is not None and self.total_workers is not None:
                    worker_id = self.worker_id
                    total_workers = self.total_workers
                    assert total_workers % self.num_tasks_in_suite == 0, "Total workers must be multiple of number of tasks in suite."
                    num_workers_per_task = total_workers // self.num_tasks_in_suite
                    assert len(self.initial_states) % num_workers_per_task == 0, "Number of initial states must be multiple of number of workers per task."
                    states_per_worker = len(self.initial_states) // num_workers_per_task
                    intial_states_idx = (worker_id // self.num_tasks_in_suite) * states_per_worker + (self.total_episodes % states_per_worker)
                else:  
                    intial_states_idx = self.total_episodes % len(self.initial_states)
            initial_state = self.initial_states[intial_states_idx]
            options["initial_state"] = initial_state
            self.total_episodes += 1
        self.env.reset()
        obs = self.env.set_init_state(options["initial_state"])
        
        t = 0
        while t < self.num_steps_wait:
            obs, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION)
            t += 1
        self.current_step = 0
        return self.prepare_obs(obs), {}
    
    def step(self, action: Action) -> tuple[Observation | None, float, bool, bool, dict]:
        if action.ndim > 1:
            action = action[0]
        obs, reward, done, info = self.env.step(action.tolist())
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        return self.prepare_obs(obs), float(reward), done, truncated, info
    
    def __del__(self):
        if hasattr(self, "env"):
            self.env.close()