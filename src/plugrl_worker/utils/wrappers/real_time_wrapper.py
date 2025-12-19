import time
from gymnasium.core import Env, Wrapper

class RealTimeWrapper(Wrapper):
    def __init__(self, env: Env, fps: float = 30.0):
        super().__init__(env)
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.last_step_time = None

    def step(self, action):
        current_time = time.time()
        if self.last_step_time is not None:
            elapsed = current_time - self.last_step_time
            sleep_time = self.frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_step_time = time.time()
        return obs, reward, terminated, truncated, info