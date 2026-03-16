import asyncio
import json
import threading
import websockets
import dataclasses
import numpy as np
from gymnasium.core import Env, Wrapper
from typing import Any, Dict
from loguru import logger
import queue
from PIL import Image
import io
import base64

from plugrl_env_client.envs.base_env import Observation


class RemoteViewerCommunicator:
    def __init__(self, websocket_uri: str):
        self.websocket_uri = websocket_uri
        self.data_queue = queue.Queue(maxsize=1)
        self._thread = None
        self._stop_event = threading.Event()

    def _numpy_to_base64_jpeg(self, np_array: np.ndarray) -> str:
        # If it's a grayscale image (2D array), convert to RGB first
        if np_array.ndim == 3 and np_array.shape[-1] == 1:
            img = Image.fromarray(np_array[..., 0], "L")
            img = img.convert("RGB")
        # If it's an RGB/RGBA image
        elif np_array.ndim == 3 and np_array.shape[-1] in [3, 4]:
            img = Image.fromarray(np_array)
        else:
            raise ValueError("Unsupported array shape for image conversion.")

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _serialize_observation(self, obs_obj: Observation) -> Dict[str, Any]:
        """Converts the custom Observation object to a JSON-friendly dictionary."""
        obs_dict = dataclasses.asdict(obs_obj)

        serialized_dict = {}
        for key, value in obs_dict.items():
            if isinstance(value, dict):
                nested_dict = {}
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, np.ndarray):
                        # For images, convert to base64 string
                        if key == "images":
                            nested_dict[nested_key] = self._numpy_to_base64_jpeg(
                                nested_value[0]
                            )
                        # For states and other arrays, convert to list
                        else:
                            nested_dict[nested_key] = nested_value.tolist()
                    else:
                        nested_dict[nested_key] = nested_value
                serialized_dict[key] = nested_dict
            else:
                serialized_dict[key] = value
        return serialized_dict

    def _run_websocket_loop(self):
        async def async_loop():
            while not self._stop_event.is_set():
                try:
                    async with websockets.connect(self.websocket_uri) as websocket:
                        logger.success("Communicator connected to BrokerServer.")
                        await self._handle_sending(websocket)
                except Exception as e:
                    logger.warning(
                        f"Communicator connection failed: {e}. Retrying in 3s..."
                    )
                    await asyncio.sleep(3)

        asyncio.run(async_loop())
        logger.info("Communicator thread has stopped.")

    async def _handle_sending(self, websocket):
        while not self._stop_event.is_set():
            try:
                obs = await asyncio.to_thread(self.data_queue.get)
                if self._stop_event.is_set():
                    break

                # Serialize the complete observation object
                serialized_obs = self._serialize_observation(obs)

                await websocket.send(
                    json.dumps(
                        {"type": "observation_update", "payload": serialized_obs}
                    )
                )
                self.data_queue.task_done()
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed while sending.")
                break

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_websocket_loop, daemon=True, name="communicator"
            )
            self._thread.start()
            logger.success("Remote viewer communicator thread started.")

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            try:
                self.data_queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=2.0)
            logger.info("Remote viewer communicator thread stopped.")

    def send_data(self, obs: Observation):
        try:
            self.data_queue.put_nowait(obs)
        except queue.Full:
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                pass
            self.data_queue.put_nowait(obs)
            logger.debug("Data queue was full, old data discarded and new one added.")


class RemoteViewerWrapper(Wrapper):
    def __init__(self, env: Env, websocket_uri: str):
        super().__init__(env)
        self.communicator = RemoteViewerCommunicator(websocket_uri)
        self.communicator.start()
        logger.info(
            f"RemoteViewerWrapper initialized for env '{env.unwrapped.__class__.__name__}'."
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.communicator.send_data(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.communicator.send_data(obs)
        return obs, info

    def close(self):
        logger.info("Closing RemoteViewerWrapper.")
        self.communicator.stop()
        self.env.close()
