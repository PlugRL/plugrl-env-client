import os
import time
from typing import Any, Dict, Optional, Tuple

from loguru import logger
import websockets.sync.client
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from plugrl_protocol import msgpack_numpy
from plugrl_protocol.websocket_protocol import (
    MessageType,
    SERVER_RESYNC_REASON,
    SERVER_STOP_REASON,
)

from . import base_agent as _base_agent


def _get_close_details(exc: ConnectionClosed) -> tuple[int | None, str]:
    if exc.rcvd is not None:
        return exc.rcvd.code, exc.rcvd.reason
    if exc.sent is not None:
        return exc.sent.code, exc.sent.reason
    return None, ""


class ServerStopped(RuntimeError):
    """Raised when the server explicitly requests env clients to stop."""


WS_PING_INTERVAL = float(os.environ.get("PLUGRL_WS_PING_INTERVAL_SECONDS", "60"))
WS_PING_TIMEOUT = float(os.environ.get("PLUGRL_WS_PING_TIMEOUT_SECONDS", "180"))
WS_CLOSE_TIMEOUT = float(os.environ.get("PLUGRL_WS_CLOSE_TIMEOUT_SECONDS", "30"))


class WebSocketEnvClientAgent(_base_agent.BaseAgent):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        reconnect_on_server_stop: bool = False,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._reconnect_on_server_stop = reconnect_on_server_stop
        self._ws, self._server_metadata = self._wait_for_server()

    def _close_connection(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.close()
        except Exception:
            pass
        finally:
            self._ws = None

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """Block until a connection to the server is established and metadata is received."""

        reconnect_delay_s = 5
        logger.info(f"Waiting for server at {self._uri}...")

        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                    close_timeout=WS_CLOSE_TIMEOUT,
                )

                metadata_msg = msgpack_numpy.unpackb(conn.recv())
                if metadata_msg.get("message_type") != str(MessageType.METADATA):
                    logger.warning(
                        "Expected METADATA but received "
                        f"{metadata_msg.get('message_type')}. Reconnecting."
                    )
                    conn.close()
                    raise ConnectionRefusedError

                return conn, metadata_msg["data"]

            except ConnectionClosedOK as exc:
                close_code, close_reason = _get_close_details(exc)
                if close_reason == SERVER_STOP_REASON:
                    if self._reconnect_on_server_stop:
                        logger.info(
                            "Server requested env client shutdown while reconnecting, "
                            "but reconnect_on_server_stop is enabled. "
                            f"Retrying in {reconnect_delay_s} seconds..."
                        )
                        time.sleep(reconnect_delay_s)
                        continue
                    raise ServerStopped(
                        "Server requested env client shutdown while reconnecting."
                    ) from exc

                logger.warning(
                    "Server closed the connection during metadata exchange. "
                    f"Retrying in {reconnect_delay_s} seconds... code={close_code}, "
                    f"reason={close_reason or '<empty>'}"
                )
                time.sleep(reconnect_delay_s)
            except (ConnectionRefusedError, TimeoutError, ConnectionClosedError):
                logger.warning(
                    "Server not available or connection failed. "
                    f"Retrying in {reconnect_delay_s} seconds..."
                )
                time.sleep(reconnect_delay_s)
            except Exception as exc:
                logger.error(
                    "Error during initial connection or metadata exchange: "
                    f"{exc}. Retrying in {reconnect_delay_s} seconds..."
                )
                time.sleep(reconnect_delay_s)

    def _ensure_connection(self) -> None:
        if self._ws is not None:
            return
        logger.warning("Connection closed. Attempting to re-establish connection.")
        self._ws, self._server_metadata = self._wait_for_server()

    def infer(
        self,
        obs: Dict,
        *,
        env_indices: Any,
        step_ids: Any,
    ) -> Dict:  # noqa: UP006
        while True:
            self._ensure_connection()
            ws = self._ws
            if ws is None:
                raise RuntimeError("WebSocket connection is not available.")

            try:
                packed_data = self._packer.pack(
                    {
                        "message_type": str(MessageType.INFER),
                        "data": obs,
                        "env_indices": env_indices,
                        "step_ids": step_ids,
                    }
                )
                ws.send(packed_data)

                response = ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"Error in inference server:\n{response}")

                return msgpack_numpy.unpackb(response)["data"]
            except ConnectionClosedOK as exc:
                close_code, close_reason = _get_close_details(exc)
                self._close_connection()

                if close_reason == SERVER_STOP_REASON:
                    if self._reconnect_on_server_stop:
                        logger.info(
                            "Server requested env client shutdown during INFER/ACTION exchange, "
                            "but reconnect_on_server_stop is enabled. "
                            "Waiting for server to come back and retrying..."
                        )
                        continue
                    raise ServerStopped(
                        "Server requested env client shutdown after algorithm stop."
                    ) from exc

                if close_reason == SERVER_RESYNC_REASON:
                    logger.info(
                        "Server requested session resync during INFER/ACTION exchange. "
                        "Waiting for server to come back and retrying..."
                    )
                    continue

                logger.warning(
                    "Connection closed normally during INFER/ACTION exchange. "
                    f"Waiting for server to come back and retrying... code={close_code}, "
                    f"reason={close_reason or '<empty>'}"
                )
                continue
            except ConnectionClosedError as exc:
                logger.warning(
                    "Connection closed during INFER/ACTION exchange. "
                    f"Error: {exc}. Retrying..."
                )
                self._close_connection()
                continue
            except Exception:
                self._close_connection()
                raise

    def feedback(
        self,
        obs: Dict,
        rewards: Any,
        terminated: Any,
        truncated: Any,
        info: Dict,
        *,
        env_indices: Any,
        step_ids: Any,
    ) -> None:
        while True:
            self._ensure_connection()
            ws = self._ws
            if ws is None:
                raise RuntimeError("WebSocket connection is not available.")

            try:
                packed_data = self._packer.pack(
                    {
                        "message_type": str(MessageType.FEEDBACK),
                        "env_indices": env_indices,
                        "step_ids": step_ids,
                        "data": {
                            "obs": obs,
                            "rewards": rewards,
                            "terminated": terminated,
                            "truncated": truncated,
                            "info": info,
                        },
                    }
                )
                ws.send(packed_data)
                return
            except ConnectionClosedOK as exc:
                close_code, close_reason = _get_close_details(exc)
                self._close_connection()

                if close_reason == SERVER_STOP_REASON:
                    if self._reconnect_on_server_stop:
                        logger.info(
                            "Server requested env client shutdown during FEEDBACK send, "
                            "but reconnect_on_server_stop is enabled. "
                            "Waiting for server to come back and retrying..."
                        )
                        continue
                    raise ServerStopped(
                        "Server requested env client shutdown after algorithm stop."
                    ) from exc

                if close_reason == SERVER_RESYNC_REASON:
                    logger.info(
                        "Server requested session resync during FEEDBACK send. "
                        "Dropping stale feedback and resuming from the next infer request."
                    )
                    return

                logger.warning(
                    "Connection closed normally during FEEDBACK send. "
                    f"Waiting for server to come back and retrying... code={close_code}, "
                    f"reason={close_reason or '<empty>'}"
                )
                continue
            except ConnectionClosedError as exc:
                logger.warning(
                    f"Connection closed during FEEDBACK send. Error: {exc}. Retrying..."
                )
                self._close_connection()
                continue
            except Exception:
                self._close_connection()
                raise

    def reset(self) -> None:
        return
