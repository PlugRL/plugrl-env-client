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


class ConnectionReplaced(RuntimeError):
    """The connection the caller's actions came from is gone.

    The server keeps each environment's half of a transition - the previous
    observation, the policy step state - on the connection that answered the
    infer, so a new connection starts with none of it (SPEC section 7.6). Any
    action chunk the caller is still executing was answered on the old one:
    its feedback can never be completed, and must not be sent on the new
    connection. The caller drops every chunk in flight and asks again for
    all of its environments.

    It used not to be told. The agent reconnected inside its next call and
    carried on, so environments in the middle of a chunk kept executing it
    and then sent its feedback on the new connection, and a feedback whose
    send failed was followed by a reconnect inside the next `feedback` call,
    whose first message was then a stale feedback. `plugrl-conformance
    --probe --scenario resync` caught both.
    """


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
        # Which connection this is, and which one answered the caller's last
        # infer. When they differ, the caller is holding actions from a
        # connection that is gone; see ConnectionReplaced.
        self._connection_id = 1
        self._actions_from: int | None = None

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
        self._connection_id += 1

    def _replaced(self, why: str) -> ConnectionReplaced:
        """Tell the caller once; its next infer then goes out as normal."""
        self._actions_from = None
        return ConnectionReplaced(why)

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
            if (
                self._actions_from is not None
                and self._actions_from != self._connection_id
            ):
                # Nothing is sent: the caller has to re-plan every env first.
                raise self._replaced(
                    "The connection was replaced; the action chunks in flight "
                    "came from the old one and are void."
                )

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

                self._actions_from = self._connection_id
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
        """Send one transition, and drop it rather than resend it after a drop.

        Feedback completes a transition the server began when it answered the
        matching infer, and everything it needs to complete it - the previous
        observation, the policy step state, the done flags - lives on that one
        connection. Resending on a new connection does not save the step; it
        makes the server build a transition out of an empty observation and
        store it, with nothing downstream able to tell. SPEC section 7.6.

        So a closed connection here costs the transitions in flight, and that
        is the cheap outcome. This method never reconnects: a feedback is only
        ever sent on the connection that answered its infer. When that
        connection is gone it raises ConnectionReplaced, so the caller drops
        every chunk it is still executing, and the next infer reconnects.
        """
        ws = self._ws
        if ws is None or self._actions_from != self._connection_id:
            raise self._replaced(
                "The connection that answered this feedback's infer is gone; "
                "dropping the feedback."
            )

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
        except ConnectionClosedOK as exc:
            close_code, close_reason = _get_close_details(exc)
            self._close_connection()

            if (
                close_reason == SERVER_STOP_REASON
                and not self._reconnect_on_server_stop
            ):
                raise ServerStopped(
                    "Server requested env client shutdown after algorithm stop."
                ) from exc
            # A resync, a stop the client waits out, or any other close: the
            # feedback is dropped either way. Before, a stop with
            # reconnect_on_server_stop retried, which resent it on the new
            # connection - the one path that broke SPEC section 7.6.
            logger.info(
                "Connection closed during FEEDBACK send. Dropping the "
                "transitions in flight and resuming from the next infer request. "
                f"code={close_code}, reason={close_reason or '<empty>'}"
            )
            raise self._replaced("dropped feedback after a close") from exc
        except ConnectionClosedError as exc:
            logger.warning(
                "Connection closed during FEEDBACK send. Dropping the "
                f"transitions in flight and resuming from the next infer request. {exc}"
            )
            self._close_connection()
            raise self._replaced("dropped feedback after a close") from exc
        except Exception:
            self._close_connection()
            raise

    def reset(self) -> None:
        return
