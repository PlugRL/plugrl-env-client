from __future__ import annotations

from queue import Queue
from threading import Thread
from typing import Any, Callable

from loguru import logger


class AsyncRecorderSink:
    def __init__(
        self,
        *,
        name: str,
        handler: Callable[[Any], None],
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._handler = handler
        self._on_close = on_close
        self._queue: Queue[Any | None] = Queue()
        self._thread = Thread(target=self._run, name=name, daemon=False)
        self._error: BaseException | None = None
        self._thread.start()

    def submit(self, event: Any) -> None:
        self._raise_if_failed()
        self._queue.put(event)

    def close(self) -> None:
        self._raise_if_failed()
        self._queue.put(None)
        self._thread.join()
        self._raise_if_failed()

    def _run(self) -> None:
        try:
            while True:
                event = self._queue.get()
                if event is None:
                    break
                self._handler(event)
        except BaseException as exc:
            self._error = exc
            logger.exception("Recorder async write failed")
        finally:
            if self._on_close is not None:
                self._on_close()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Recorder async write failed") from self._error
