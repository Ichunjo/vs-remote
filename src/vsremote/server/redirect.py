from __future__ import annotations

import io
import logging
import sys
import threading
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from logging import Handler, LogRecord
from typing import Any, Literal, Self, TextIO, override

import msgspec

from ..protocol import RemoteLogRecord, StreamEvent, StreamOutputEvent

# Scoped context flag to suppress re-capturing streams when log handlers write to stderr/stdout
_IN_LOGGING: ContextVar[bool] = ContextVar("in_logging", default=False)
_PATCH_LOCK = threading.Lock()
_PATCH_COUNT = 0
_ORIG_CALL_HANDLERS = logging.Logger.callHandlers


def _patched_call_handlers(self: logging.Logger, record: LogRecord) -> None:
    token = _IN_LOGGING.set(True)
    try:
        _ORIG_CALL_HANDLERS(self, record)
    finally:
        _IN_LOGGING.reset(token)


def _install_logging_hook() -> None:
    global _PATCH_COUNT

    with _PATCH_LOCK:
        if _PATCH_COUNT == 0:
            setattr(logging.Logger, "callHandlers", _patched_call_handlers)
        _PATCH_COUNT += 1


def _uninstall_logging_hook() -> None:
    global _PATCH_COUNT

    with _PATCH_LOCK:
        if _PATCH_COUNT > 0:
            _PATCH_COUNT -= 1
            if _PATCH_COUNT == 0:
                setattr(logging.Logger, "callHandlers", _ORIG_CALL_HANDLERS)


def _serialize_log_args(args: Any) -> tuple[Any, ...]:
    if not args:
        return ()

    raw_args = args if isinstance(args, tuple) else (args,)
    try:
        msgspec.msgpack.encode(raw_args)
        return raw_args
    except Exception:
        return tuple(str(a) for a in raw_args)


class LogForwarder(Handler):
    """Logging handler that converts LogRecords into serializable RemoteLogRecords."""

    def __init__(self, dispatch: Callable[[StreamEvent], None]) -> None:
        super().__init__()
        self.dispatch = dispatch

    @override
    def emit(self, record: LogRecord) -> None:
        if getattr(record, "_is_remote", False):
            return

        try:
            exc_text = None
            if record.exc_info and not record.exc_text:
                record.exc_text = self.format(record)
            if record.exc_text:
                exc_text = record.exc_text

            self.dispatch(
                RemoteLogRecord(
                    name=record.name,
                    levelno=record.levelno,
                    levelname=record.levelname,
                    msg=record.msg,
                    args=_serialize_log_args(record.args),
                    filename=record.filename,
                    lineno=record.lineno,
                    funcName=record.funcName,
                    created=record.created,
                    exc_text=exc_text,
                    stack_info=record.stack_info,
                )
            )
        except Exception:
            self.handleError(record)


class StreamRedirector(io.TextIOBase):
    """
    Standard text stream redirector for sys.stdout/sys.stderr.
    Tees output to the underlying stream and dispatches StreamOutputEvents.
    """

    def __init__(self, name: Literal["stdout", "stderr"], dispatch: Callable[[StreamEvent], None]) -> None:
        super().__init__()
        self.name: Literal["stdout", "stderr"] = name
        self.dispatch = dispatch
        self._target: TextIO = getattr(sys, name)
        self._installed = False

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._target, attr)

    @override
    def __enter__(self) -> Self:
        self.install()
        return self

    @override
    def __exit__(self, *_: object) -> None:
        self.uninstall()

    @property
    @override
    def closed(self) -> bool:
        return getattr(self._target, "closed", False)

    @override
    def close(self) -> None: ...

    @override
    def fileno(self) -> int:
        fileno_fn = getattr(self._target, "fileno", None)
        if fileno_fn is not None:
            return fileno_fn()
        raise io.UnsupportedOperation("Underlying stream has no fileno")

    @override
    def flush(self) -> None:
        if not self.closed:
            with suppress(ValueError):
                self._target.flush()

    @override
    def isatty(self) -> bool:
        return getattr(self._target, "isatty", lambda: False)()

    @override
    def readable(self) -> bool:
        return False

    @override
    def seekable(self) -> bool:
        return False

    @override
    def write(self, text: str) -> int:
        written = self._target.write(text)
        if text and not _IN_LOGGING.get():
            self.dispatch(StreamOutputEvent(stream=self.name, text=text))
        return written

    @override
    def writelines(self, lines: Iterable[str]) -> None:  # type: ignore[override]
        for line in lines:
            self.write(line)

    @override
    def writable(self) -> bool:
        return True

    def install(self) -> None:
        if not self._installed:
            self._target = getattr(sys, self.name)
            setattr(sys, self.name, self)
            self._installed = True
            _install_logging_hook()

    def uninstall(self) -> None:
        if self._installed:
            setattr(sys, self.name, self._target)
            self._installed = False
            _uninstall_logging_hook()


@contextmanager
def capture_streams(dispatch: Callable[[StreamEvent], None]) -> Generator[None, None, None]:
    log_handler = LogForwarder(dispatch)

    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    with StreamRedirector("stdout", dispatch), StreamRedirector("stderr", dispatch):
        try:
            yield
        finally:
            root_logger.removeHandler(log_handler)
