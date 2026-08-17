from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from logging import Handler, LogRecord
from typing import Literal, TextIO, override

import msgspec

from ..protocol import RemoteLogRecord, StreamEvent, StreamOutputEvent

_IN_LOGGING = threading.local()
_ORIG_HANDLER_HANDLE = Handler.handle
_PATCH_LOCK = threading.Lock()
_PATCH_COUNT = 0
_ACTIVE_REDIRECTORS: dict[str, StreamRedirector] = {}


def _restore_installed_redirectors() -> None:
    for name, redir in list(_ACTIVE_REDIRECTORS.items()):
        current = getattr(sys, name, None)
        if current is not None and current is not redir:
            redir._original_stream = current
            setattr(sys, name, redir)


def _patched_handler_handle(self: Handler, record: LogRecord) -> bool:
    _IN_LOGGING.depth = getattr(_IN_LOGGING, "depth", 0) + 1
    try:
        return _ORIG_HANDLER_HANDLE(self, record)
    finally:
        _IN_LOGGING.depth -= 1
        _restore_installed_redirectors()


def _install_logging_patch() -> None:
    global _PATCH_COUNT

    with _PATCH_LOCK:
        if _PATCH_COUNT == 0:
            setattr(Handler, "handle", _patched_handler_handle)
        _PATCH_COUNT += 1


def _uninstall_logging_patch() -> None:
    global _PATCH_COUNT

    with _PATCH_LOCK:
        if _PATCH_COUNT > 0:
            _PATCH_COUNT -= 1

            if _PATCH_COUNT == 0:
                setattr(Handler, "handle", _ORIG_HANDLER_HANDLE)


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
            args = record.args
            if isinstance(args, tuple):
                try:
                    msgspec.msgpack.encode(args)
                    serializable_args = args
                except Exception:
                    serializable_args = tuple(str(a) for a in args)
            elif isinstance(args, dict):
                try:
                    msgspec.msgpack.encode(args)
                    serializable_args = (args,)
                except Exception:
                    serializable_args = (str(args),)
            elif args:
                serializable_args = (str(args),)
            else:
                serializable_args = ()

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
                    args=serializable_args,
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


class StreamRedirector:
    """Redirector for sys.stdout and sys.stderr that emits StreamOutputEvents."""

    def __init__(self, name: Literal["stdout", "stderr"], dispatch: Callable[[StreamEvent], None]) -> None:
        self.name: Literal["stdout", "stderr"] = name
        self.dispatch = dispatch
        self._original_stream: TextIO = getattr(sys, name)
        self._installed = False

    def __getattr__(self, name: str) -> object:
        return getattr(self._original_stream, name)

    def install(self) -> None:
        if not self._installed:
            self._original_stream = getattr(sys, self.name)
            setattr(sys, self.name, self)
            _ACTIVE_REDIRECTORS[self.name] = self
            self._installed = True
            _install_logging_patch()

    def uninstall(self) -> None:
        if self._installed:
            _ACTIVE_REDIRECTORS.pop(self.name, None)
            setattr(sys, self.name, self._original_stream)
            self._installed = False
            _uninstall_logging_patch()

    def write(self, text: str) -> int:
        res = self._original_stream.write(text)
        if text and getattr(_IN_LOGGING, "depth", 0) == 0:
            self.dispatch(StreamOutputEvent(stream=self.name, text=text))
        return res

    def flush(self) -> None:
        self._original_stream.flush()

    def isatty(self) -> bool:
        return getattr(self._original_stream, "isatty", lambda: False)()


@contextmanager
def capture_streams(dispatch: Callable[[StreamEvent], None]) -> Generator[None, None, None]:
    stdout_redir = StreamRedirector("stdout", dispatch)
    stderr_redir = StreamRedirector("stderr", dispatch)
    log_handler = LogForwarder(dispatch)

    stdout_redir.install()
    stderr_redir.install()
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    try:
        yield
    finally:
        stdout_redir.uninstall()
        stderr_redir.uninstall()
        root_logger.removeHandler(log_handler)
