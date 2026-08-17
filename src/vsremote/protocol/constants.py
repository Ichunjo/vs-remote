from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal, NoReturn, Self

from ..exceptions import (
    RemoteAuthenticationError,
    RemoteCommandError,
    RemoteError,
    RemoteExecutionError,
    RemoteNotFoundError,
    RemotePayloadError,
    RemotePermissionError,
)

type Compression = Literal["none", "zstd"]
type StatusCodeError = Literal[
    StatusCode.ERROR,
    StatusCode.NOT_FOUND,
    StatusCode.INVALID_COMMAND,
    StatusCode.INVALID_PAYLOAD,
    StatusCode.UNAUTHORIZED,
    StatusCode.PERMISSION_DENIED,
]


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_ADDRESS = f"tcp://{DEFAULT_HOST}:{DEFAULT_PORT}"
PROTOCOL_VERSION = 1


class Command(IntEnum):
    """Protocol command identifiers."""

    PING = 1
    LIST_OUTPUTS = 2
    GET_CLIP_INFO = 3
    GET_FRAME = 4
    CLOSE = 5
    SUBSCRIBE_STREAM = 6
    UNSUBSCRIBE_STREAM = 7
    RELOAD = 8
    LOAD_CODE = 9
    LOAD_SCRIPT = 10


class StatusCode(IntEnum):
    """Response status codes."""

    OK = 0
    ERROR = 1, RemoteExecutionError
    NOT_FOUND = 2, RemoteNotFoundError
    INVALID_COMMAND = 3, RemoteCommandError
    INVALID_PAYLOAD = 4, RemotePayloadError
    UNAUTHORIZED = 5, RemoteAuthenticationError
    PERMISSION_DENIED = 6, RemotePermissionError

    exception_cls: type[RemoteError] | None

    def __new__(cls, value: int, exception_cls: type[RemoteError] | None = None) -> Self:
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.exception_cls = exception_cls
        return obj

    def raise_for_status(self: StatusCodeError, message: str, payload: Any = None) -> NoReturn:  # type: ignore[misc]
        """Raise the corresponding RemoteError if this status code represents an error."""
        if not self.exception_cls:
            raise NotImplementedError

        raise self.exception_cls(message, status=self, payload=payload)
