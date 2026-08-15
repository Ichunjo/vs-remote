from __future__ import annotations

from enum import IntEnum
from typing import Literal

type Compression = Literal["none", "zstd"]

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
    ERROR = 1
    NOT_FOUND = 2
    INVALID_COMMAND = 3
    INVALID_PAYLOAD = 4
    UNAUTHORIZED = 5
    PERMISSION_DENIED = 6
