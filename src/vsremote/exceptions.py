from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import StatusCode


class VSRemoteError(Exception):
    """Base exception for all vs-remote errors."""


# Transport & Network Errors
class TransportError(VSRemoteError):
    """Base exception for client transport and connection errors."""


class TransportNotStartedError(TransportError, RuntimeError):
    """Raised when an operation is attempted before the transport is started."""


class TransportNotConnectedError(TransportError, ConnectionError):
    """Raised when transport is not connected to the server."""


class TransportClosedError(TransportError, ConnectionResetError):
    """Raised when an operation is attempted on a closed transport or socket."""


class RemoteTimeoutError(TransportError, TimeoutError):
    """Raised when a remote operation or handshake times out."""


class RemoteError(VSRemoteError):
    """Base exception for errors returned by the remote server."""

    def __init__(self, message: str, status: StatusCode | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload

    @property
    def error(self) -> str:
        """The formatted error message string."""
        return self._payload_attribute("error", str, str(self))

    @property
    def exc_type(self) -> str:
        """The exception class name (e.g. ValueError, SyntaxError)."""
        return self._payload_attribute("exc_type", str, "Error")

    @property
    def exc_msg(self) -> str:
        """The exception message string."""
        return self._payload_attribute("exc_msg", str, "")

    @property
    def filename(self) -> str | None:
        """The source file path where the error occurred, if known."""
        return self._payload_attribute("filename", str, None)

    @property
    def lineno(self) -> int | None:
        """The line number where the error occurred, if known."""
        return self._payload_attribute("lineno", int, None)

    @property
    def code_line(self) -> str | None:
        """The source code line where the error occurred, if known."""
        return self._payload_attribute("code_line", str, None)

    @property
    def formatted_traceback(self) -> str | None:
        """The formatted full traceback from the remote execution, if available."""
        return self._payload_attribute("formatted_traceback", str, None)

    @property
    def frames(self) -> list[Any]:
        """List of stack frames if provided by the remote server."""
        return self._payload_attribute("frames", list[Any], list[Any]())

    def _payload_attribute[T0, T1](self, name: str, t: type[T0], fallback: T1) -> T0 | T1:
        if (val := getattr(self.payload, name, None)) is not None:
            return val
        if isinstance(self.payload, dict) and (val := self.payload.get(name)) is not None:
            return val
        return fallback


class RemoteExecutionError(RemoteError, RuntimeError):
    """Raised when script loading, execution, or frame rendering fails on the remote server."""


class RemotePermissionError(RemoteError, PermissionError):
    """Raised when server denies permission (e.g. dynamic code evaluation disabled)."""


class RemoteAuthenticationError(RemoteError, PermissionError):
    """Raised when authentication fails (missing or invalid auth token)."""


class RemoteNotFoundError(RemoteError, KeyError):
    """Raised when a requested resource (e.g. output clip index) is not found on the server."""


class RemotePayloadError(RemoteError, ValueError):
    """Raised when the server reports an invalid payload."""


class RemoteCommandError(RemoteError, ValueError):
    """Raised when the server reports an unknown or invalid command."""


# Server Runner / Local State Errors
class ScriptRunnerError(VSRemoteError):
    """Base exception for server-side ScriptRunner errors."""


class ScriptNotLoadedError(ScriptRunnerError, RuntimeError):
    """Raised when an operation requires an active script that is not loaded."""


class EnvironmentNotSetError(ScriptRunnerError, RuntimeError):
    """Raised when an operation requires a VapourSynth environment that is not set."""


class OutputNotFoundError(ScriptRunnerError, KeyError):
    """Raised when an output clip index is not registered in the ScriptRunner."""


# Protocol & Wire Serialization Errors
class ProtocolError(VSRemoteError, ValueError):
    """Base exception for protocol framing and codec errors."""


class MalformedMessageError(ProtocolError):
    """Raised when multipart ZeroMQ frames cannot be parsed."""


class UnknownCommandError(ProtocolError):
    """Raised when an unknown command byte is received."""


class UnknownStatusCodeError(ProtocolError):
    """Raised when an unknown status code byte is received."""


class UnsupportedFormatError(VSRemoteError, ValueError):
    """Raised when a clip has an unsupported format (e.g. variable format)."""
