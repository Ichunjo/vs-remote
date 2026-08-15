from __future__ import annotations

from collections.abc import Sequence
from logging import getLogger
from typing import Any, Literal, Self, overload

import msgspec
import vapoursynth as vs
from typing_extensions import TypeForm

from .codec import unpack_payload
from .constants import Command, Compression, StatusCode

type ResponseEnvelopeLike[T] = ResponseEnvelope[T]

logger = getLogger(__name__)


class ResponseEnvelope[T](msgspec.Struct, frozen=True):
    """Structured representation of a parsed ZeroMQ DEALER multipart response."""

    status: StatusCode
    payload: T
    payload_bytes: bytes = b""
    extra_frames: list[bytes] = msgspec.field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        """Check if the server responded with OK status."""
        return self.status == StatusCode.OK

    @overload
    @classmethod
    def from_frames[S](cls, frames: Sequence[bytes], /, target_type: TypeForm[S]) -> ResponseEnvelopeLike[S]: ...
    @overload
    @classmethod
    def from_frames(cls, frames: Sequence[bytes], /, target_type: None = None) -> ResponseEnvelopeLike[bytes]: ...
    @classmethod
    def from_frames(cls, frames: Sequence[bytes], target_type: Any = None) -> ResponseEnvelopeLike[Any]:
        match frames:
            case [status_byte, payload_bytes, *extra]:
                pass
            case [status_byte]:
                payload_bytes, extra = b"", []
            case _:
                raise ValueError(f"Malformed multipart response: expected at least 1 part, got {len(frames)}")

        try:
            status = StatusCode(int.from_bytes(status_byte, byteorder="big"))
        except ValueError as err:
            raise ValueError(f"Unknown status code byte: {status_byte!r}") from err

        if target_type is None or target_type is bytes:
            payload: Any = payload_bytes
        elif status == StatusCode.OK:
            payload = unpack_payload(payload_bytes, target_type)
        else:
            try:
                payload = unpack_payload(payload_bytes, target_type)
            except Exception:
                payload = unpack_payload(payload_bytes)

        return cls(status=status, payload=payload, payload_bytes=payload_bytes, extra_frames=extra)


class RequestEnvelope(msgspec.Struct, frozen=True):
    """Structured representation of a parsed ZeroMQ ROUTER multipart request."""

    identity: bytes
    request_id_bytes: bytes
    request_id: int
    command: Command
    payload_bytes: bytes = b""
    auth_token: str | None = None
    extra_frames: list[bytes] = msgspec.field(default_factory=list)

    @classmethod
    def from_frames(cls, frames: list[bytes]) -> Self:
        match frames:
            case [identity, req_id_bytes, cmd_byte, payload_bytes, *extra]:
                pass
            case [identity, req_id_bytes, cmd_byte]:
                payload_bytes, extra = b"", []
            case _:
                raise ValueError(f"Malformed multipart request: expected at least 3 parts, got {len(frames)}")

        try:
            cmd = Command(int.from_bytes(cmd_byte, byteorder="big"))
        except ValueError as err:
            raise ValueError(f"Unknown command byte: {cmd_byte!r}") from err

        req_id = int.from_bytes(req_id_bytes, byteorder="big")

        auth_token = None
        if extra and extra[0]:
            try:
                auth_token = extra[0].decode("utf-8")
            except UnicodeDecodeError as e:
                logger.error("Fail to decode auth token %s", e)

        return cls(
            identity=identity,
            request_id_bytes=req_id_bytes,
            request_id=req_id,
            command=cmd,
            payload_bytes=payload_bytes,
            auth_token=auth_token,
            extra_frames=extra,
        )


class PlaneInfo(msgspec.Struct, frozen=True):
    """Metadata describing a single video plane layout."""

    width: int
    height: int
    bytes_per_sample: int
    size_bytes: int

    @classmethod
    def from_clip(cls, clip: vs.VideoNode) -> list[Self]:
        if not clip.format:
            raise ValueError("Variable format clips are not supported by vs-remote")

        fmt = clip.format
        bps = fmt.bytes_per_sample
        planes = list[Self]()

        for p in range(fmt.num_planes):
            if p == 0:
                pw, ph = clip.width, clip.height
            else:
                pw = clip.width >> fmt.subsampling_w
                ph = clip.height >> fmt.subsampling_h

            planes.append(cls(width=pw, height=ph, bytes_per_sample=bps, size_bytes=pw * ph * bps))

        return planes


class ClipInfo(msgspec.Struct, frozen=True):
    """Static metadata describing a VapourSynth VideoNode."""

    width: int
    height: int
    fps_num: int
    fps_den: int
    num_frames: int
    format_id: int
    format_name: str
    num_planes: int
    bytes_per_sample: int
    subsampling_w: int
    subsampling_h: int
    planes: list[PlaneInfo]
    name: str = ""

    @classmethod
    def from_clip(cls, clip: vs.VideoNode, name: str = "") -> Self:
        if not clip.format:
            raise ValueError("Variable format clips are not supported by vs-remote")

        fmt = clip.format

        return cls(
            width=clip.width,
            height=clip.height,
            fps_num=clip.fps.numerator,
            fps_den=clip.fps.denominator,
            num_frames=clip.num_frames,
            format_id=fmt.id,
            format_name=fmt.name,
            num_planes=fmt.num_planes,
            bytes_per_sample=fmt.bytes_per_sample,
            subsampling_w=fmt.subsampling_w,
            subsampling_h=fmt.subsampling_h,
            planes=PlaneInfo.from_clip(clip),
            name=name,
        )


class OutputItem(msgspec.Struct, frozen=True):
    """Description of an available output on the server."""

    index: int
    name: str
    info: ClipInfo


class OutputIndexRequest(msgspec.Struct, frozen=True):
    """Request payload specifying a target output clip index."""

    output_index: int = 0


class FrameRequest(msgspec.Struct, frozen=True):
    """Request for a specific frame from an output clip."""

    output_index: int = 0
    n: int = 0
    compression: Compression = "zstd"


class FrameHeader(msgspec.Struct, frozen=True):
    """Header sent alongside video frame plane buffers."""

    status: StatusCode
    request_id: int
    n: int
    output_index: int
    compression: Compression
    plane_sizes: list[int] = msgspec.field(default_factory=list)
    props: dict[str, Any] = msgspec.field(default_factory=dict)
    error_message: str = ""


class StreamSubscribeRequest(msgspec.Struct, frozen=True):
    """Request payload for subscribing to server logs and stdout/stderr streams."""

    replay_history: bool = True


class StreamOutputEvent(msgspec.Struct, tag="stream", frozen=True):
    """Raw stdout or stderr text chunk event."""

    stream: Literal["stdout", "stderr"]
    text: str


class RemoteLogRecord(msgspec.Struct, tag="log", frozen=True):
    """Structured log record serialized from Python logging and VapourSynth core logs."""

    name: str
    levelno: int
    levelname: str
    msg: str
    args: tuple[Any, ...] = ()
    filename: str = ""
    lineno: int = 0
    funcName: str = ""
    created: float = 0.0
    exc_text: str | None = None
    stack_info: str | None = None


type StreamEvent = RemoteLogRecord | StreamOutputEvent


class ReloadRequest(msgspec.Struct, frozen=True):
    """Request payload for reloading the active script file on the server."""

    chdir: bool = True


class LoadCodeRequest(msgspec.Struct, frozen=True):
    """Request payload to execute Python code string on the server."""

    code: str
    filename: str = "<remote_code>"


class LoadScriptRequest(msgspec.Struct, frozen=True):
    """Request payload to load or switch to a script file on the server."""

    script_path: str
    chdir: bool = True
