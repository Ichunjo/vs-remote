from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from logging import getLogger
from typing import Self, TextIO, assert_never, overload

import vapoursynth as vs
from vsengine.futures import UnifiedFuture

from .._strides import copy_plane_strided
from ..protocol import (
    DEFAULT_ADDRESS,
    ClipInfo,
    Compression,
    FrameHeader,
    OutputItem,
    RemoteLogRecord,
    StatusCode,
    StreamEvent,
    StreamOutputEvent,
    decompress_plane,
)
from ..utils import ensure_vsengine_loop
from .transport import ClientTransport

core = vs.core
logger = getLogger(__name__)


class RemoteClient:
    """High-level client for interacting with a remote vs-remote server."""

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        compression: Compression = "zstd",
        *,
        auth_token: str | None = None,
        curve_server_key: str | bytes | None = None,
        curve_public_key: str | bytes | None = None,
        curve_secret_key: str | bytes | None = None,
        stdout: TextIO | None = sys.stdout,
        stderr: TextIO | None = sys.stderr,
        forward_logs: bool = True,
        subscribe_streams: bool = True,
    ) -> None:
        ensure_vsengine_loop()

        if not address.startswith(("tcp://", "ipc://", "inproc://")):
            address = f"tcp://{address}"

        self.address = address
        self.compression: Compression = compression
        self.auth_token = auth_token
        self.curve_server_key = curve_server_key
        self.curve_public_key = curve_public_key
        self.curve_secret_key = curve_secret_key
        self.stdout = stdout
        self.stderr = stderr
        self.forward_logs = forward_logs
        self.subscribe_streams = subscribe_streams
        self.transport = ClientTransport(
            address,
            auth_token=self.auth_token,
            curve_server_key=curve_server_key,
            curve_public_key=curve_public_key,
            curve_secret_key=curve_secret_key,
            on_event=self._handle_event,
            subscribe_streams=subscribe_streams,
        )
        self._streams = {"stdout": self.stdout, "stderr": self.stderr}

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self.start()

    async def __aexit__(self, *args: object) -> None:
        self.close()

    def start(self) -> Self:
        """Start client connection and background transport."""
        self.transport.start()
        return self

    def close(self) -> None:
        """Close client connection and release background transport resources."""
        self.transport.close()

    def ping(self) -> UnifiedFuture[bool]:
        """Check if the remote server is reachable."""
        return self.transport.ping()

    def list_outputs(self) -> UnifiedFuture[list[OutputItem]]:
        """Query all available output clips from the remote server."""
        return self.transport.list_outputs()

    def load_script(self, script_path: str | os.PathLike[str], chdir: bool = True) -> UnifiedFuture[list[OutputItem]]:
        """
        Request the remote server to load or switch to a script file.

        Args:
            script_path: Path to the script file to load.
            chdir: Change the current working directory of the remote server to the script directory before loading.

        Returns:
            A future that resolves to a list of output clips.
        """
        return self.transport.load_script(script_path, chdir=chdir)

    def load_code(self, code: str, filename: str = "<remote_code>") -> UnifiedFuture[list[OutputItem]]:
        """
        Send Python code to execute dynamically on the remote server.

        Args:
            code: The Python code to execute.
            filename: The filename to use for the code.

        Returns:
            A future that resolves to a list of output clips.
        """
        return self.transport.load_code(code, filename=filename)

    def reload(self, chdir: bool = True) -> UnifiedFuture[list[OutputItem]]:
        """
        Request the remote server to reload its active script file from disk.

        Args:
            chdir: Change the current working directory of the remote server to the script directory before reloading.

        Returns:
            A future that resolves to a list of output clips.
        """
        return self.transport.reload(chdir=chdir)

    def get_clip_info(self, output_index: int = 0) -> UnifiedFuture[ClipInfo]:
        """
        Request static clip metadata for the specified output index.

        Args:
            output_index: Output index on the server.

        Returns:
            A future that resolves to the clip info.
        """
        return self.transport.get_clip_info(output_index)

    def request_frame(self, output_index: int, n: int) -> UnifiedFuture[tuple[FrameHeader, list[bytes]]]:
        """
        Request a specific frame from the remote server.

        Args:
            output_index: Output index on the server.
            n: Frame number to request.

        Returns:
            A future that resolves to a tuple of frame header and plane data.
        """
        return self.transport.request_frame(output_index, n, compression=self.compression)

    def get_output(self, output_index: int = 0, prefetch: int = 4, backlog: int | None = None) -> vs.VideoNode:
        """
        Create a local VideoNode proxy mirroring a remote output clip.

        Args:
            output_index: Output index on the server.
            prefetch: Number of subsequent frames to prefetch asynchronously ahead of time (0 to disable).
            backlog: Maximum number of in-flight and prefetched frame requests buffered
                (defaults to max(prefetch * 3, prefetch)).

        Returns:
            A vs.VideoNode that lazily requests and renders frames from the server.
        """
        return create_remote_vnode(
            transport=self.transport,
            output_index=output_index,
            compression=self.compression,
            prefetch=prefetch,
            backlog=backlog,
        )

    def get_outputs(self, prefetch: int = 4, backlog: int | None = None) -> dict[int, vs.VideoNode]:
        """
        Create local VideoNode proxies for all available outputs on the remote server.

        Args:
            prefetch: Number of subsequent frames to prefetch asynchronously ahead of time (0 to disable).
            backlog: Maximum number of in-flight and prefetched frame requests buffered
                (defaults to max(prefetch * 3, prefetch)).

        Returns:
            A dictionary mapping output index to its corresponding vs.VideoNode proxy.
        """
        outputs = self.list_outputs().result(timeout=30.0)
        return {item.index: self.get_output(item.index, prefetch=prefetch, backlog=backlog) for item in outputs}

    def _handle_event(self, event: StreamEvent) -> None:
        match event:
            case RemoteLogRecord():
                record_dict = {
                    "name": event.name,
                    "levelno": event.levelno,
                    "levelname": event.levelname,
                    "msg": event.msg,
                    "args": event.args,
                    "filename": event.filename,
                    "lineno": event.lineno,
                    "funcName": event.funcName,
                    "created": event.created,
                    "exc_text": event.exc_text,
                    "stack_info": event.stack_info,
                }
                rec = logging.makeLogRecord(record_dict)
                setattr(rec, "_is_remote", True)
                if self.forward_logs:
                    logging.getLogger(rec.name).handle(rec)

            case StreamOutputEvent():
                if not (target := self._streams[event.stream]):
                    return
                try:
                    target.write(event.text)
                    target.flush()
                except Exception:
                    logger.exception("Error writing to client stream %s", event.stream)
            case _:
                assert_never(event)


@overload
def source(
    address: str = DEFAULT_ADDRESS,
    /,
    output: int = 0,
    compression: Compression = "zstd",
    prefetch: int = 4,
    backlog: int | None = None,
    *,
    auth_token: str | None = None,
    curve_server_key: str | bytes | None = None,
    curve_public_key: str | bytes | None = None,
    curve_secret_key: str | bytes | None = None,
    stdout: TextIO | None = sys.stdout,
    stderr: TextIO | None = sys.stderr,
    forward_logs: bool = True,
    subscribe_streams: bool = True,
) -> vs.VideoNode: ...
@overload
def source(
    transport: ClientTransport,
    /,
    output: int = 0,
    compression: Compression = "zstd",
    prefetch: int = 4,
    backlog: int | None = None,
) -> vs.VideoNode: ...
def source(
    address_or_transport: str | ClientTransport = DEFAULT_ADDRESS,
    output: int = 0,
    compression: Compression = "zstd",
    prefetch: int = 4,
    backlog: int | None = None,
    *,
    auth_token: str | None = None,
    curve_server_key: str | bytes | None = None,
    curve_public_key: str | bytes | None = None,
    curve_secret_key: str | bytes | None = None,
    stdout: TextIO | None = sys.stdout,
    stderr: TextIO | None = sys.stderr,
    forward_logs: bool = True,
    subscribe_streams: bool = True,
) -> vs.VideoNode:
    """
    Connect to a remote vs-remote server and mirror a video output as a local VideoNode.

    If passing an address, the client will create and start a new transport.
    The client will be closed upon environment cleanup (registered with `vs.register_on_destroy`).

    If passing an existing transport, the client will reuse the transport.
    The transport will NOT be closed when the node is destroyed.

    Args:
        address_or_transport: Remote server address or existing transport.
        output: Remote output index to bind to (default 0).
        compression: Preferred plane compression (default: zstd).
        prefetch: Number of subsequent frames to prefetch ahead of time (default 4).
        backlog: Maximum number of in-flight and prefetched frame requests buffered
            (defaults to max(prefetch * 3, prefetch)).
        auth_token: Optional authentication token.
        curve_server_key: Optional CurveZMQ server public key for end-to-end encryption.
        curve_public_key: Optional CurveZMQ client public key.
        curve_secret_key: Optional CurveZMQ client secret key.
        stdout: Target stream or callable for remote stdout.
        stderr: Target stream or callable for remote stderr.
        forward_logs: Whether to dispatch remote LogRecords to client logging system.
        subscribe_streams: Whether to subscribe to remote streams.

    Returns:
        A vs.VideoNode that fetches frames on demand over the network.
    """
    if isinstance(address_or_transport, str):
        client = RemoteClient(
            address_or_transport,
            compression,
            auth_token=auth_token,
            curve_server_key=curve_server_key,
            curve_public_key=curve_public_key,
            curve_secret_key=curve_secret_key,
            stdout=stdout,
            stderr=stderr,
            forward_logs=forward_logs,
            subscribe_streams=subscribe_streams,
        )
        trans = client.transport
        trans.start()
        vs.register_on_destroy(client.close)
    else:
        trans = address_or_transport

    return create_remote_vnode(
        transport=trans,
        output_index=output,
        compression=compression,
        prefetch=prefetch,
        backlog=backlog,
    )


def create_remote_vnode(
    transport: ClientTransport,
    output_index: int,
    compression: Compression,
    prefetch: int = 4,
    backlog: int | None = None,
) -> vs.VideoNode:
    """
    Construct a VapourSynth VideoNode that lazily requests and renders frames from a remote server.

    Args:
        transport: Connected ClientTransport instance.
        output_index: Output index of the clip on the remote server.
        compression: Preferred plane compression algorithm.
        prefetch: Number of subsequent frames to prefetch asynchronously ahead of time (0 to disable).
        backlog: Maximum number of in-flight and prefetched frame requests buffered
            (defaults to max(prefetch * 3, prefetch)).

    Returns:
        A standard vs.VideoNode proxy.
    """
    info = transport.get_clip_info(output_index).result(timeout=30.0)

    blank = core.std.BlankClip(
        width=info.width,
        height=info.height,
        fpsnum=info.fps_num,
        fpsden=info.fps_den,
        length=info.num_frames,
        format=info.format_id,
        keep=True,
    )

    prefetch = max(prefetch, 0)

    if backlog is None or backlog < 0:
        backlog = max(prefetch * 3, prefetch)
    elif backlog < prefetch:
        backlog = prefetch

    inflight = dict[int, UnifiedFuture[tuple[FrameHeader, list[bytes]]]]()
    lock = threading.Lock()

    def fetch_frame(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        with lock:
            if (fut := inflight.pop(n, None)) is None:
                fut = transport.request_frame(output_index, n, compression=compression)

            # Fire ahead-of-time prefetch requests for subsequent frames within the backlog limit
            if prefetch > 0:
                # Prune stale futures outside the active window [n, n + backlog] (e.g. after seeking)
                if inflight:
                    for stale_n in list(inflight.keys()):
                        if (
                            (stale_n < n or stale_n > n + backlog)  # no fmt
                            and (stale_fut := inflight.pop(stale_n, None)) is not None
                        ):
                            stale_fut.cancel()

                for next_n in range(n + 1, min(n + prefetch + 1, info.num_frames)):
                    if len(inflight) >= backlog:
                        break
                    if next_n not in inflight:
                        inflight[next_n] = transport.request_frame(output_index, next_n, compression=compression)

        header, plane_parts = fut.result(timeout=30.0)

        if header.status != StatusCode.OK:
            header.status.raise_for_status(
                f"Failed to fetch remote frame (n={n}, output={output_index}):\n{header.error_message}"
            )

        f_out = f.copy()

        # Populate frame properties
        f_out.props.update(header.props)

        # Decompress and copy planar bytes into the frame buffer
        for p in range(info.num_planes):
            plane_info = info.planes[p]
            decompressed = decompress_plane(plane_parts[p], plane_info.size_bytes, header.compression)

            dst_ptr = f_out.get_write_ptr(p)
            stride = f_out.get_stride(p)
            row_size = plane_info.width * plane_info.bytes_per_sample

            if stride == row_size:
                ctypes.memmove(dst_ptr, decompressed, len(decompressed))
            else:
                dst_addr = dst_ptr.value
                assert dst_addr
                copy_plane_strided(
                    dst_addr,
                    decompressed,
                    plane_info.width,
                    plane_info.height,
                    plane_info.bytes_per_sample,
                    stride,
                )

        return f_out

    return core.std.ModifyFrame(blank, blank, fetch_frame)
