from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from collections.abc import Callable
from logging import getLogger
from typing import Any, Self, overload

import zmq
import zmq.asyncio
from typing_extensions import TypeForm
from vsengine.futures import UnifiedFuture

from ..exceptions import (
    RemoteTimeoutError,
    TransportClosedError,
    TransportError,
    TransportNotConnectedError,
    TransportNotStartedError,
)
from ..protocol import (
    DEFAULT_ADDRESS,
    CancelRequest,
    ClipInfo,
    Command,
    Compression,
    FrameHeader,
    FrameRequest,
    LoadCodeRequest,
    LoadScriptRequest,
    OutputIndexRequest,
    OutputItem,
    ReloadRequest,
    RemoteLogRecord,
    ResponseEnvelope,
    StatusCode,
    StreamEvent,
    StreamOutputEvent,
    StreamSubscribeRequest,
    pack_payload,
    unpack_payload,
    validate_curve_key,
)

logger = getLogger(__name__)


class ClientTransport:
    """Thread-safe ZeroMQ DEALER transport client for multiplexing frame requests and logs."""

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        *,
        auth_token: str | None = None,
        curve_server_key: str | bytes | None = None,
        curve_public_key: str | bytes | None = None,
        curve_secret_key: str | bytes | None = None,
        on_event: Callable[[StreamEvent], None] | None = None,
        subscribe_streams: bool = True,
        replay_history: bool = True,
    ) -> None:
        self._ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._send_queue: asyncio.Queue[list[bytes] | None] | None = None

        self._pending = dict[int, UnifiedFuture[list[bytes]]]()
        self._next_request_id = 1
        self._lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._running = False
        self._start_lock = threading.Lock()
        self._started = False

        if (curve_public_key is None) != (curve_secret_key is None):
            raise ValueError("curve_public_key and curve_secret_key must both be specified for client authentication")

        if (curve_public_key or curve_secret_key) and not curve_server_key:
            raise ValueError("curve_public_key and curve_secret_key require curve_server_key to be specified")

        if not address.startswith(("tcp://", "ipc://", "inproc://")):
            address = f"tcp://{address}"

        self.address = address
        self.auth_token = auth_token
        self.curve_server_key = validate_curve_key(curve_server_key, "curve_server_key")
        self.curve_public_key = validate_curve_key(curve_public_key, "curve_public_key")
        self.curve_secret_key = validate_curve_key(curve_secret_key, "curve_secret_key")
        self.on_event = on_event
        self.subscribe_streams = subscribe_streams
        self.replay_history = replay_history

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self.start()

    async def __aexit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def start(self) -> Self:
        """
        Start background worker thread and establish ZeroMQ connection (idempotent).

        Returns:
            The connected ClientTransport instance.

        Raises:
            RemoteTimeoutError: If the background worker thread fails to initialize within timeout (5s).
        """
        with self._start_lock:
            if not self._started or not self._running:
                self._start_worker_thread()
                self._started = True
                if self.subscribe_streams:
                    self.subscribe_stream(replay_history=self.replay_history)
        return self

    def close(self) -> None:
        """Close socket, cancel pending requests, and stop background transport thread."""
        with self._start_lock:
            if not self._started or not self._running:
                return

            self._running = False

            with self._lock:
                for pending_fut in self._pending.values():
                    if not pending_fut.done():
                        pending_fut.set_exception(ConnectionResetError("Transport closed"))
                self._pending.clear()

            # Cancel active coroutines in the worker loop
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._cancel_tasks)

            if self._thread and self._thread.is_alive() and self._thread != threading.current_thread():
                self._thread.join(timeout=1.0)
            self._thread = None

            self._started = False
            logger.debug("Client transport closed")

    def _cancel_tasks(self) -> None:
        for task in asyncio.all_tasks(self._loop):
            task.cancel()

    @overload
    def send_request(
        self,
        cmd: Command,
        payload: Any = None,
        *,
        response_type: None = None,
    ) -> UnifiedFuture[ResponseEnvelope[bytes]]: ...

    @overload
    def send_request[T](
        self,
        cmd: Command,
        payload: Any = None,
        *,
        response_type: TypeForm[T],
    ) -> UnifiedFuture[ResponseEnvelope[T]]: ...

    def send_request[T](
        self,
        cmd: Command,
        payload: Any = None,
        *,
        response_type: Any | None = None,
    ) -> UnifiedFuture[ResponseEnvelope[Any]]:
        """
        Send a request asynchronously over the transport (thread-safe).

        Args:
            cmd: Command enum to execute.
            payload: Optional dataclass or dictionary payload to MsgPack serialize.
            response_type: Optional expected payload type to decode in the ResponseEnvelope.

        Returns:
            A UnifiedFuture resolving to a typed ResponseEnvelope containing status, decoded payload, and extra frames.

        Raises (via Future):
            TransportNotStartedError: If the transport has not been started.
            TransportClosedError: If the transport is closed or closing.
            TransportNotConnectedError: If the internal transport queue/loop is unavailable.
            MalformedMessageError: If the server response cannot be framed.
            UnknownStatusCodeError: If the server returns an unrecognized status code byte.
        """
        fut = UnifiedFuture[list[bytes]]()

        def to_response_envelope(frames: list[bytes]) -> ResponseEnvelope[T] | ResponseEnvelope[bytes]:
            return ResponseEnvelope.from_frames(frames, response_type)

        if not self._started:
            return fut.reject(TransportNotStartedError("Transport is not started"))

        if not self._running:
            return fut.reject(TransportClosedError("ClientTransport is closed"))

        with self._lock:
            req_id = self._next_request_id
            self._next_request_id = ((self._next_request_id + 1) & 0x7FFFFFFF) or 1
            self._pending[req_id] = fut

        payload_bytes = pack_payload(payload) if payload is not None else b""
        try:
            self._send_message(req_id, cmd, payload_bytes)
        except Exception as exc:
            with self._lock:
                self._pending.pop(req_id, None)
            fut.set_exception(exc)

        return fut.map(to_response_envelope, cancel_cb=lambda: self._cancel_request(req_id))

    def ping(self) -> UnifiedFuture[bool]:
        """
        Check connection liveness to the remote server.

        Returns:
            A UnifiedFuture resolving to True if the server responded with PONG,
            False otherwise (suppresses all errors).
        """
        return (
            self.send_request(Command.PING)
            .map(lambda resp: resp.status == StatusCode.OK and resp.payload == b"PONG")
            .catch(lambda _: False)
        )

    def list_outputs(self) -> UnifiedFuture[list[OutputItem]]:
        """
        List all available VideoNode outputs on the server.

        Returns:
            A UnifiedFuture resolving to a list of OutputItem metadata.

        Raises (via Future):
            TransportNotStartedError: If the transport is not started.
            TransportClosedError: If the transport is closed.
            RemoteAuthenticationError: If authentication failed (StatusCode.UNAUTHORIZED).
            RemoteExecutionError: If the server encountered an error querying outputs (StatusCode.ERROR).
            RemoteError: If any other server-side error occurred.
        """
        return self.send_request(
            Command.LIST_OUTPUTS,
            response_type=list[OutputItem],
        ).map(lambda r: r.raise_for_status("Failed to list outputs").payload)

    def load_script(self, script_path: str | os.PathLike[str], chdir: bool = True) -> UnifiedFuture[list[OutputItem]]:
        """
        Request the remote server to load or switch to a script file.

        Args:
            script_path: Path to the script file to load.
            chdir: Change the current working directory of the remote server to the script directory before loading.

        Returns:
            A UnifiedFuture resolving to a list of available OutputItem metadata from the loaded script.

        Raises (via Future):
            TransportNotStartedError: If the transport is not started.
            TransportClosedError: If the transport is closed.
            RemoteAuthenticationError: If authentication failed (StatusCode.UNAUTHORIZED).
            RemoteExecutionError: If script evaluation or execution failed on the server (StatusCode.ERROR).
            RemoteNotFoundError: If the script file does not exist (StatusCode.NOT_FOUND).
            RemoteError: If any other server-side error occurred.
        """
        return self.send_request(
            Command.LOAD_SCRIPT,
            LoadScriptRequest(os.fspath(script_path), chdir),
            response_type=list[OutputItem],
        ).map(lambda r: r.raise_for_status(f"Failed to load script {script_path}").payload)

    def load_code(self, code: str, filename: str = "<remote_code>") -> UnifiedFuture[list[OutputItem]]:
        """
        Request the remote server to execute Python/VapourSynth code dynamically.

        Args:
            code: Python code string to execute.
            filename: Virtual filename for traceback and error reporting.

        Returns:
            A UnifiedFuture resolving to a list of available OutputItem metadata.

        Raises (via Future):
            TransportNotStartedError: If the transport is not started.
            TransportClosedError: If the transport is closed.
            RemoteAuthenticationError: If authentication failed (StatusCode.UNAUTHORIZED).
            RemotePermissionError: If dynamic code evaluation is disabled on the server (StatusCode.PERMISSION_DENIED).
            RemoteExecutionError: If code evaluation or execution failed on the server (StatusCode.ERROR).
            RemoteError: If any other server-side error occurred.
        """
        return self.send_request(
            Command.LOAD_CODE,
            LoadCodeRequest(code, filename),
            response_type=list[OutputItem],
        ).map(lambda r: r.raise_for_status("Failed to load code").payload)

    def reload(self, chdir: bool = True) -> UnifiedFuture[list[OutputItem]]:
        """
        Request the remote server to reload its current script file.

        Args:
            chdir: Change the current working directory of the remote server to the script directory before reloading.

        Returns:
            A UnifiedFuture resolving to a list of available OutputItem metadata.

        Raises (via Future):
            TransportNotStartedError: If the transport is not started.
            TransportClosedError: If the transport is closed.
            RemoteAuthenticationError: If authentication failed (StatusCode.UNAUTHORIZED).
            RemoteExecutionError: If reloading the script failed on the server (StatusCode.ERROR).
            RemoteNotFoundError: If no active script is loaded on the server (StatusCode.NOT_FOUND).
            RemoteError: If any other server-side error occurred.
        """
        return self.send_request(
            Command.RELOAD,
            ReloadRequest(chdir=chdir),
            response_type=list[OutputItem],
        ).map(lambda r: r.raise_for_status("Failed to reload script").payload)

    def get_clip_info(self, output_index: int = 0) -> UnifiedFuture[ClipInfo]:
        """
        Fetch static metadata for the specified output index.

        Args:
            output_index: Output clip index on the server.

        Returns:
            A UnifiedFuture resolving to ClipInfo metadata.

        Raises (via Future):
            TransportNotStartedError: If the transport is not started.
            TransportClosedError: If the transport is closed.
            RemoteAuthenticationError: If authentication failed (StatusCode.UNAUTHORIZED).
            RemoteNotFoundError: If the output index was not found on the server (StatusCode.NOT_FOUND).
            RemoteExecutionError: If the server failed to inspect the output clip (StatusCode.ERROR).
            RemoteError: If any other server-side error occurred.
        """
        return self.send_request(
            Command.GET_CLIP_INFO,
            OutputIndexRequest(output_index),
            response_type=ClipInfo,
        ).map(lambda r: r.raise_for_status(f"Failed to retrieve clip info for output {output_index}").payload)

    def request_frame(
        self,
        output_index: int,
        n: int,
        compression: Compression = "zstd",
    ) -> UnifiedFuture[tuple[FrameHeader, list[bytes]]]:
        """
        Request a specific video frame from the remote server.

        Args:
            output_index: Index of the output clip on the server.
            n: Frame index to retrieve.
            compression: Preferred plane compression (zstd or none).

        Returns:
            A UnifiedFuture resolving to (FrameHeader, list of plane byte buffers).
            Note: If the remote frame render failed, the returned FrameHeader will contain
            status != StatusCode.OK and header.error_message containing the error string.

        Raises (via Future):
            TransportNotStartedError: If the transport is not started.
            TransportClosedError: If the transport is closed.
            MalformedMessageError: If the server response cannot be decoded.
        """
        req_payload = FrameRequest(output_index=output_index, n=n, compression=compression)

        def parse_frame_response(resp: ResponseEnvelope[bytes]) -> tuple[FrameHeader, list[bytes]]:
            header = unpack_payload(resp.payload_bytes, FrameHeader)
            return header, resp.extra_frames

        return self.send_request(Command.GET_FRAME, req_payload).map(parse_frame_response)

    def subscribe_stream(self, replay_history: bool = True) -> UnifiedFuture[bool]:
        """
        Subscribe to log records and stream events from the remote server.

        Args:
            replay_history: Whether to replay historical log records upon subscribing.

        Returns:
            A UnifiedFuture resolving to True on successful subscription, False on failure (suppresses all errors).
        """
        req_payload = StreamSubscribeRequest(replay_history=replay_history)
        return (
            self.send_request(Command.SUBSCRIBE_STREAM, req_payload)
            .map(lambda r: r.status == StatusCode.OK)
            .catch(lambda _: False)
        )

    def unsubscribe_stream(self) -> UnifiedFuture[bool]:
        """
        Unsubscribe from log records and stream events.

        Returns:
            A UnifiedFuture resolving to True on successful unsubscription, False on failure (suppresses all errors).
        """
        return (
            self.send_request(Command.UNSUBSCRIBE_STREAM)
            .map(lambda r: r.status == StatusCode.OK)
            .catch(lambda _: False)
        )

    def _start_worker_thread(self) -> None:
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._worker, name="VSRemoteTransport", daemon=True)
        self._thread.start()

        if not self._ready_event.wait(timeout=5.0):
            raise RemoteTimeoutError("Timed out waiting for transport worker thread to initialize")

    def _worker(self) -> None:
        try:
            asyncio.run(self._async_worker(), loop_factory=asyncio.SelectorEventLoop)
        finally:
            self._loop = None

    async def _async_worker(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._send_queue = asyncio.Queue()
        self._ctx, self._socket = self._create_socket()
        self._running = True
        self._ready_event.set()

        logger.debug("Client transport connected to %s", self.address)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._send_loop(self._socket, self._send_queue))
                tg.create_task(self._recv_loop(self._socket))
        except* (asyncio.CancelledError, zmq.ZMQError):
            pass
        finally:
            self._cleanup_resources()

    def _create_socket(self) -> tuple[zmq.asyncio.Context, zmq.asyncio.Socket]:
        ctx = zmq.asyncio.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)

        if self.curve_server_key:
            socket.setsockopt(zmq.CURVE_SERVERKEY, self.curve_server_key)

            if not self.curve_public_key or not self.curve_secret_key:
                client_pub, client_sec = zmq.curve_keypair()
            else:
                client_pub = self.curve_public_key
                client_sec = self.curve_secret_key

            socket.setsockopt(zmq.CURVE_PUBLICKEY, client_pub)
            socket.setsockopt(zmq.CURVE_SECRETKEY, client_sec)

        socket.setsockopt(zmq.SNDBUF, 8 * 1024 * 1024)
        socket.setsockopt(zmq.RCVBUF, 8 * 1024 * 1024)
        socket.setsockopt(zmq.SNDHWM, 1024)
        socket.setsockopt(zmq.RCVHWM, 1024)
        socket.connect(self.address)
        return ctx, socket

    async def _send_loop(self, socket: zmq.asyncio.Socket, queue: asyncio.Queue[list[bytes] | None]) -> None:
        while self._running:
            msg = await queue.get()
            if msg is None or not self._running:
                break
            try:
                await socket.send_multipart(msg)
            except Exception:
                logger.exception("Failed to send message over DEALER socket")

    async def _recv_loop(self, socket: zmq.asyncio.Socket) -> None:
        while self._running:
            try:
                parts = await socket.recv_multipart()
            except (asyncio.CancelledError, zmq.ZMQError):
                break
            except Exception:
                logger.exception("Error in client receiver loop")
                break

            if parts and self._running:
                self._dispatch_response(parts)

    def _dispatch_response(self, parts: list[bytes]) -> None:
        req_id = int.from_bytes(parts[0], byteorder="big")

        if req_id == 0:
            if self.on_event and len(parts) >= 3 and parts[1] == bytes([StatusCode.OK]):
                try:
                    event = unpack_payload(parts[2], RemoteLogRecord | StreamOutputEvent)
                    self.on_event(event)
                except Exception:
                    logger.exception("Error decoding or handling server stream event")
            return

        with self._lock:
            fut = self._pending.pop(req_id, None)
        if fut and not fut.done():
            fut.set_result(parts[1:])

    def _cleanup_resources(self) -> None:
        self._ready_event.set()

        with self._lock:
            for pending_fut in self._pending.values():
                if not pending_fut.done():
                    pending_fut.set_exception(ConnectionResetError("Transport closed"))
            self._pending.clear()

        if self._socket:
            self._socket.close(linger=0)
            self._socket = None

        if self._ctx:
            self._ctx.destroy(linger=0)
            self._ctx = None

    def _send_message(self, req_id: int, cmd: Command, payload_bytes: bytes) -> None:
        if not self._started:
            raise TransportNotStartedError("Transport is not started")
        if not self._running or self._loop is None or self._send_queue is None:
            raise TransportNotConnectedError("Transport is not connected")

        parts = [req_id.to_bytes(4, byteorder="big"), bytes([cmd.value]), payload_bytes]
        if self.auth_token:
            parts.append(self.auth_token.encode("utf-8"))

        try:
            self._loop.call_soon_threadsafe(self._send_queue.put_nowait, parts)
        except RuntimeError:
            raise TransportClosedError("ClientTransport is closed")

    def _cancel_request(self, req_id: int) -> None:
        with self._lock:
            fut = self._pending.pop(req_id, None)
        if fut and not fut.done():
            fut.cancel()

        if self._running and self._loop and self._send_queue is not None:
            cancel_payload = pack_payload(CancelRequest(request_id=req_id))
            with contextlib.suppress(TransportError, RuntimeError):
                self._send_message(0, Command.CANCEL_REQUEST, cancel_payload)
