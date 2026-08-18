from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from logging import getLogger
from typing import Any, Self, overload

import zmq
import zmq.asyncio
from typing_extensions import TypeForm
from vsengine.futures import UnifiedFuture

from ..exceptions import RemoteTimeoutError, TransportClosedError, TransportNotConnectedError, TransportNotStartedError
from ..protocol import (
    DEFAULT_ADDRESS,
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
    ) -> None:
        if not address.startswith(("tcp://", "ipc://", "inproc://")):
            address = f"tcp://{address}"

        self.address = address
        self.auth_token = auth_token
        self.curve_server_key = curve_server_key
        self.curve_public_key = curve_public_key
        self.curve_secret_key = curve_secret_key
        self.on_event = on_event
        self.subscribe_streams = subscribe_streams

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

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self.start()

    async def __aexit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def start(self) -> Self:
        """Start background worker thread (idempotent)."""
        with self._start_lock:
            if not self._started or not self._running:
                self._start_worker_thread()
                self._started = True
                if self.subscribe_streams:
                    self.subscribe_stream(replay_history=True)
        return self

    def close(self) -> None:
        """Close socket and stop background transport thread."""
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

        return fut.map(to_response_envelope)

    def ping(self) -> UnifiedFuture[bool]:
        """Check connection liveness to the remote server."""
        return (
            self.send_request(Command.PING)
            .map(lambda resp: resp.status == StatusCode.OK and resp.payload == b"PONG")
            .catch(lambda _: False)
        )

    def list_outputs(self) -> UnifiedFuture[list[OutputItem]]:
        """List all available VideoNode outputs on the server."""
        return self.send_request(
            Command.LIST_OUTPUTS,
            response_type=list[OutputItem],
        ).map(lambda r: r.raise_for_status("Failed to list outputs").payload)

    def load_script(self, script_path: str | os.PathLike[str], chdir: bool = True) -> UnifiedFuture[list[OutputItem]]:
        """Request the remote server to load or switch to a script file."""

        return self.send_request(
            Command.LOAD_SCRIPT,
            LoadScriptRequest(os.fspath(script_path), chdir),
            response_type=list[OutputItem],
        ).map(lambda r: r.raise_for_status(f"Failed to load script {script_path}").payload)

    def load_code(self, code: str, filename: str = "<remote_code>") -> UnifiedFuture[list[OutputItem]]:
        """Request the remote server to execute Python/VapourSynth code."""

        return self.send_request(
            Command.LOAD_CODE,
            LoadCodeRequest(code, filename),
            response_type=list[OutputItem],
        ).map(lambda r: r.raise_for_status("Failed to load code").payload)

    def reload(self, chdir: bool = True) -> UnifiedFuture[list[OutputItem]]:
        """Request the remote server to reload its current script file."""

        return self.send_request(
            Command.RELOAD,
            ReloadRequest(chdir=chdir),
            response_type=list[OutputItem],
        ).map(lambda r: r.raise_for_status("Failed to reload script").payload)

    def get_clip_info(self, output_index: int = 0) -> UnifiedFuture[ClipInfo]:
        """Fetch static metadata for the specified output index."""

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
            UnifiedFuture resolving to (FrameHeader, list of plane byte buffers).
        """
        req_payload = FrameRequest(output_index=output_index, n=n, compression=compression)

        def parse_frame_response(resp: ResponseEnvelope[bytes]) -> tuple[FrameHeader, list[bytes]]:
            header = unpack_payload(resp.payload_bytes, FrameHeader)
            return header, resp.extra_frames

        return self.send_request(Command.GET_FRAME, req_payload).map(parse_frame_response)

    def subscribe_stream(self, replay_history: bool = True) -> UnifiedFuture[bool]:
        """Subscribe to log records and stream events from the remote server."""
        req_payload = StreamSubscribeRequest(replay_history=replay_history)
        return (
            self.send_request(Command.SUBSCRIBE_STREAM, req_payload)
            .map(lambda r: r.status == StatusCode.OK)
            .catch(lambda _: False)
        )

    def unsubscribe_stream(self) -> UnifiedFuture[bool]:
        """Unsubscribe from log records and stream events."""
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
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.LINGER, 0)

        if self.curve_server_key:
            server_key_bytes = (
                self.curve_server_key.encode("ascii")
                if isinstance(self.curve_server_key, str)
                else self.curve_server_key
            )
            self._socket.setsockopt(zmq.CURVE_SERVERKEY, server_key_bytes)

            if not self.curve_public_key or not self.curve_secret_key:
                client_pub, client_sec = zmq.curve_keypair()
            else:
                client_pub = (
                    self.curve_public_key.encode("ascii")
                    if isinstance(self.curve_public_key, str)
                    else self.curve_public_key
                )
                client_sec = (
                    self.curve_secret_key.encode("ascii")
                    if isinstance(self.curve_secret_key, str)
                    else self.curve_secret_key
                )

            self._socket.setsockopt(zmq.CURVE_PUBLICKEY, client_pub)
            self._socket.setsockopt(zmq.CURVE_SECRETKEY, client_sec)

        self._socket.setsockopt(zmq.SNDBUF, 8 * 1024 * 1024)
        self._socket.setsockopt(zmq.RCVBUF, 8 * 1024 * 1024)
        self._socket.setsockopt(zmq.SNDHWM, 1024)
        self._socket.setsockopt(zmq.RCVHWM, 1024)
        self._socket.connect(self.address)

        self._send_queue = asyncio.Queue()
        self._running = True
        self._ready_event.set()

        logger.debug("Client transport connected to %s", self.address)

        async def send_loop() -> None:
            assert self._send_queue is not None
            assert self._socket is not None
            while self._running:
                msg = await self._send_queue.get()
                if msg is None or not self._running:
                    break
                try:
                    await self._socket.send_multipart(msg)
                except Exception:
                    logger.exception("Failed to send message over DEALER socket")

        async def recv_loop() -> None:
            assert self._socket is not None
            while self._running:
                try:
                    parts = await self._socket.recv_multipart()
                except (asyncio.CancelledError, zmq.ZMQError):
                    break
                except Exception:
                    logger.exception("Error in client receiver loop")
                    break

                if not parts or not self._running:
                    continue

                req_id = int.from_bytes(parts[0], byteorder="big")

                if req_id == 0:
                    if self.on_event and len(parts) >= 3 and parts[1] == bytes([StatusCode.OK]):
                        try:
                            event = unpack_payload(parts[2], RemoteLogRecord | StreamOutputEvent)
                            self.on_event(event)
                        except Exception:
                            logger.exception("Error decoding or handling server stream event")
                    continue

                with self._lock:
                    fut = self._pending.pop(req_id, None)
                if fut and not fut.done():
                    fut.set_result(parts[1:])

        try:
            await asyncio.gather(send_loop(), recv_loop())
        except asyncio.CancelledError:
            pass
        finally:
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
