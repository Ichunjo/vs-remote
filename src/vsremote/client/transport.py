from __future__ import annotations

import contextlib
import os
import queue
import socket
import threading
from collections.abc import Callable
from logging import getLogger
from typing import Any, Self, overload

import zmq
from typing_extensions import TypeForm
from vsengine.futures import UnifiedFuture

from ..protocol.codec import pack_payload, unpack_payload
from ..protocol.constants import DEFAULT_ADDRESS, Command, Compression, StatusCode
from ..protocol.messages import (
    ClipInfo,
    FrameHeader,
    FrameRequest,
    LoadCodeRequest,
    LoadScriptRequest,
    OutputIndexRequest,
    OutputItem,
    ReloadRequest,
    RemoteLogRecord,
    ResponseEnvelope,
    StreamEvent,
    StreamOutputEvent,
    StreamSubscribeRequest,
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

        self._ctx: zmq.Context[zmq.Socket[bytes]] | None = None
        self._socket: zmq.Socket[bytes] | None = None
        self._waker_r: socket.socket | None = None
        self._waker_w: socket.socket | None = None

        self._send_queue = queue.Queue[list[bytes]]()
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

            # Wake worker thread immediately to terminate without waiting for poll timeout
            if self._waker_w:
                with contextlib.suppress(OSError):
                    self._waker_w.send(b"\x00")

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._thread = None

            self._started = False
            logger.debug("Client transport closed")

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
            return fut.reject(RuntimeError("Transport is not started")).map(to_response_envelope)

        if not self._running:
            return fut.reject(RuntimeError("ClientTransport is closed")).map(to_response_envelope)

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
            .map(lambda resp: resp.is_ok and resp.payload == b"PONG")
            .catch(lambda _: False)
        )

    def list_outputs(self) -> UnifiedFuture[list[OutputItem]]:
        """List all available VideoNode outputs on the server."""

        def parse(resp: ResponseEnvelope[list[OutputItem]]) -> list[OutputItem]:
            if not resp.is_ok:
                raise RuntimeError(f"Failed to list outputs: {resp.payload!r}")
            return resp.payload

        return self.send_request(Command.LIST_OUTPUTS, response_type=list[OutputItem]).map(parse)

    def load_script(self, script_path: str | os.PathLike[str], chdir: bool = True) -> UnifiedFuture[list[OutputItem]]:
        """Request the remote server to load or switch to a script file."""

        def parse(resp: ResponseEnvelope[list[OutputItem]]) -> list[OutputItem]:
            if not resp.is_ok:
                raise RuntimeError(f"Failed to load script {script_path}: {resp.payload!r}")
            return resp.payload

        return self.send_request(
            Command.LOAD_SCRIPT,
            LoadScriptRequest(os.fspath(script_path), chdir),
            response_type=list[OutputItem],
        ).map(parse)

    def load_code(self, code: str, filename: str = "<remote_code>") -> UnifiedFuture[list[OutputItem]]:
        """Request the remote server to execute Python/VapourSynth code."""

        def parse(resp: ResponseEnvelope[list[OutputItem]]) -> list[OutputItem]:
            if not resp.is_ok:
                raise RuntimeError(f"Failed to load code: {resp.payload!r}")
            return resp.payload

        return self.send_request(
            Command.LOAD_CODE,
            LoadCodeRequest(code, filename),
            response_type=list[OutputItem],
        ).map(parse)

    def reload(self, chdir: bool = True) -> UnifiedFuture[list[OutputItem]]:
        """Request the remote server to reload its current script file."""

        def parse(resp: ResponseEnvelope[list[OutputItem]]) -> list[OutputItem]:
            if not resp.is_ok:
                raise RuntimeError(f"Failed to reload script: {resp.payload!r}")
            return resp.payload

        return self.send_request(Command.RELOAD, ReloadRequest(chdir=chdir), response_type=list[OutputItem]).map(parse)

    def get_clip_info(self, output_index: int = 0) -> UnifiedFuture[ClipInfo]:
        """Fetch static metadata for the specified output index."""

        def parse(resp: ResponseEnvelope[ClipInfo]) -> ClipInfo:
            if not resp.is_ok:
                raise KeyError(f"Failed to retrieve clip info for output {output_index}: {resp.payload}")
            return resp.payload

        return self.send_request(
            Command.GET_CLIP_INFO,
            OutputIndexRequest(output_index),
            response_type=ClipInfo,
        ).map(parse)

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
            n: Frame number to fetch.
            compression: Desired compression mode.

        Returns:
            A UnifiedFuture resolving to (FrameHeader, list of compressed plane byte buffers).
        """
        payload = FrameRequest(output_index=output_index, n=n, compression=compression)

        def parse(resp: ResponseEnvelope[FrameHeader]) -> tuple[FrameHeader, list[bytes]]:
            return resp.payload, resp.extra_frames

        return self.send_request(Command.GET_FRAME, payload, response_type=FrameHeader).map(parse)

    def subscribe_stream(self, replay_history: bool = True) -> UnifiedFuture[bool]:
        """Subscribe to log and output streaming from the server."""
        return (
            self.send_request(Command.SUBSCRIBE_STREAM, StreamSubscribeRequest(replay_history=replay_history))
            .map(lambda resp: resp.is_ok)
            .catch(lambda _: False)
        )

    def unsubscribe_stream(self) -> UnifiedFuture[bool]:
        """Unsubscribe from log and output streaming from the server."""
        return self.send_request(Command.UNSUBSCRIBE_STREAM).map(lambda resp: resp.is_ok).catch(lambda _: False)

    def _start_worker_thread(self) -> None:
        """Start background worker thread running a native ZeroMQ poller loop."""
        self._ready_event.clear()
        self._thread = threading.Thread(target=self._worker, name="VSRemoteTransport", daemon=True)
        self._thread.start()

        if not self._ready_event.wait(timeout=5.0):
            raise TimeoutError("Timed out waiting for transport worker thread to initialize")

    def _worker(self) -> None:
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.DEALER)

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

        self._waker_r, self._waker_w = socket.socketpair()
        self._waker_r.setblocking(False)

        poller = zmq.Poller()
        poller.register(self._waker_r, zmq.POLLIN)
        self._running = True
        self._ready_event.set()

        logger.debug("Client transport connected to %s", self.address)

        pending_send: list[bytes] | None = None

        try:
            while self._running:
                # Flush queued outgoing messages
                while pending_send is not None or not self._send_queue.empty():
                    if pending_send is None:
                        try:
                            pending_send = self._send_queue.get_nowait()
                        except queue.Empty:
                            break
                    try:
                        self._socket.send_multipart(pending_send, flags=zmq.NOBLOCK)
                        pending_send = None
                    except zmq.Again:
                        break
                    except Exception:
                        logger.exception("Failed to send message over DEALER socket")
                        pending_send = None

                # Poll for incoming messages, write-readiness if pending send, or waker notifications
                poll_flags = zmq.POLLIN | (zmq.POLLOUT if pending_send is not None else 0)
                poller.register(self._socket, poll_flags)

                socks = dict(poller.poll(timeout=100))

                # Drain waker notification
                waker_fd = self._waker_r.fileno() if self._waker_r else None
                if waker_fd is not None and (self._waker_r in socks or waker_fd in socks):
                    with contextlib.suppress(OSError):
                        self._waker_r.recv(1024)

                if self._socket in socks and (socks[self._socket] & zmq.POLLIN):
                    while self._running:
                        try:
                            if not (parts := self._socket.recv_multipart(flags=zmq.NOBLOCK)):
                                break
                        except (zmq.Again, zmq.ZMQError):
                            break
                        except Exception:
                            logger.exception("Error in client receiver loop")
                            break

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
        finally:
            self._ready_event.set()

            with contextlib.suppress(Exception):
                self._socket.close(linger=0)
            with contextlib.suppress(Exception):
                self._ctx.term()
            with contextlib.suppress(Exception):
                self._waker_r.close()
            with contextlib.suppress(Exception):
                self._waker_w.close()

            self._socket = None
            self._ctx = None
            self._waker_r = None
            self._waker_w = None

            with self._lock:
                for pending_fut in self._pending.values():
                    if not pending_fut.done():
                        pending_fut.set_exception(ConnectionResetError("Transport closed"))
                self._pending.clear()

    def _send_message(self, req_id: int, cmd: Command, payload_bytes: bytes) -> None:
        """Enqueue message to be sent through the DEALER socket and wake worker immediately."""
        if not self._started:
            raise RuntimeError("Transport is not started")
        if not self._running:
            raise RuntimeError("Transport is not connected")

        parts = [req_id.to_bytes(4, byteorder="big"), bytes([cmd.value]), payload_bytes]
        if self.auth_token:
            parts.append(self.auth_token.encode("utf-8"))

        self._send_queue.put(parts)
        if self._waker_w:
            with contextlib.suppress(OSError):
                self._waker_w.send(b"\x00")
