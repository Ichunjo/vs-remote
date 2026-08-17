from __future__ import annotations

import asyncio
import logging
import secrets
import threading
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from typing import Any

import vapoursynth as vs
import zmq.asyncio

from ..protocol import (
    DEFAULT_ADDRESS,
    Command,
    Compression,
    FrameHeader,
    FrameRequest,
    LoadCodeRequest,
    LoadScriptRequest,
    OutputIndexRequest,
    ReloadRequest,
    RequestEnvelope,
    StatusCode,
    StreamEvent,
    StreamSubscribeRequest,
    compress_plane,
    pack_payload,
    sanitize_props,
    unpack_payload,
)
from ..utils import ensure_vsengine_loop
from .redirect import LogForwarder, StreamRedirector
from .runner import ScriptRunner

logger = getLogger(__name__)


class ServerDaemon:
    """Asynchronous ZeroMQ ROUTER server for streaming VapourSynth video frames and output logs."""

    def __init__(
        self,
        runner: ScriptRunner,
        address: str = DEFAULT_ADDRESS,
        compression: Compression = "zstd",
        max_workers: int | None = None,
        *,
        allow_eval: bool = False,
        auth_token: str | None = None,
        curve_secret_key: str | bytes | None = None,
        curve_public_key: str | bytes | None = None,
    ) -> None:
        self.runner = runner
        if not address.startswith(("tcp://", "ipc://", "inproc://")):
            address = f"tcp://{address}"
        self.address = address

        self.compression: Compression = compression
        self.max_workers = max_workers
        self.allow_eval = allow_eval
        self.auth_token = auth_token
        self.curve_secret_key = curve_secret_key
        self.curve_public_key = curve_public_key

        self._ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._running = False
        self._send_lock: asyncio.Lock | None = None
        self._active_tasks = set[asyncio.Task[None]]()
        self._subscribers = set[bytes]()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._log_handler: LogForwarder | None = None
        self._stdout_redirector: StreamRedirector | None = None
        self._stderr_redirector: StreamRedirector | None = None

    async def start(self, ready_event: threading.Event | asyncio.Event | None = None) -> None:
        """Start the async server loop."""
        loop = asyncio.get_running_loop()
        ensure_vsengine_loop(loop)
        self._loop = loop

        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.ROUTER)

        if self.curve_secret_key:
            self._socket.setsockopt(zmq.CURVE_SERVER, 1)
            sec_bytes = (
                self.curve_secret_key.encode("ascii")
                if isinstance(self.curve_secret_key, str)
                else self.curve_secret_key
            )
            self._socket.setsockopt(zmq.CURVE_SECRETKEY, sec_bytes)
            if self.curve_public_key:
                pub_bytes = (
                    self.curve_public_key.encode("ascii")
                    if isinstance(self.curve_public_key, str)
                    else self.curve_public_key
                )
                self._socket.setsockopt(zmq.CURVE_PUBLICKEY, pub_bytes)

        self._socket.setsockopt(zmq.SNDBUF, 8 * 1024 * 1024)
        self._socket.setsockopt(zmq.RCVBUF, 8 * 1024 * 1024)
        self._socket.setsockopt(zmq.SNDHWM, 1024)
        self._socket.setsockopt(zmq.RCVHWM, 1024)
        self._socket.bind(self.address)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers or 4)
        self._send_lock = asyncio.Lock()
        self._running = True

        # Install log forwarder and stream redirectors
        self._log_handler = LogForwarder(self._on_event_from_thread)
        logging.getLogger().addHandler(self._log_handler)

        self._stdout_redirector = StreamRedirector("stdout", self._on_event_from_thread)
        self._stderr_redirector = StreamRedirector("stderr", self._on_event_from_thread)
        self._stdout_redirector.install()
        self._stderr_redirector.install()

        logger.info(
            "Server started on %s (default compression: %s, allow_eval: %s, auth: %s, curve: %s)",
            self.address,
            self.compression,
            self.allow_eval,
            bool(self.auth_token),
            bool(self.curve_secret_key),
        )

        if ready_event is not None:
            ready_event.set()

        try:
            while self._running:
                msg = await self._socket.recv_multipart()
                task = asyncio.create_task(self._handle_request(msg))
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)
        except (asyncio.CancelledError, zmq.ZMQError):
            pass
        finally:
            await self.stop()

    def _on_event_from_thread(self, event: StreamEvent) -> None:
        """Thread-safe dispatch of stream and log events from any worker thread."""
        if self._running and self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._broadcast_event, event)

    def _broadcast_event(self, event: StreamEvent) -> None:
        """Broadcast event to all subscribed client DEALER identities."""
        if not self._running or not self._subscribers or not self._socket:
            return

        payload = pack_payload(event)
        zero_req_id = (0).to_bytes(4, byteorder="big")
        status_ok = bytes([StatusCode.OK])
        subscribers = list(self._subscribers)

        async def send_all() -> None:
            for identity in subscribers:
                try:
                    await self._send_multipart([identity, zero_req_id, status_ok, payload])
                except Exception:
                    logger.debug("Failed to send broadcast event to subscriber %r", identity)

        task = asyncio.create_task(send_all())
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def stop(self) -> None:
        """Stop server and clean up active connections and resources."""
        if not self._running:
            return

        self._running = False
        logger.info("Shutting down server...")

        # Uninstall redirectors and handlers
        if self._stdout_redirector:
            self._stdout_redirector.uninstall()
            self._stdout_redirector = None
        if self._stderr_redirector:
            self._stderr_redirector.uninstall()
            self._stderr_redirector = None
        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

        self._subscribers.clear()

        # Cancel any in-flight request tasks
        for task in list(self._active_tasks):
            task.cancel()

        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()

        if self._socket:
            self._socket.close(linger=0)
            self._socket = None

        if self._ctx:
            self._ctx.term()
            self._ctx = None

        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

        self.runner.close()
        logger.info("Server shutdown complete")

    async def _handle_request(self, msg: list[bytes]) -> None:
        """Process an incoming ROUTER message and dispatch response."""
        if not self._socket:
            return

        try:
            req = RequestEnvelope.from_frames(msg)
        except ValueError as err:
            logger.warning("Received malformed message: %s", err)
            if len(msg) >= 2:
                await self._send_error(msg[0], msg[1], StatusCode.INVALID_COMMAND, str(err))
            return

        # Check auth token if authentication is enabled
        if self.auth_token is not None and (
            not req.auth_token or not secrets.compare_digest(req.auth_token, self.auth_token)
        ):
            logger.warning("Unauthorized request from identity %r (invalid or missing auth token)", req.identity)
            await self._send_error(
                req.identity,
                req.request_id_bytes,
                StatusCode.UNAUTHORIZED,
                "Unauthorized: invalid or missing auth token",
            )
            return

        match req.command:
            case Command.PING:
                await self._send_reply(req, StatusCode.OK, b"PONG")

            case Command.LIST_OUTPUTS:
                outputs = self.runner.list_outputs()
                await self._send_reply(req, StatusCode.OK, pack_payload(outputs))

            case Command.GET_CLIP_INFO:
                try:
                    payload = unpack_payload(req.payload_bytes, OutputIndexRequest)
                    info = self.runner.get_clip_info(payload.output_index)
                    await self._send_reply(req, StatusCode.OK, pack_payload(info))
                except KeyError as e:
                    await self._send_reply_error(req, StatusCode.NOT_FOUND, str(e))
                except Exception as e:
                    await self._send_reply_error(req, StatusCode.INVALID_PAYLOAD, f"Invalid OutputIndexRequest: {e}")

            case Command.GET_FRAME:
                await self._handle_get_frame(req)

            case Command.SUBSCRIBE_STREAM:
                self._subscribers.add(req.identity)
                await self._send_reply(req, StatusCode.OK, b"OK")

                # Replay historical startup logs if requested
                if unpack_payload(req.payload_bytes, StreamSubscribeRequest).replay_history:
                    zero_req_id = (0).to_bytes(4, byteorder="big")
                    status_ok = bytes([StatusCode.OK])
                    for event in self.runner.startup_events:
                        await self._send_multipart([req.identity, zero_req_id, status_ok, pack_payload(event)])

            case Command.UNSUBSCRIBE_STREAM:
                self._subscribers.discard(req.identity)
                await self._send_reply(req, StatusCode.OK, b"OK")

            case Command.CLOSE:
                self._subscribers.discard(req.identity)
                await self._send_reply(req, StatusCode.OK, b"BYE")

            case Command.RELOAD:
                await self._dispatch_executor(
                    req,
                    ReloadRequest,
                    lambda p: self.runner.reload(chdir=p.chdir),
                    "Failed to reload script",
                )

            case Command.LOAD_CODE:
                if self.allow_eval:
                    await self._dispatch_executor(
                        req,
                        LoadCodeRequest,
                        lambda p: self.runner.load_code(p.code, filename=p.filename),
                        "Failed to load code",
                    )
                else:
                    await self._send_reply_error(
                        req,
                        StatusCode.PERMISSION_DENIED,
                        "Dynamic code evaluation is disabled on this server (allow_eval=False)",
                    )

            case Command.LOAD_SCRIPT:
                if self.allow_eval:
                    await self._dispatch_executor(
                        req,
                        LoadScriptRequest,
                        lambda p: self.runner.load_script(p.script_path, chdir=p.chdir),
                        "Failed to load script",
                    )
                else:
                    await self._send_reply_error(
                        req,
                        StatusCode.PERMISSION_DENIED,
                        "Dynamic script loading is disabled on this server (allow_eval=False)",
                    )
                    return

    async def _handle_get_frame(self, req: RequestEnvelope) -> None:
        """Handle async frame rendering and compression."""
        if not self._socket:
            return

        request_id = req.request_id

        try:
            frame_req = unpack_payload(req.payload_bytes, FrameRequest)
        except Exception as e:
            header = FrameHeader(
                status=StatusCode.INVALID_PAYLOAD,
                request_id=req.request_id,
                n=0,
                output_index=0,
                compression=self.compression,
                plane_sizes=[],
                props={},
                error_message=f"Failed to decode frame request payload: {e}",
            )
            await self._send_reply(req, header.status, pack_payload(header))
            return

        output_index = frame_req.output_index
        n = frame_req.n
        compression_str = frame_req.compression

        try:
            clip = self.runner.get_clip(output_index)
        except KeyError as e:
            header = FrameHeader(
                status=StatusCode.NOT_FOUND,
                request_id=request_id,
                n=n,
                output_index=output_index,
                compression=compression_str,
                plane_sizes=[],
                props={},
                error_message=str(e),
            )
            await self._send_reply(req, header.status, pack_payload(header))
            return

        if n < 0 or n >= clip.num_frames:
            header = FrameHeader(
                status=StatusCode.ERROR,
                request_id=request_id,
                n=n,
                output_index=output_index,
                compression=compression_str,
                plane_sizes=[],
                props={},
                error_message=f"Frame index out of bounds: {n} (num_frames: {clip.num_frames})",
            )
            await self._send_reply(req, header.status, pack_payload(header))
            return

        try:
            with self.runner.environment.use():
                future = clip.get_frame_async(n)

            with await asyncio.wrap_future(future) as frame:
                clean_props = sanitize_props(frame.props)
                planes = await asyncio.get_running_loop().run_in_executor(
                    self._executor, _extract_and_compress_planes, frame, compression_str
                )

                header = FrameHeader(
                    status=StatusCode.OK,
                    request_id=request_id,
                    n=n,
                    output_index=output_index,
                    compression=compression_str,
                    plane_sizes=[p.nbytes if isinstance(p, memoryview) else len(p) for p in planes],
                    props=clean_props,
                )

                # Stream response back to client
                await self._send_reply(req, StatusCode.OK, pack_payload(header), planes)

        except Exception as e:
            logger.exception("Failed to render frame %d for output %d", n, output_index)
            tb = traceback.format_exc()
            header = FrameHeader(
                status=StatusCode.ERROR,
                request_id=request_id,
                n=n,
                output_index=output_index,
                compression=compression_str,
                plane_sizes=[],
                props={},
                error_message=f"{e}\n\n[Remote Traceback]:\n{tb}".strip(),
            )
            await self._send_reply(req, header.status, pack_payload(header))

    async def _dispatch_executor[T](
        self,
        req: RequestEnvelope,
        payload_type: type[T],
        action: Callable[[T], Any],
        error_context: str,
    ) -> None:
        """Unpack payload, execute blocking action on thread pool executor, and respond."""
        try:
            payload = unpack_payload(req.payload_bytes, payload_type)
        except Exception as e:
            await self._send_reply_error(req, StatusCode.INVALID_PAYLOAD, f"Invalid {payload_type.__name__}: {e}")
            return

        try:
            outputs = await asyncio.get_running_loop().run_in_executor(self._executor, lambda: action(payload))
            await self._send_reply(req, StatusCode.OK, pack_payload(outputs))
        except Exception as e:
            logger.exception(error_context)
            tb = traceback.format_exc()
            err_str = f"{error_context}: {e}\n\n[Remote Traceback]:\n{tb}".strip()
            await self._send_reply_error(req, StatusCode.ERROR, err_str)

    async def _send_reply(
        self,
        req: RequestEnvelope,
        status: StatusCode,
        payload_bytes: bytes,
        extra_frames: Sequence[bytes | memoryview] = (),
    ) -> None:
        await self._send_response(req.identity, req.request_id_bytes, status, payload_bytes, extra_frames)

    async def _send_response(
        self,
        identity: bytes,
        request_id_bytes: bytes,
        status: StatusCode,
        payload_bytes: bytes,
        extra_frames: Sequence[bytes | memoryview] = (),
    ) -> None:
        await self._send_multipart([identity, request_id_bytes, bytes([status]), payload_bytes, *extra_frames])

    async def _send_reply_error(self, req: RequestEnvelope, status: StatusCode, message: str) -> None:
        await self._send_error(req.identity, req.request_id_bytes, status, message)

    async def _send_error(self, identity: bytes, request_id_bytes: bytes, status: StatusCode, message: str) -> None:
        await self._send_multipart([identity, request_id_bytes, bytes([status]), pack_payload({"error": message})])

    async def _send_multipart(self, parts: Sequence[bytes | memoryview]) -> None:
        if not self._socket or not self._send_lock:
            raise RuntimeError

        async with self._send_lock:
            await self._socket.send_multipart(parts)


def _extract_and_compress_planes(frame: vs.VideoFrame, compression: Compression) -> list[bytes | memoryview]:
    planes = list[bytes | memoryview]()
    num_planes = frame.format.num_planes

    for p in range(num_planes):
        plane = frame[p]
        if compression == "none":
            planes.append(plane.cast("B") if plane.c_contiguous else plane.tobytes())
        else:
            planes.append(compress_plane(plane, compression))

    return planes
