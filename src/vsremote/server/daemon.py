from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import secrets
import threading
import traceback
import urllib.parse
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from typing import Any, overload

import vapoursynth as vs
import zmq
import zmq.asyncio
from vsengine.vpy import ExecutionError

from ..exceptions import TransportClosedError
from ..protocol import (
    DEFAULT_ADDRESS,
    CancelRequest,
    Command,
    Compression,
    FrameHeader,
    FrameRequest,
    LoadCodeRequest,
    LoadScriptRequest,
    OutputIndexRequest,
    ReloadRequest,
    RemoteErrorPayload,
    RequestEnvelope,
    StackFrame,
    StatusCode,
    StreamEvent,
    StreamSubscribeRequest,
    compress_plane,
    pack_payload,
    sanitize_props,
    unpack_payload,
    validate_curve_allowed_keys,
    validate_curve_key,
    z85_encode,
)
from ..utils import ensure_vsengine_loop
from .redirect import LogForwarder, StreamRedirector
from .runner import ScriptRunner

logger = getLogger(__name__)

_ZAP_STATUS_OK = b"200"
_ZAP_MSG_OK = b"OK"
_ZAP_USER_AUTH = b"authorized_client"
_ZAP_EMPTY = b""
_ZAP_STATUS_ERR = b"400"
_ZAP_MSG_UNAUTHORIZED = b"Unauthorized client key"


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
        curve_allowed_keys: Sequence[str | bytes] | None = None,
    ) -> None:
        self._ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._zap_socket: zmq.asyncio.Socket | None = None
        self._zap_task: asyncio.Task[None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._running = False
        self._send_lock: asyncio.Lock | None = None
        self._active_tasks = set[asyncio.Task[None]]()
        self._inflight_tasks = dict[tuple[bytes, int], asyncio.Task[None]]()
        self._subscribers = set[bytes]()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._log_handler: LogForwarder | None = None
        self._stdout_redirector: StreamRedirector | None = None
        self._stderr_redirector: StreamRedirector | None = None

        if curve_public_key is not None and curve_secret_key is None:
            raise ValueError("curve_public_key requires curve_secret_key to be specified on the server")

        if curve_allowed_keys is not None and curve_secret_key is None:
            raise ValueError("curve_allowed_keys requires curve_secret_key to be specified on the server")

        if not address.startswith(("tcp://", "ipc://", "inproc://")):
            address = f"tcp://{address}"

        self.runner = runner
        self.address = address
        self.compression: Compression = compression
        self.max_workers = max_workers
        self.allow_eval = allow_eval
        self.auth_token = auth_token
        self.curve_secret_key = validate_curve_key(curve_secret_key, "curve_secret_key")
        self.curve_public_key = validate_curve_key(curve_public_key, "curve_public_key")
        self.curve_allowed_keys = validate_curve_allowed_keys(curve_allowed_keys)

    async def start(self, ready_event: threading.Event | asyncio.Event | None = None) -> None:
        """Start the async server loop."""
        loop = asyncio.get_running_loop()
        ensure_vsengine_loop(loop)
        self._loop = loop

        self._ctx = zmq.asyncio.Context()

        if self.curve_secret_key and self.curve_allowed_keys:
            self._zap_socket = self._ctx.socket(zmq.REP)
            self._zap_socket.setsockopt(zmq.LINGER, 0)
            self._zap_socket.bind("inproc://zeromq.zap.01")
            self._zap_task = asyncio.create_task(self._zap_handler_loop(), name="VSRemoteZAP")

        self._socket = self._ctx.socket(zmq.ROUTER)

        if self.curve_secret_key:
            self._socket.setsockopt(zmq.CURVE_SERVER, 1)
            self._socket.setsockopt(zmq.CURVE_SECRETKEY, self.curve_secret_key)
            if self.curve_public_key:
                self._socket.setsockopt(zmq.CURVE_PUBLICKEY, self.curve_public_key)

            if self.curve_allowed_keys:
                self._socket.setsockopt(zmq.ZAP_DOMAIN, b"vsremote")

        self._socket.setsockopt(zmq.SNDBUF, 8 * 1024 * 1024)
        self._socket.setsockopt(zmq.RCVBUF, 8 * 1024 * 1024)
        self._socket.setsockopt(zmq.SNDHWM, 1024)
        self._socket.setsockopt(zmq.RCVHWM, 1024)
        self._socket.bind(self.address)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
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
            "Server started on %s (default compression: %s, allow_eval: %s, auth: %s, curve: %s, client_auth: %s)",
            self.address,
            self.compression,
            self.allow_eval,
            bool(self.auth_token),
            bool(self.curve_secret_key),
            bool(self.curve_allowed_keys),
        )

        if not _is_loopback_address(self.address):
            if not (self.auth_token or self.curve_secret_key):
                if self.allow_eval:
                    logger.warning("Server bound to %s with allow_eval=True and NO auth/encryption", self.address)
                    logger.warning("Anyone on the network can execute arbitrary code.")
                else:
                    logger.warning("Server bound to %s without auth or encryption.", self.address)
            elif self.allow_eval:
                logger.warning(
                    "Server bound to %s with allow_eval=True. "  # no fmt
                    "Authenticated clients can execute arbitrary code.",
                    self.address,
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

        if self._zap_task:
            self._zap_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, zmq.ZMQError):
                await self._zap_task
            self._zap_task = None

        if self._zap_socket:
            self._zap_socket.close(linger=0)
            self._zap_socket = None

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

    async def _zap_handler_loop(self) -> None:
        """Handle ZeroMQ Authentication Protocol (ZAP) requests to whitelist Curve client keys."""
        if not self._zap_socket:
            return

        while self._running:
            try:
                msg = await self._zap_socket.recv_multipart()
            except (asyncio.CancelledError, zmq.ZMQError):
                break

            version = msg[0] if len(msg) >= 1 else b"1.0"
            req_id = msg[1] if len(msg) >= 2 else b"0"

            try:
                if len(msg) < 7:
                    await self._zap_socket.send_multipart(
                        [version, req_id, _ZAP_STATUS_ERR, b"Invalid ZAP request", _ZAP_EMPTY, _ZAP_EMPTY]
                    )
                    continue

                _, address, _, mechanism, client_key = msg[2:7]
                addr_str = address.decode("utf-8", errors="replace")

                if mechanism == b"CURVE" and client_key in self.curve_allowed_keys:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "ZAP: Authorized CURVE client %s (key: %s)",
                            addr_str,
                            z85_encode(client_key).decode("ascii"),
                        )
                    await self._zap_socket.send_multipart(
                        [version, req_id, _ZAP_STATUS_OK, _ZAP_MSG_OK, _ZAP_USER_AUTH, _ZAP_EMPTY]
                    )
                else:
                    logger.warning(
                        "ZAP: Unauthorized %s client connection from %s (key %s not in allowed keys)",
                        mechanism.decode("ascii", errors="replace"),
                        addr_str,
                        z85_encode(client_key).decode("ascii", errors="replace"),
                    )
                    await self._zap_socket.send_multipart(
                        [version, req_id, _ZAP_STATUS_ERR, _ZAP_MSG_UNAUTHORIZED, _ZAP_EMPTY, _ZAP_EMPTY]
                    )
            except (asyncio.CancelledError, zmq.ZMQError):
                break
            except Exception as exc:
                logger.error("Error handling ZAP authentication request: %s", exc)
                with contextlib.suppress(Exception):
                    await self._zap_socket.send_multipart(
                        [version, req_id, _ZAP_STATUS_ERR, b"Internal authentication error", _ZAP_EMPTY, _ZAP_EMPTY]
                    )

    async def _handle_request(self, msg: list[bytes]) -> None:
        """Process an incoming ROUTER message and dispatch response."""
        if not self._socket:
            return None

        try:
            req = RequestEnvelope.from_frames(msg)
        except ValueError as err:
            logger.warning("Received malformed message: %s", err)
            if len(msg) >= 2:
                await self._send_error(RequestEnvelope.from_data(msg[0], msg[1]), StatusCode.INVALID_COMMAND, str(err))
            return None

        # Check auth token if authentication is enabled
        if self.auth_token is not None and (
            not req.auth_token or not secrets.compare_digest(req.auth_token, self.auth_token)
        ):
            logger.warning("Unauthorized request from identity %r (invalid or missing auth token)", req.identity)
            return await self._send_error(req, StatusCode.UNAUTHORIZED, "Unauthorized: invalid or missing auth token")

        key = req.identity, req.request_id
        if (current_task := asyncio.current_task()) is not None:
            self._inflight_tasks[key] = current_task

        try:
            await self._dispatch_request(req)
        finally:
            self._inflight_tasks.pop(key, None)

    async def _dispatch_request(self, req: RequestEnvelope) -> None:
        match req.command:
            case Command.CANCEL_REQUEST:
                try:
                    cancel_payload = unpack_payload(req.payload_bytes, CancelRequest)
                    if task := self._inflight_tasks.get((req.identity, cancel_payload.request_id)):
                        task.cancel()
                except Exception as e:
                    logger.debug("Failed to cancel request: %s", e)

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
                    await self._send_error(req, StatusCode.NOT_FOUND, str(e))
                except Exception as e:
                    await self._send_error(req, StatusCode.INVALID_PAYLOAD, f"Invalid OutputIndexRequest: {e}")

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

            case Command.LOAD_CODE if self.allow_eval:
                await self._dispatch_executor(
                    req,
                    LoadCodeRequest,
                    lambda p: self.runner.load_code(p.code, filename=p.filename),
                    "Failed to load code",
                )
            case Command.LOAD_CODE if not self.allow_eval:
                await self._send_error(
                    req,
                    StatusCode.PERMISSION_DENIED,
                    "Dynamic code evaluation is disabled on this server (allow_eval=False)",
                )

            case Command.LOAD_SCRIPT if self.allow_eval:
                await self._dispatch_executor(
                    req,
                    LoadScriptRequest,
                    lambda p: self.runner.load_script(p.script_path, chdir=p.chdir),
                    "Failed to load script",
                )
            case Command.LOAD_SCRIPT if not self.allow_eval:
                await self._send_error(
                    req,
                    StatusCode.PERMISSION_DENIED,
                    "Dynamic script loading is disabled on this server (allow_eval=False)",
                )

    async def _handle_get_frame(self, req: RequestEnvelope) -> None:
        if not self._socket:
            return None

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
            return await self._send_reply(req, header.status, pack_payload(header))

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
            return await self._send_reply(req, header.status, pack_payload(header))

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

        except asyncio.CancelledError:
            logger.debug("Frame request %d for output %d was cancelled", n, output_index)
            raise
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
            return await self._send_error(req, StatusCode.INVALID_PAYLOAD, f"Invalid {payload_type.__name__}: {e}")

        loop = asyncio.get_running_loop()

        try:
            outputs = await loop.run_in_executor(self._executor, action, payload)
            await self._send_reply(req, StatusCode.OK, pack_payload(outputs))
        except ExecutionError as e:
            logger.debug("%s: %s", error_context, e)
            await self._send_error(req, StatusCode.ERROR, _build_error_payload(e))
            await loop.run_in_executor(self._executor, self.runner.teardown_environment)
        except Exception as e:
            logger.exception(error_context)
            await self._send_error(req, StatusCode.ERROR, _build_error_payload(e))
            await loop.run_in_executor(self._executor, self.runner.teardown_environment)

    async def _send_reply(
        self,
        req: RequestEnvelope,
        status: StatusCode,
        payload_bytes: bytes,
        extra_frames: Sequence[bytes | memoryview] = (),
    ) -> None:
        await self._send_multipart([req.identity, req.request_id_bytes, bytes([status]), payload_bytes, *extra_frames])

    @overload
    async def _send_error(self, req: RequestEnvelope, status: StatusCode, message: str, /) -> None: ...
    @overload
    async def _send_error(self, req: RequestEnvelope, status: StatusCode, payload: RemoteErrorPayload, /) -> None: ...
    async def _send_error(self, req: RequestEnvelope, status: StatusCode, v: str | RemoteErrorPayload, /) -> None:
        payload = RemoteErrorPayload(v, status.name.replace("_", " ").title(), v) if isinstance(v, str) else v
        await self._send_multipart([req.identity, req.request_id_bytes, bytes([status]), pack_payload(payload)])

    async def _send_multipart(self, parts: Sequence[bytes | memoryview]) -> None:
        if not self._socket or not self._send_lock:
            raise TransportClosedError("Server socket or send lock is not initialized")

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


def _is_loopback_address(address: str) -> bool:
    if address.startswith(("ipc://", "inproc://")):
        return True

    if not address.startswith("tcp://"):
        address = f"tcp://{address}"

    hostname = urllib.parse.urlsplit(address).hostname

    if not hostname:
        return False

    if hostname.lower() in ("localhost", "ip6-localhost", "ip6-loopback"):
        return True

    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _build_error_payload(e: ExecutionError | Exception) -> RemoteErrorPayload:
    orig = e.parent_error if isinstance(e, ExecutionError) else e

    tb = traceback.TracebackException.from_exception(orig)
    frames = [StackFrame.from_summary(f) for f in tb.stack if not f.filename.startswith("src/cython/")]

    code_line: str | None = None
    if isinstance(orig, SyntaxError) and orig.filename is not None:
        filename = orig.filename
        lineno = orig.lineno
        code_line = orig.text.strip() if orig.text else None
        exc_msg = orig.msg or str(orig)
    else:
        filename = getattr(orig, "filename", None)
        lineno = getattr(orig, "lineno", None)
        exc_msg = str(orig)

        for f in reversed(frames):
            norm = f.filename.lower().replace("\\", "/")
            if not any(
                m in norm
                for m in ("site-packages/", "/lib/", ".venv/", "venv/", "lib/python", "vsengine/", "vsremote/")
            ):
                filename = f.filename
                lineno = f.lineno
                code_line = f.code
                break

        if lineno is None and frames:
            filename = frames[-1].filename
            lineno = frames[-1].lineno
            code_line = frames[-1].code

    exc_type = type(orig).__name__
    err_str = f"{exc_type}: {exc_msg}" if exc_msg else exc_type
    formatted_tb = "".join(tb.format())

    return RemoteErrorPayload(err_str, exc_type, exc_msg, filename, lineno, code_line, formatted_tb, frames)
