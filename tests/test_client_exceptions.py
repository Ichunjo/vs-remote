from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import vapoursynth as vs
import zmq

from vsremote.client import ClientTransport, RemoteClient, source
from vsremote.client.client import create_remote_vnode
from vsremote.client.transport import RequestTracker
from vsremote.exceptions import (
    MalformedMessageError,
    RemoteAuthenticationError,
    RemoteCommandError,
    RemoteExecutionError,
    RemoteNotFoundError,
    RemotePayloadError,
    RemotePermissionError,
    RemoteTimeoutError,
    TransportClosedError,
    TransportError,
    TransportNotConnectedError,
    TransportNotStartedError,
    UnknownStatusCodeError,
    UnsupportedFormatError,
)
from vsremote.protocol import ClipInfo, Command, ResponseEnvelope, StatusCode
from vsremote.utils import SafeUnifiedFuture

if TYPE_CHECKING:
    from conftest import ServerFactory


# Transport Lifecycle & Connection Error States
def test_transport_start_worker_timeout_raises_remote_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a worker thread startup timeout in ClientTransport.start raises RemoteTimeoutError."""
    trans = ClientTransport("tcp://127.0.0.1:5555", startup_timeout=0.05)
    # Simulate a slow worker thread initialization on this instance
    monkeypatch.setattr(trans, "_worker", lambda: time.sleep(0.2))

    with pytest.raises(
        RemoteTimeoutError,
        match="Timed out waiting for transport worker thread to initialize",
    ) as exc_info:
        trans.start()

    assert isinstance(exc_info.value, TimeoutError)
    assert isinstance(exc_info.value, TransportError)
    assert not trans.is_started
    assert not trans.is_running
    assert trans.is_closed


def test_remote_client_start_propagates_remote_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that RemoteClient.start() propagates RemoteTimeoutError when transport startup times out."""
    client = RemoteClient("tcp://127.0.0.1:5555", startup_timeout=0.05)
    monkeypatch.setattr(client.transport, "_worker", lambda: time.sleep(0.2))

    with pytest.raises(RemoteTimeoutError, match="Timed out waiting for transport worker thread to initialize"):
        client.start()


def test_transport_not_started_error_on_send_and_requests() -> None:
    """Verify that operations on unstarted ClientTransport fail with TransportNotStartedError."""
    trans = ClientTransport("tcp://127.0.0.1:5555")
    assert not trans.is_started
    assert not trans.is_closed

    # send_request returns rejected future
    with pytest.raises(TransportNotStartedError, match="Transport is not started"):
        trans.send_request(Command.PING).result()

    # Typed methods return rejected futures
    with pytest.raises(TransportNotStartedError):
        trans.list_outputs().result()

    with pytest.raises(TransportNotStartedError):
        trans.get_stats().result()

    with pytest.raises(TransportNotStartedError):
        trans.get_clip_info(0).result()

    with pytest.raises(TransportNotStartedError):
        trans.request_frame(0, 0).result()


def test_remote_client_not_started_raises_transport_not_started_error() -> None:
    """Verify that RemoteClient methods on unstarted client raise TransportNotStartedError."""
    client = RemoteClient("tcp://127.0.0.1:5555")
    assert not client.is_started
    assert not client.is_closed

    with pytest.raises(TransportNotStartedError):
        client.get_output(0)

    with pytest.raises(TransportNotStartedError):
        client.get_outputs()


def test_transport_closed_error_states() -> None:
    """Verify that operations on closed transport raise TransportClosedError."""
    # 1. Unstarted transport closed
    trans = ClientTransport("tcp://127.0.0.1:5555")
    trans.close()
    assert trans.is_closed

    with pytest.raises(TransportClosedError, match="Cannot start a closed ClientTransport"):
        trans.start()

    with pytest.raises(TransportClosedError, match="ClientTransport is closed"):
        trans.send_request(Command.PING).result()

    with pytest.raises(TransportClosedError):
        trans.list_outputs().result()

    with pytest.raises(TransportClosedError):
        trans.get_stats().result()

    with pytest.raises(TransportClosedError):
        trans.get_clip_info(0).result()

    with pytest.raises(TransportClosedError):
        trans.request_frame(0, 0).result()

    # 2. Started transport closed
    trans2 = ClientTransport("tcp://127.0.0.1:5555").start()
    trans2.close()
    assert trans2.is_closed

    with pytest.raises(TransportClosedError, match="ClientTransport is closed"):
        trans2.send_request(Command.PING).result()


def test_transport_event_loop_closed_race(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that an event loop closure race during send_message raises TransportClosedError."""

    def raise_loop_closed(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Event loop is closed")

    with ClientTransport("tcp://127.0.0.1:5555") as trans:
        monkeypatch.setattr(trans._loop, "call_soon_threadsafe", raise_loop_closed)

        with pytest.raises(TransportClosedError, match="ClientTransport is closed"):
            trans.send_request(Command.PING).result()


def test_transport_not_connected_error_on_worker_exit() -> None:
    """Verify that unexpected worker thread termination causes TransportNotConnectedError."""
    with ClientTransport("tcp://127.0.0.1:5555") as trans:
        assert trans._loop is not None
        assert trans._thread is not None
        trans._loop.call_soon_threadsafe(trans._loop.stop)
        trans._thread.join(timeout=1.0)

        assert trans.is_started
        assert not trans.is_running
        assert not trans.is_closed

        with pytest.raises(TransportNotConnectedError, match="Transport is not connected"):
            trans.send_request(Command.PING).result()


def test_remote_client_closed_raises_transport_closed_error() -> None:
    """Verify that RemoteClient methods on closed client raise TransportClosedError."""
    # Unstarted then closed
    client = RemoteClient("tcp://127.0.0.1:5555")
    client.close()
    assert client.is_closed

    with pytest.raises(TransportClosedError):
        client.get_output(0)

    with pytest.raises(TransportClosedError):
        client.get_outputs()

    with pytest.raises(TransportClosedError, match="Cannot start a closed ClientTransport"):
        client.start()

    # Started then closed
    client2 = RemoteClient("tcp://127.0.0.1:5555")
    client2.start()
    client2.close()
    assert client2.is_closed

    with pytest.raises(TransportClosedError):
        client2.get_output(0)

    with pytest.raises(TransportClosedError):
        client2.get_outputs()


def test_request_tracker_close_rejects_pending_with_transport_closed_error() -> None:
    """Verify that closing RequestTracker rejects all pending entries with TransportClosedError."""
    tracker = RequestTracker()
    _, fut1 = tracker.allocate(bytes)
    _, fut2 = tracker.allocate(bytes)

    tracker.close()

    with pytest.raises(TransportClosedError, match="ClientTransport is closed"):
        fut1.result()

    with pytest.raises(TransportClosedError, match="ClientTransport is closed"):
        fut2.result()


# Handshake and Operation Timeout Errors (RemoteTimeoutError) - Real Network
def test_create_remote_vnode_handshake_timeout_raises_remote_timeout_error(port: int) -> None:
    """Verify that get_output/create_remote_vnode raises RemoteTimeoutError if clip info times out over network."""
    # Start a raw ROUTER socket that accepts connection but drops all requests (simulating dead/hung server)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(f"tcp://127.0.0.1:{port}")

    try:
        with ClientTransport(f"tcp://127.0.0.1:{port}", startup_timeout=1.0) as trans:
            with pytest.raises(RemoteTimeoutError, match="Timed out fetching clip info for output 0") as exc_info:
                create_remote_vnode(trans, output_index=0, compression="zstd", timeout=0.05)

            assert isinstance(exc_info.value, TimeoutError)
            assert isinstance(exc_info.value, TransportError)
    finally:
        sock.close(linger=0)
        ctx.term()


def test_client_get_outputs_timeout_raises_remote_timeout_error(port: int) -> None:
    """Verify that RemoteClient.get_outputs raises RemoteTimeoutError if list_outputs times out over network."""

    with zmq.Context() as ctx, ctx.socket(zmq.ROUTER) as sock:
        sock.bind(f"tcp://127.0.0.1:{port}")

        with RemoteClient(f"tcp://127.0.0.1:{port}", startup_timeout=1.0) as client:
            with pytest.raises(RemoteTimeoutError, match="Timed out listing outputs from remote server") as exc_info:
                client.get_outputs(timeout=0.05)

            assert isinstance(exc_info.value, TimeoutError)
            assert isinstance(exc_info.value, TransportError)


@pytest.mark.vpy("initial-core")
def test_fetch_frame_timeout_raises_remote_timeout_error(server: ServerFactory) -> None:
    """Verify that fetch_frame inside proxy VideoNode raises RemoteTimeoutError on real network slow render."""
    clip = vs.core.std.BlankClip(width=64, height=64, length=5)

    def slow_filter(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        time.sleep(0.3)
        return f

    slow_clip = clip.std.ModifyFrame(clip, slow_filter)

    with server([slow_clip]) as (host, port), RemoteClient(f"tcp://{host}:{port}") as client:
        # Connect proxy with a 50ms timeout; server render takes 300ms
        proxy = client.get_output(0, prefetch=0, timeout=0.05)

        # Calling get_frame triggers fetch_frame; VapourSynth catches RemoteTimeoutError and wraps in vs.Error
        with pytest.raises(vs.Error) as exc_info:
            proxy.get_frame(0)

        assert "Timed out fetching remote frame" in str(exc_info.value)


# Protocol & Framing Errors
def test_transport_resolve_malformed_response_frames() -> None:
    """Verify that malformed server response frames reject the future with MalformedMessageError."""
    tracker = RequestTracker()
    req_id, fut = tracker.allocate(bytes)

    # Empty frames -> MalformedMessageError
    assert tracker.resolve(req_id, [])
    with pytest.raises(MalformedMessageError, match="Malformed multipart response"):
        fut.result()


def test_transport_resolve_unknown_status_code() -> None:
    """Verify that an unrecognized status code byte rejects the future with UnknownStatusCodeError."""
    tracker = RequestTracker()
    req_id, fut = tracker.allocate(bytes)

    # Status byte 255 is not a valid StatusCode
    assert tracker.resolve(req_id, [bytes([255]), b""])
    with pytest.raises(UnknownStatusCodeError, match="Unknown status code byte"):
        fut.result()


def test_create_remote_vnode_unsupported_variable_format() -> None:
    """Verify that create_remote_vnode defends against variable format (format_id=0)."""

    class DummyTransport:
        def get_clip_info(self, output_index: int = 0) -> SafeUnifiedFuture[ClipInfo]:
            fut = SafeUnifiedFuture[ClipInfo]()
            fut.set_result(
                ClipInfo(
                    width=64,
                    height=64,
                    fps_num=24,
                    fps_den=1,
                    num_frames=10,
                    format_id=0,  # Variable format
                    format_name="Variable",
                    num_planes=0,
                    bytes_per_sample=1,
                    bits_per_sample=8,
                    subsampling_w=0,
                    subsampling_h=0,
                    planes=[],
                )
            )
            return fut

    with pytest.raises(UnsupportedFormatError, match="unsupported variable format"):
        create_remote_vnode(DummyTransport(), output_index=0, compression="none")  # type: ignore[arg-type]


@pytest.mark.vpy("initial-core")
def test_clip_info_variable_format_raises_unsupported_format_error() -> None:
    """Verify that ClipInfo.from_clip raises UnsupportedFormatError on variable-format clips."""
    var_clip = vs.core.std.BlankClip(varformat=True)

    with pytest.raises(UnsupportedFormatError, match="Variable format clips are not supported"):
        ClipInfo.from_clip(var_clip)


# ==============================================================================
# 4. Server Error Status Codes & Custom Remote Exceptions
# ==============================================================================


@pytest.mark.vpy("initial-core")
def test_remote_not_found_error_on_invalid_output(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    """Verify that querying a non-existent output index raises RemoteNotFoundError."""
    with server([test_clip]) as (host, port), RemoteClient(f"tcp://{host}:{port}") as client:
        # get_clip_info on invalid index
        with pytest.raises(RemoteNotFoundError) as exc_info:
            client.get_clip_info(output_index=42).result()
        assert exc_info.value.status == StatusCode.NOT_FOUND
        assert isinstance(exc_info.value, KeyError)

        # get_output on invalid index
        with pytest.raises(RemoteNotFoundError) as exc_info_vnode:
            client.get_output(output_index=42)
        assert exc_info_vnode.value.status == StatusCode.NOT_FOUND


@pytest.mark.vpy("initial-core")
def test_remote_authentication_error_on_all_client_methods(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    """Verify that missing auth token raises RemoteAuthenticationError across all client methods."""
    token = "secure_token_123"
    with (
        server([test_clip], auth_token=token) as (host, port),
        RemoteClient(f"tcp://{host}:{port}", auth_token=None) as unauthed_client,
    ):
        with pytest.raises(RemoteAuthenticationError) as exc_info:
            unauthed_client.list_outputs().result()
        assert exc_info.value.status == StatusCode.UNAUTHORIZED
        assert isinstance(exc_info.value, PermissionError)

        with pytest.raises(RemoteAuthenticationError):
            unauthed_client.get_stats().result()

        with pytest.raises(RemoteAuthenticationError):
            unauthed_client.get_clip_info(0).result()

        with pytest.raises(RemoteAuthenticationError):
            unauthed_client.get_output(0)


@pytest.mark.vpy("initial-core")
def test_remote_not_found_on_load_script_and_reload(
    server: ServerFactory, test_clip: vs.VideoNode, tmp_path: Path
) -> None:
    """Verify that load_script on non-existent file and reload without script raise RemoteNotFoundError."""
    script = tmp_path / "valid.vpy"
    script.write_text("import vapoursynth as vs\nvs.core.std.BlankClip().set_output(0)\n", encoding="utf-8")

    with server(script, allow_eval=True) as (host, port), RemoteClient(f"tcp://{host}:{port}") as client:
        # load_script on non-existent file -> RemoteNotFoundError (StatusCode.NOT_FOUND)
        with pytest.raises(RemoteNotFoundError, match="Failed to load script") as exc_info:
            client.load_script(tmp_path / "missing_file.vpy").result()
        assert exc_info.value.status == StatusCode.NOT_FOUND
        assert isinstance(exc_info.value, KeyError)

    # Reload on server with no script file
    with server([test_clip]) as (host, port), RemoteClient(f"tcp://{host}:{port}") as client2:
        with pytest.raises(RemoteNotFoundError, match="Failed to reload script") as exc_info2:
            client2.reload().result()
        assert exc_info2.value.status == StatusCode.NOT_FOUND
        assert isinstance(exc_info2.value, KeyError)


@pytest.mark.vpy("initial-core")
def test_remote_permission_error(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    """Verify that disabled dynamic execution raises RemotePermissionError on load_code and load_script."""
    with server([test_clip], allow_eval=False) as (host, port), RemoteClient(f"tcp://{host}:{port}") as client:
        with pytest.raises(RemotePermissionError, match="Dynamic code evaluation is disabled") as exc_info:
            client.load_code("import vapoursynth as vs").result()
        assert exc_info.value.status == StatusCode.PERMISSION_DENIED
        assert isinstance(exc_info.value, PermissionError)


@pytest.mark.vpy("initial-core")
def test_remote_execution_error_attributes_on_failed_eval(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    """Verify that server-side execution errors populate RemoteExecutionError with structured traceback."""
    with server([test_clip], allow_eval=True) as (host, port), RemoteClient(f"tcp://{host}:{port}") as client:
        bad_code = "def foo():\n    raise ValueError('Custom failure in script')\nfoo()\n"
        with pytest.raises(RemoteExecutionError) as exc_info:
            client.load_code(bad_code).result()

        err = exc_info.value
        assert err.status == StatusCode.ERROR
        assert err.exc_type == "ValueError"
        assert "Custom failure in script" in err.exc_msg
        assert err.formatted_traceback is not None
        assert "Custom failure in script" in err.formatted_traceback
        assert isinstance(err, RuntimeError)


def test_remote_command_and_payload_errors_via_response_envelope() -> None:
    """Verify that INVALID_COMMAND and INVALID_PAYLOAD status codes raise respective RemoteError types."""
    # StatusCode.INVALID_COMMAND -> RemoteCommandError
    cmd_env = ResponseEnvelope(status=StatusCode.INVALID_COMMAND, payload="Unknown command byte")
    with pytest.raises(RemoteCommandError) as exc_cmd:
        cmd_env.raise_for_status("Command rejected")
    assert exc_cmd.value.status == StatusCode.INVALID_COMMAND
    assert isinstance(exc_cmd.value, ValueError)

    # StatusCode.INVALID_PAYLOAD -> RemotePayloadError
    payload_env = ResponseEnvelope(status=StatusCode.INVALID_PAYLOAD, payload="Malformed msgpack")
    with pytest.raises(RemotePayloadError) as exc_pay:
        payload_env.raise_for_status("Payload rejected")
    assert exc_pay.value.status == StatusCode.INVALID_PAYLOAD
    assert isinstance(exc_pay.value, ValueError)


# source() Helper Function Edge Cases
@pytest.mark.vpy("initial-core")
def test_source_helper_error_handling(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    """Verify source() raises RemoteNotFoundError and RemoteAuthenticationError correctly."""
    token = "my_token"
    with server([test_clip], auth_token=token) as (host, port):
        # Unauthenticated source()
        with pytest.raises(RemoteAuthenticationError):
            source(f"tcp://{host}:{port}", output=0, auth_token="wrong_token")

        # Invalid output index
        with pytest.raises(RemoteNotFoundError):
            source(f"tcp://{host}:{port}", output=99, auth_token=token)
