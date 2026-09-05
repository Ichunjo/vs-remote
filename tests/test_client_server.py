from __future__ import annotations

import asyncio
import io
import logging
import struct
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, override
from unittest.mock import MagicMock

import pytest
import vapoursynth as vs
import vsengine.video
import zmq
import zmq.asyncio
from vsengine.policy import ManagedEnvironment, Policy

from vsremote.client import ClientTransport, RemoteClient, source
from vsremote.exceptions import (
    EnvironmentNotSetError,
    OutputNotFoundError,
    RemoteAuthenticationError,
    RemotePermissionError,
    ScriptNotLoadedError,
    TransportClosedError,
    TransportNotStartedError,
)
from vsremote.protocol import Command, RemoteLogRecord, StatusCode, StreamEvent, StreamOutputEvent, pack_payload
from vsremote.server import LogForwarder, RemotePolicy, ScriptRunner, ServerDaemon
from vsremote.server.daemon import _is_loopback_address
from vsremote.utils import setup_logging

if TYPE_CHECKING:
    from conftest import ServerFactory

HOST = "127.0.0.1"
core = vs.core


class LogCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records = list[logging.LogRecord]()

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.vpy("initial-core")
def test_ping(running_server: tuple[str, int]) -> None:
    host, port = running_server
    with RemoteClient(f"tcp://{host}:{port}") as client:
        assert client.ping().result() is True


@pytest.mark.vpy("initial-core")
def test_list_outputs(running_server: tuple[str, int]) -> None:
    host, port = running_server
    with RemoteClient(f"tcp://{host}:{port}") as client:
        outputs = client.list_outputs().result()
        assert len(outputs) == 2
        assert outputs[0].index == 0
        assert outputs[0].info.width == 128
        assert outputs[0].info.height == 128
        assert outputs[0].info.num_frames == 20

        assert outputs[1].index == 1
        assert outputs[1].info.width == 64
        assert outputs[1].info.height == 64
        assert outputs[1].info.num_frames == 10


@pytest.mark.asyncio(loop_factories=["stdlib"])
@pytest.mark.vpy("initial-core")
async def test_async_client_operations(running_server: tuple[str, int]) -> None:
    host, port = running_server
    async with RemoteClient(f"tcp://{host}:{port}") as client:
        # Async ping
        assert (await client.ping()) is True

        # Async list outputs
        outputs = await client.list_outputs()
        assert len(outputs) == 2
        assert outputs[0].index == 0

        # Async get clip info
        info = await client.get_clip_info(0)
        assert info.width == 128
        assert info.num_frames == 20

        # Async request frame
        header, planes = await client.request_frame(0, 5)
        assert header.status == StatusCode.OK
        assert header.n == 5
        assert len(planes) == 3


@pytest.mark.vpy("initial-core")
def test_remote_source_frame_rendering(running_server: tuple[str, int], test_clip: vs.VideoNode) -> None:
    host, port = running_server
    address = f"tcp://{host}:{port}"

    # Connect proxy node using source() helper
    remote_clip = source(address, output=0, compression="zstd")

    assert remote_clip.width == test_clip.width
    assert remote_clip.height == test_clip.height
    assert remote_clip.num_frames == test_clip.num_frames
    assert remote_clip.format.id == test_clip.format.id

    # Test individual frames and verify bit-for-bit pixel matching and props
    for frame_num, (remote_frame, local_frame) in enumerate(
        zip(remote_clip.frames(close=True), test_clip.frames(close=True))
    ):
        remote_frame = remote_clip.get_frame(frame_num)
        local_frame = test_clip.get_frame(frame_num)

        # Check props
        assert remote_frame.props["_SARNum"] == local_frame.props["_SARNum"]
        assert remote_frame.props["TestString"] == f"Frame_{frame_num}"
        assert remote_frame.props["TestInt"] == frame_num * 10
        assert remote_frame.props["TestFloat"] == frame_num * 0.5
        assert remote_frame.props["TestBytes"] == b"raw_data_bytes"

        # Check all plane pixels
        for p in range(remote_frame.format.num_planes):
            assert bytes(remote_frame[p]) == bytes(local_frame[p])


@pytest.mark.vpy("initial-core")
def test_remote_source_10bit(running_server: tuple[str, int]) -> None:
    host, port = running_server
    address = f"tcp://{host}:{port}"

    with RemoteClient(address, compression="none") as client:
        remote_clip = client.get_output(1)
        assert remote_clip.format.id == vs.YUV420P10
        assert remote_clip.width == 64
        assert remote_clip.num_frames == 10


@pytest.mark.vpy("initial-core")
def test_concurrent_frame_requests(running_server: tuple[str, int]) -> None:
    host, port = running_server
    address = f"tcp://{host}:{port}"

    with RemoteClient(address, compression="zstd") as client:
        remote_clip = client.get_output(0)

        # Concurrently request 20 frames across 8 worker threads
        def fetch(n: int) -> int:
            f = remote_clip.get_frame(n % 20)
            return f.props["TestInt"]  # type: ignore[return-value]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(fetch, range(40)))

        assert len(results) == 40
        for idx, res in enumerate(results):
            assert res == (idx % 20) * 10


@pytest.mark.vpy("initial-core")
def test_vsengine_frames_streaming(running_server: tuple[str, int]) -> None:
    host, port = running_server
    address = f"tcp://{host}:{port}"

    with RemoteClient(address, compression="zstd") as client:
        remote_clip = client.get_output(0)

        # Stream frames through vsengine.video.frames with prefetching and backlog
        count = 0
        for idx, f in enumerate(vsengine.video.frames(remote_clip, prefetch=4, backlog=8)):
            assert f.props["TestInt"] == idx * 10
            count += 1
        assert count == 20


@pytest.mark.vpy("initial-core")
def test_error_handling_out_of_bounds(running_server: tuple[str, int]) -> None:
    host, port = running_server
    address = f"tcp://{host}:{port}"

    with RemoteClient(address) as client:
        # Local VideoNode bounds check
        remote_clip = client.get_output(0)
        with pytest.raises(ValueError, match="beyond the last frame"):
            remote_clip.get_frame(999)

        # Server-side invalid frame request error response
        header, _ = client.transport.request_frame(output_index=0, n=999).result()
        assert header.status == StatusCode.ERROR
        assert "Invalid frame number" in header.error_message

        # Server-side invalid output index error
        with pytest.raises(KeyError, match="not found"):
            client.get_clip_info(output_index=999).result()


@pytest.mark.vpy("no-policy")
def test_script_runner_no_policy(tmp_path: Path) -> None:
    """Verify ScriptRunner initializes its own Policy when running in a zero-policy environment."""
    script_file = tmp_path / "test_no_policy.vpy"
    script_file.write_text(
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=320, height=240, length=5)\n"
        "clip.set_output(0)\n",
        encoding="utf-8",
    )

    assert not vs.has_policy()
    runner = ScriptRunner.from_script(script_file)
    try:
        outputs = runner.list_outputs()
        assert len(outputs) == 1
        assert outputs[0].index == 0
        assert outputs[0].info.width == 320
        assert outputs[0].info.height == 240
        assert outputs[0].info.num_frames == 5

        clip = runner.get_clip(0)
        assert clip.num_frames == 5
    finally:
        runner.close()
    assert not vs.has_policy()


@pytest.mark.vpy("no-core")
def test_script_runner_with_policy(tmp_path: Path, vpy_policy: Policy) -> None:
    """Verify ScriptRunner runs correctly with an explicitly supplied Policy."""
    script_file = tmp_path / "test_policy.vpy"
    script_file.write_text(
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=160, height=120, length=3)\n"
        "clip.set_output(0)\n",
        encoding="utf-8",
    )

    runner = ScriptRunner.from_script(script_file, environment=vpy_policy)
    try:
        outputs = runner.list_outputs()
        assert len(outputs) == 1
        assert outputs[0].info.width == 160
    finally:
        runner.close()


@pytest.mark.vpy("no-core")
def test_script_runner_with_env_factory(tmp_path: Path, vpy_env_factory: Callable[[], ManagedEnvironment]) -> None:
    """Verify ScriptRunner runs correctly with an explicitly supplied ManagedEnvironment."""
    script_file = tmp_path / "test_env.vpy"
    script_file.write_text(
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=64, height=64, length=2)\n"
        "clip.set_output(0)\n",
        encoding="utf-8",
    )

    runner = ScriptRunner.from_script(script_file, environment=vpy_env_factory())
    try:
        outputs = runner.list_outputs()
        assert len(outputs) == 1
        assert outputs[0].info.width == 64
    finally:
        runner.close()


@pytest.mark.vpy("initial-core")
def test_client_transport_lifecycle(running_server: tuple[str, int]) -> None:
    """Verify ClientTransport requires explicit start and raises RuntimeError when not started."""
    host, port = running_server
    address = f"tcp://{host}:{port}"

    transport = ClientTransport(address)

    # Unstarted transport should reject requests
    with pytest.raises(RuntimeError, match="Transport is not started"):
        transport.send_request(Command.PING).result()

    with pytest.raises(RuntimeError, match="Transport is not started"):
        transport.list_outputs().result()

    # Start transport and verify communication
    transport.start()
    assert transport.ping().result() is True
    assert len(transport.list_outputs().result()) == 2

    # Multiple start calls should be idempotent
    transport.start()
    assert transport.ping().result() is True

    # Close transport and verify idempotency
    transport.close()
    transport.close()


@pytest.mark.vpy("initial-core")
def test_client_transport_context_manager(running_server: tuple[str, int]) -> None:
    """Verify ClientTransport works as a synchronous context manager."""
    host, port = running_server
    with ClientTransport(f"tcp://{host}:{port}") as transport:
        assert transport.ping().result() is True
        outputs = transport.list_outputs().result()
        assert len(outputs) == 2


@pytest.mark.asyncio(loop_factories=["stdlib"])
@pytest.mark.vpy("initial-core")
async def test_client_transport_async_context_manager(running_server: tuple[str, int]) -> None:
    """Verify ClientTransport works as an asynchronous context manager."""
    host, port = running_server
    async with ClientTransport(f"tcp://{host}:{port}") as transport:
        assert (await transport.ping()) is True
        outputs = await transport.list_outputs()
        assert len(outputs) == 2


@pytest.mark.vpy("initial-core")
def test_remote_client_context_manager_clip_lifecycle(running_server: tuple[str, int]) -> None:
    """Verify clips created on RemoteClient cannot render uncached frames after client context manager is closed."""
    host, port = running_server
    with RemoteClient(f"tcp://{host}:{port}") as client:
        clip = client.get_output(0)
        # Frame request succeeds while inside context manager
        frame = clip.get_frame(0)
        assert frame.width == 128
        assert frame.height == 128

    # Frame request for uncached frame fails after client context manager has closed the transport
    with pytest.raises((vs.Error, RuntimeError)):
        clip.get_frame(10)


@pytest.mark.vpy("initial-core")
def test_remote_client_shared_transport_multiple_clips(running_server: tuple[str, int]) -> None:
    """Verify multiple clips created on a single RemoteClient share the same transport."""
    host, port = running_server
    client = RemoteClient(f"tcp://{host}:{port}").start()
    try:
        clip0 = client.get_output(0)
        clip1 = client.get_output(1)

        frame0 = clip0.get_frame(0)
        assert frame0.width == 128

        frame1 = clip1.get_frame(0)
        assert frame1.width == 64
    finally:
        client.close()

    # Frame request for uncached frame fails after client is closed
    with pytest.raises((vs.Error, RuntimeError)):
        clip0.get_frame(10)
    with pytest.raises((vs.Error, RuntimeError)):
        clip1.get_frame(9)


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_stream_structured_log_records(server: ServerFactory, tmp_path: Path) -> None:
    """Verify server-side Python logging is streamed as structured LogRecords to the client."""
    script_file = tmp_path / "test_logging.vpy"
    script_file.write_text(
        "import logging\n"
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=64, height=64, length=2)\n"
        "clip.set_output(0)\n"
        "logging.getLogger('test_custom_logger').warning('Custom warning: %s', 'arg_val')\n",
        encoding="utf-8",
    )

    async with server(script_file, compression="zstd") as (host, port):
        collector = LogCollector()
        target_logger = logging.getLogger("test_custom_logger")
        target_logger.addHandler(collector)
        try:
            async with RemoteClient(f"tcp://{host}:{port}") as client:
                assert (await client.ping()) is True
                await asyncio.sleep(0.1)

            matching = [r for r in collector.records if r.name == "test_custom_logger"]
            assert len(matching) >= 1
            assert matching[0].levelno == logging.WARNING
            assert matching[0].msg == "Custom warning: %s"
            assert matching[0].args == ("arg_val",)
            assert matching[0].getMessage() == "Custom warning: arg_val"
        finally:
            target_logger.removeHandler(collector)


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_stream_vapoursynth_log_message(server: ServerFactory, tmp_path: Path) -> None:
    """Verify VapourSynth core.log_message is intercepted and streamed to the client."""
    script_file = tmp_path / "test_vs_log.vpy"
    script_file.write_text(
        "import logging\n"
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=64, height=64, length=2)\n"
        "clip.set_output(0)\n"
        "logging.getLogger('vapoursynth').warning('VS core warning raised')\n",
        encoding="utf-8",
    )

    async with server(script_file, compression="zstd") as (host, port):
        collector = LogCollector()
        vs_logger = logging.getLogger("vapoursynth")
        vs_logger.addHandler(collector)
        try:
            async with RemoteClient(f"tcp://{host}:{port}") as client:
                assert (await client.ping()) is True
                await asyncio.sleep(0.1)

            vs_records = [
                r for r in collector.records if r.name == "vapoursynth" and "VS core warning raised" in r.getMessage()
            ]
            assert len(vs_records) >= 1
            assert vs_records[0].levelno == logging.WARNING
        finally:
            vs_logger.removeHandler(collector)


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_stream_replay_history_disabled(server: ServerFactory, tmp_path: Path) -> None:
    """Verify replay_history=False suppresses replaying startup events from previously loaded scripts."""
    script_file = tmp_path / "test_replay_history.vpy"
    script_file.write_text(
        "import logging\n"
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=64, height=64, length=2)\n"
        "clip.set_output(0)\n"
        "logging.getLogger('test_replay_logger').warning('Startup warning to ignore')\n",
        encoding="utf-8",
    )

    async with server(script_file, compression="zstd") as (host, port):
        collector = LogCollector()
        target_logger = logging.getLogger("test_replay_logger")
        target_logger.addHandler(collector)
        try:
            async with RemoteClient(f"tcp://{host}:{port}", replay_history=False) as client:
                assert (await client.ping()) is True
                await asyncio.sleep(0.1)

            matching = [r for r in collector.records if r.name == "test_replay_logger"]
            assert len(matching) == 0
        finally:
            target_logger.removeHandler(collector)


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_stream_stdout_and_stderr(server: ServerFactory, tmp_path: Path) -> None:
    """Verify direct print() and sys.stderr.write() from script are streamed to client buffers."""
    script_file = tmp_path / "test_streams.vpy"
    script_file.write_text(
        "import sys\n"
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=64, height=64, length=2)\n"
        "clip.set_output(0)\n"
        "print('Hello from remote stdout!')\n"
        "sys.stderr.write('Direct stderr message\\n')\n",
        encoding="utf-8",
    )

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    async with (
        server(script_file, compression="zstd") as (host, port),
        RemoteClient(f"tcp://{host}:{port}", stdout=stdout_buf, stderr=stderr_buf) as client,
    ):
        assert (await client.ping()) is True
        await asyncio.sleep(0.1)

        assert "Hello from remote stdout!" in stdout_buf.getvalue()
        assert "Direct stderr message" in stderr_buf.getvalue()


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_stream_logs_not_duplicated_in_stderr(server: ServerFactory, tmp_path: Path) -> None:
    """Verify logs outputted during script run are forwarded as LogRecords without duplicating to stderr."""

    script_file = tmp_path / "test_no_dup.vpy"
    script_file.write_text(
        "import sys\n"
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=64, height=64, length=2)\n"
        "clip.set_output(0)\n"
        "core.log_message(vs.MESSAGE_TYPE_WARNING, 'VS core warning message!')\n"
        "print('Remote script stdout message')\n",
        encoding="utf-8",
    )

    async with server(script_file, compression="zstd") as (host, port):
        collector = LogCollector()
        vs_logger = logging.getLogger("vapoursynth")
        vs_logger.addHandler(collector)

        try:
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            async with RemoteClient(f"tcp://{host}:{port}", stdout=stdout_buf, stderr=stderr_buf) as client:
                assert (await client.ping()) is True
                await asyncio.sleep(0.1)

            vs_records = [
                r
                for r in collector.records
                if getattr(r, "_is_remote", False) and "VS core warning message!" in r.getMessage()
            ]
            assert len(vs_records) == 1
            assert vs_records[0].levelno == logging.WARNING

            assert "Remote script stdout message" in stdout_buf.getvalue()
            assert "VS core warning message!" not in stderr_buf.getvalue()
        finally:
            vs_logger.removeHandler(collector)


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_forward_logs_disabled(server: ServerFactory, tmp_path: Path) -> None:
    """Verify forward_logs=False suppresses forwarding records to Python logging."""
    script_file = tmp_path / "test_logging_disabled.vpy"
    script_file.write_text(
        "import logging\n"
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=64, height=64, length=2)\n"
        "clip.set_output(0)\n"
        "logging.getLogger('test_disabled_logger').warning('Disabled warning')\n",
        encoding="utf-8",
    )

    async with server(script_file, compression="zstd") as (host, port):
        collector = LogCollector()
        target_logger = logging.getLogger("test_disabled_logger")
        target_logger.addHandler(collector)
        try:
            async with RemoteClient(f"tcp://{host}:{port}", forward_logs=False) as client:
                assert (await client.ping()) is True
                await asyncio.sleep(0.1)

            matching = [r for r in collector.records if r.name == "test_disabled_logger"]
            assert len(matching) == 0
        finally:
            target_logger.removeHandler(collector)


@pytest.mark.vpy("no-policy")
def test_remote_policy_logger_interception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RemotePolicy intercepts core.log_message via api.set_logger."""
    logs = list[tuple[int, str]]()

    monkeypatch.setattr(
        logging.getLogger("vapoursynth"),
        "log",
        lambda lvl, msg, *args, **kwargs: logs.append((lvl, msg)),
    )

    with RemotePolicy() as policy, policy.new_environment() as env, env.use():
        env.core.log_message(vs.MESSAGE_TYPE_WARNING, "Direct core log from RemotePolicy")

    assert len(logs) == 1
    assert logs[0][0] == logging.WARNING
    assert logs[0][1] == "Direct core log from RemotePolicy"


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_client_reload_script(server: ServerFactory, tmp_path: Path) -> None:
    """Test client triggering a server script reload after modifying the script on disk."""
    script_file = tmp_path / "reload_test.vpy"
    script_file.write_text(
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=100, height=80, format=vs.YUV420P8, length=5)\n"
        "clip.set_output(0)\n",
        encoding="utf-8",
    )

    async with server(script_file) as (host, port), RemoteClient(f"tcp://{host}:{port}") as client:
        outputs = await client.list_outputs()
        assert len(outputs) == 1
        assert outputs[0].info.width == 100
        assert outputs[0].info.num_frames == 5

        # Modify script on disk
        script_file.write_text(
            "import vapoursynth as vs\n"
            "core = vs.core\n"
            "clip0 = core.std.BlankClip(width=200, height=160, format=vs.YUV420P8, length=15)\n"
            "clip1 = core.std.BlankClip(width=50, height=40, format=vs.RGB24, length=8)\n"
            "clip0.set_output(0)\n"
            "clip1.set_output(1)\n",
            encoding="utf-8",
        )

        # Trigger reload via client
        new_outputs = await client.reload()
        assert len(new_outputs) == 2
        assert new_outputs[0].index == 0
        assert new_outputs[0].info.width == 200
        assert new_outputs[0].info.num_frames == 15
        assert new_outputs[1].index == 1
        assert new_outputs[1].info.width == 50
        assert new_outputs[1].info.num_frames == 8

        # Verify frame rendering from reloaded environment
        header, _ = await client.request_frame(0, 10)
        assert header.status == StatusCode.OK
        assert header.n == 10


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_client_load_code_and_error_handling(server: ServerFactory) -> None:
    """Test dynamic code execution on the server via client.load_code and error resilience."""

    async with server(ScriptRunner(), allow_eval=True) as (host, port), RemoteClient(f"tcp://{host}:{port}") as client:
        # Execute valid code string
        code = (
            "import vapoursynth as vs\n"
            "core = vs.core\n"
            "clip = core.std.BlankClip(width=320, height=240, format=vs.YUV420P8, length=12)\n"
            "clip.set_output(0)\n"
        )
        outputs = await client.load_code(code)
        assert len(outputs) == 1
        assert outputs[0].info.width == 320
        assert outputs[0].info.num_frames == 12

        header, _ = await client.request_frame(0, 5)
        assert header.status == StatusCode.OK

        # Execute invalid code - should return error without crashing server
        with pytest.raises(RuntimeError, match="Failed to load code"):
            await client.load_code("this is not valid python code !!!")

        # Server remains healthy and can execute subsequent code
        valid_code_2 = (
            "import vapoursynth as vs\n"
            "core = vs.core\n"
            "clip = core.std.BlankClip(width=640, height=480, format=vs.YUV420P8, length=25)\n"
            "clip.set_output(0)\n"
        )
        new_outputs = await client.load_code(valid_code_2)
        assert len(new_outputs) == 1
        assert new_outputs[0].info.width == 640
        assert new_outputs[0].info.num_frames == 25


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_client_load_script_switch(server: ServerFactory, tmp_path: Path) -> None:
    """Test switching between different script files dynamically via client.load_script."""
    script1 = tmp_path / "script1.vpy"
    script1.write_text(
        "import vapoursynth as vs\ncore = vs.core\ncore.std.BlankClip(width=120, height=90, length=4).set_output(0)\n",
        encoding="utf-8",
    )

    script2 = tmp_path / "script2.vpy"
    script2.write_text(
        "import vapoursynth as vs\ncore = vs.core\ncore.std.BlankClip(width=240, height=180, length=8).set_output(0)\n",
        encoding="utf-8",
    )

    async with server(script1, allow_eval=True) as (host, port), RemoteClient(f"tcp://{host}:{port}") as client:
        outputs1 = await client.list_outputs()
        assert outputs1[0].info.width == 120

        # Switch script
        outputs2 = await client.load_script(script2)
        assert outputs2[0].info.width == 240
        assert outputs2[0].info.num_frames == 8

        # Try loading nonexistent script
        with pytest.raises(RuntimeError, match="Failed to load script"):
            await client.load_script(tmp_path / "nonexistent.vpy")


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_permission_denied_when_allow_eval_disabled(server: ServerFactory, tmp_path: Path) -> None:
    """Test that dynamic code and script execution is blocked when allow_eval is False (default)."""
    script = tmp_path / "static_script.vpy"
    script.write_text(
        "import vapoursynth as vs\ncore = vs.core\n"
        "core.std.BlankClip(width=100, height=100, length=10).set_output(0)\n",
        encoding="utf-8",
    )

    async with (
        server(script, allow_eval=False) as (host, port),
        RemoteClient(f"tcp://{host}:{port}") as client,
    ):
        # Listing outputs and fetching frames should work normally
        outputs = await client.list_outputs()
        assert len(outputs) == 1

        # Dynamic code loading must fail with permission denied error
        with pytest.raises(RemotePermissionError, match="Dynamic code evaluation is disabled on this server"):
            await client.load_code("core.std.BlankClip().set_output(0)")

        # Dynamic script switching must fail with permission denied error
        with pytest.raises(RemotePermissionError, match="Dynamic script loading is disabled on this server"):
            await client.load_script(script)


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_auth_token_security(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    """Test authentication token enforcement on ServerDaemon."""
    token = "secret_access_token_xyz"
    async with server([test_clip], auth_token=token) as (host, port):
        # Connecting without token -> should fail operations with unauthorized / ConnectionReset
        async with RemoteClient(f"tcp://{host}:{port}", auth_token=None) as client_unauthed:
            assert (await client_unauthed.ping()) is False
            with pytest.raises(RemoteAuthenticationError, match="Failed to list outputs"):
                await client_unauthed.list_outputs()

        # Connecting with wrong token -> should fail
        async with RemoteClient(f"tcp://{host}:{port}", auth_token="wrong_token_abc") as client_wrong:
            assert (await client_wrong.ping()) is False
            with pytest.raises(RemoteAuthenticationError, match="Failed to list outputs"):
                await client_wrong.list_outputs()

        # Connecting with valid token -> should succeed
        async with RemoteClient(f"tcp://{host}:{port}", auth_token=token) as client_authed:
            assert (await client_authed.ping()) is True
            outputs = await client_authed.list_outputs()
            assert len(outputs) == 1
            header, _ = await client_authed.request_frame(0, 0)
            assert header.status == StatusCode.OK


@pytest.mark.vpy("initial-core")
def test_curvezmq_end_to_end_encryption(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    """Test ZeroMQ CurveZMQ end-to-end encrypted connection and frame streaming."""
    server_public, server_secret = zmq.curve_keypair()

    with (
        server([test_clip], curve_secret_key=server_secret, curve_public_key=server_public) as (host, port),
        RemoteClient(f"tcp://{host}:{port}", curve_server_key=server_public) as client,
    ):
        assert client.ping().result() is True
        outputs = client.list_outputs().result()
        assert len(outputs) == 1

        # Fetch encrypted frame proxy via source()
        proxy = source(f"tcp://{host}:{port}", output=0, curve_server_key=server_public)
        frame = proxy.get_frame(0)
        assert frame.width == test_clip.width
        assert bytes(frame[0]) == bytes(test_clip.get_frame(0)[0])


@pytest.mark.vpy("initial-core")
def test_client_get_outputs(running_server: tuple[str, int]) -> None:
    """Test RemoteClient.get_outputs() returning dictionary of all proxy VideoNodes."""
    host, port = running_server
    with RemoteClient(f"tcp://{host}:{port}") as client:
        clips = client.get_outputs()
        assert len(clips) == 2
        assert 0 in clips
        assert 1 in clips
        assert clips[0].width == 128
        assert clips[1].width == 64
        assert clips[0].get_frame(0).width == 128
        assert clips[1].get_frame(0).width == 64


@pytest.mark.vpy("initial-core")
def test_remote_traceback_propagation(server: ServerFactory) -> None:
    """Test that server evaluation errors include formatted remote traceback."""
    # Create a clip that raises an error on frame 1
    clip = core.std.BlankClip(width=64, height=64, length=5)

    def faulty_filter(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        if n == 1:
            raise ValueError("Intentional error inside custom filter")
        return f

    faulty_clip = clip.std.ModifyFrame(clip, faulty_filter)

    with server([faulty_clip]) as (host, port), RemoteClient(f"tcp://{host}:{port}") as client:
        clip = client.get_output(0)
        # Frame 0 succeeds
        assert clip.get_frame(0)
        # Frame 1 fails and includes remote traceback
        with pytest.raises((vs.Error, RuntimeError)) as exc_info:
            clip.get_frame(1)
        err_msg = str(exc_info.value)
        assert "Intentional error inside custom filter" in err_msg
        assert "[Remote Traceback]" in err_msg


@pytest.mark.vpy("initial-core")
def test_strided_copy_non_aligned(server: ServerFactory) -> None:
    """Test that clips with unaligned dimensions requiring strided line-by-line copies work perfectly."""
    # Odd width on RGB24 creates stride > row_size on standard aligned blank clips
    unaligned_clip = core.std.BlankClip(width=157, height=93, format=vs.RGB24, length=5, color=[120, 80, 200])
    with server([unaligned_clip]) as (host, port), RemoteClient(f"tcp://{host}:{port}", compression="none") as client:
        clip = client.get_output(0)
        assert clip.width == 157
        assert clip.height == 93
        frame = clip.get_frame(0)
        orig_frame = unaligned_clip.get_frame(0)
        for p in range(3):
            assert bytes(frame[p]) == bytes(orig_frame[p])


@pytest.mark.vpy("initial-core")
def test_client_address_normalization() -> None:
    client = RemoteClient("127.0.0.1:5555")
    assert client.address == "tcp://127.0.0.1:5555"

    trans = ClientTransport("127.0.0.1:5555")
    assert trans.address == "tcp://127.0.0.1:5555"


@pytest.mark.vpy("initial-core")
def test_source_reuse_transport(running_server: tuple[str, int]) -> None:
    host, port = running_server
    address = f"tcp://{host}:{port}"
    with ClientTransport(address) as trans:
        clip = source(trans, output=0)
        assert clip.width == 128
        frame = clip.get_frame(0)
        assert frame.width == 128


def test_client_stream_event_handlers() -> None:
    # Test stream event when stream target is None (returns early)
    client = RemoteClient("tcp://127.0.0.1:5555", stdout=None, stderr=None)
    client._handle_event(StreamOutputEvent(stream="stdout", text="ignore"))

    # Test stream event when stream target raises exception (logged)
    faulty_stream = MagicMock()
    faulty_stream.write.side_effect = OSError("Disk full")
    client_faulty = RemoteClient("tcp://127.0.0.1:5555", stdout=faulty_stream)
    client_faulty._handle_event(StreamOutputEvent(stream="stdout", text="error"))
    assert faulty_stream.write.called


def test_transport_event_dispatch_decoupling() -> None:
    received = list[tuple[str, StreamEvent]]()
    evt = threading.Event()

    def on_event(event: StreamEvent) -> None:
        # Simulate blocking I/O (e.g. slow flush or logging lock)
        time.sleep(0.01)
        received.append((threading.current_thread().name, event))
        if len(received) == 10:
            evt.set()

    with ClientTransport("tcp://127.0.0.1:5555", on_event=on_event) as trans:
        # Dispatch 10 events via _dispatch_response
        t0 = time.perf_counter()
        for i in range(10):
            parts = [
                b"\x00\x00\x00\x00",
                bytes([StatusCode.OK]),
                pack_payload(StreamOutputEvent(stream="stdout", text=f"chunk_{i}")),
            ]
            trans._dispatch_response(parts)
        dispatch_duration = time.perf_counter() - t0

        # All 10 dispatches must return immediately without waiting for the 10 * 10ms sleep
        assert dispatch_duration < 0.05, f"Dispatch took too long ({dispatch_duration:.4f}s), was it blocked?"

        # Wait for worker thread to process all items
        assert evt.wait(timeout=2.0), "Timed out waiting for decoupled event dispatcher to process events"

        # Verify items were processed on the dispatcher thread in strict FIFO order
        assert len(received) == 10
        for i, (thread_name, event) in enumerate(received):
            assert thread_name == "VSRemoteEventDispatcher"
            assert isinstance(event, StreamOutputEvent)
            assert event.text == f"chunk_{i}"


@pytest.mark.vpy("initial-core")
def test_client_seeking_future_pruning(server: ServerFactory) -> None:
    # Create 100 frame clip
    clip = core.std.BlankClip(width=64, height=64, length=100)
    with server([clip]) as (host, port):
        # Connect with prefetch=2
        proxy = source(f"tcp://{host}:{port}", output=0, prefetch=2)
        # Non-contiguous fetches add prefetches without popping previous prefetches
        proxy.get_frame(0)
        proxy.get_frame(10)
        proxy.get_frame(20)
        proxy.get_frame(30)
        proxy.get_frame(40)
        # Now len(inflight) is 10 > 2 * 4 (8). Jump to 90 to trigger pruning
        proxy.get_frame(90)


def test_transport_error_states() -> None:
    trans = ClientTransport("tcp://127.0.0.1:5555")
    # _send_message when not started
    with pytest.raises(TransportNotStartedError, match="Transport is not started"):
        trans._send_message(1, Command.PING, b"")

    # send_request when started but running is False
    trans._started = True
    trans._running = False
    fut = trans.send_request(Command.PING)
    with pytest.raises(TransportClosedError, match="ClientTransport is closed"):
        fut.result()


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_transport_unsubscribe_stream(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    async with server([test_clip]) as (host, port), ClientTransport(f"tcp://{host}:{port}") as trans:
        assert (await trans.ping()) is True
        assert (await trans.unsubscribe_stream()) is True


def test_curvezmq_explicit_client_keys_and_bytes() -> None:
    pub, sec = zmq.curve_keypair()
    server_pub, _ = zmq.curve_keypair()

    # String keys
    client_str = RemoteClient(
        "tcp://127.0.0.1:5555",
        curve_server_key=server_pub.decode("ascii"),
        curve_public_key=pub.decode("ascii"),
        curve_secret_key=sec.decode("ascii"),
    )
    assert client_str.curve_public_key == pub.decode("ascii")

    # Bytes keys
    client_bytes = RemoteClient(
        "tcp://127.0.0.1:5555",
        curve_server_key=server_pub,
        curve_public_key=pub,
        curve_secret_key=sec,
    )
    assert client_bytes.curve_public_key == pub


@pytest.mark.vpy("initial-core")
def test_curvezmq_explicit_keys_connected(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    server_public, server_secret = zmq.curve_keypair()
    client_public, client_secret = zmq.curve_keypair()

    with (
        server([test_clip], curve_secret_key=server_secret, curve_public_key=server_public) as (host, port),
        ClientTransport(
            f"tcp://{host}:{port}",
            curve_server_key=server_public,
            curve_public_key=client_public,
            curve_secret_key=client_secret,
        ) as trans,
    ):
        assert trans.ping().result() is True


@pytest.mark.vpy("initial-core")
def test_curvezmq_client_auth_whitelisted(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    server_public, server_secret = zmq.curve_keypair()
    client_public, client_secret = zmq.curve_keypair()

    with (
        server(
            [test_clip],
            curve_secret_key=server_secret,
            curve_public_key=server_public,
            curve_allowed_keys=[client_public],
        ) as (host, port),
        RemoteClient(
            f"tcp://{host}:{port}",
            curve_server_key=server_public,
            curve_public_key=client_public,
            curve_secret_key=client_secret,
        ) as client,
    ):
        assert client.ping().result() is True
        outputs = client.list_outputs().result()
        assert len(outputs) == 1


@pytest.mark.vpy("initial-core")
def test_curvezmq_client_auth_rejected_unauthorized(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    server_public, server_secret = zmq.curve_keypair()
    allowed_public, _ = zmq.curve_keypair()
    unauthorized_public, unauthorized_secret = zmq.curve_keypair()

    with (
        server(
            [test_clip],
            curve_secret_key=server_secret,
            curve_public_key=server_public,
            curve_allowed_keys=[allowed_public],
        ) as (host, port),
        RemoteClient(
            f"tcp://{host}:{port}",
            curve_server_key=server_public,
            curve_public_key=unauthorized_public,
            curve_secret_key=unauthorized_secret,
        ) as client,
        pytest.raises(TimeoutError),
    ):
        # Unauthorized client handshake is rejected by server ZAP handler, leading to timeout
        client.ping().result(timeout=0.5)


@pytest.mark.vpy("initial-core")
def test_curvezmq_client_auth_multiple_keys(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    server_public, server_secret = zmq.curve_keypair()
    c1_pub, c1_sec = zmq.curve_keypair()
    c2_pub, c2_sec = zmq.curve_keypair()
    c3_pub, c3_sec = zmq.curve_keypair()

    with (
        server(
            [test_clip],
            curve_secret_key=server_secret,
            curve_public_key=server_public,
            curve_allowed_keys=[c1_pub, c2_pub.decode("ascii")],
        ) as (host, port),
        RemoteClient(
            f"tcp://{host}:{port}",
            curve_server_key=server_public,
            curve_public_key=c1_pub,
            curve_secret_key=c1_sec,
        ) as client1,
        RemoteClient(
            f"tcp://{host}:{port}",
            curve_server_key=server_public,
            curve_public_key=c2_pub,
            curve_secret_key=c2_sec,
        ) as client2,
        RemoteClient(
            f"tcp://{host}:{port}",
            curve_server_key=server_public,
            curve_public_key=c3_pub,
            curve_secret_key=c3_sec,
        ) as client3,
    ):
        assert client1.ping().result() is True
        assert client2.ping().result() is True
        with pytest.raises(TimeoutError):
            client3.ping().result(timeout=0.5)


def test_curvezmq_invalid_key_length_raises() -> None:
    pub, sec = zmq.curve_keypair()

    with pytest.raises(ValueError, match="Invalid Curve key length"):
        ServerDaemon(ScriptRunner(), curve_secret_key=sec, curve_allowed_keys=["short_key"])

    with pytest.raises(ValueError, match="Invalid Curve key length"):
        ServerDaemon(ScriptRunner(), curve_secret_key=sec, curve_allowed_keys=[b"short_bytes"])

    with pytest.raises(ValueError, match="Invalid curve_secret_key length"):
        ServerDaemon(ScriptRunner(), curve_secret_key="short")

    with pytest.raises(ValueError, match="Invalid curve_public_key length"):
        ServerDaemon(ScriptRunner(), curve_secret_key=sec, curve_public_key="short")

    # Corrupted Z85 string (40 chars with invalid Z85 characters)
    with pytest.raises(struct.error, match="'I' format requires 0 <= number <= 4294967295"):
        ServerDaemon(ScriptRunner(), curve_secret_key=sec, curve_allowed_keys=["%" * 40])

    # Missing dependent curve parameters on server
    with pytest.raises(ValueError, match="curve_allowed_keys requires curve_secret_key"):
        ServerDaemon(ScriptRunner(), curve_allowed_keys=[pub])

    with pytest.raises(ValueError, match="curve_public_key requires curve_secret_key"):
        ServerDaemon(ScriptRunner(), curve_public_key=pub)

    # Missing dependent curve parameters on client transport
    with pytest.raises(ValueError, match="must both be specified"):
        ClientTransport(curve_server_key=pub, curve_public_key=pub)

    with pytest.raises(ValueError, match="must both be specified"):
        ClientTransport(curve_server_key=pub, curve_secret_key=sec)

    with pytest.raises(ValueError, match="require curve_server_key"):
        ClientTransport(curve_public_key=pub, curve_secret_key=sec)


@pytest.mark.vpy("initial-core")
def test_transport_reload_failure_handling(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    # Server with runner that has no script file -> reload fails
    with (
        server([test_clip]) as (host, port),
        ClientTransport(f"tcp://{host}:{port}") as trans,
        pytest.raises(RuntimeError, match="Failed to reload script"),
    ):
        trans.reload().result()


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_transport_on_event_error_handling(server: ServerFactory, tmp_path: Path) -> None:
    script_file = tmp_path / "event_err.vpy"
    script_file.write_text(
        "import logging\nimport vapoursynth as vs\ncore = vs.core\ncore.std.BlankClip().set_output(0)\n"
        "logging.getLogger('t').warning('msg')\n",
        encoding="utf-8",
    )

    def faulty_handler(evt: StreamEvent) -> None:
        raise ValueError("Handler error")

    async with (
        server(script_file) as (host, port),
        ClientTransport(f"tcp://{host}:{port}", on_event=faulty_handler) as trans,
    ):
        assert (await trans.ping()) is True
        await asyncio.sleep(0.1)


def test_server_daemon_address_normalization() -> None:
    runner = ScriptRunner()
    daemon = ServerDaemon(runner, "127.0.0.1:5555")
    assert daemon.address == "tcp://127.0.0.1:5555"


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_server_daemon_invalid_commands_and_payloads(server: ServerFactory, test_clip: vs.VideoNode) -> None:
    async with server([test_clip], allow_eval=True) as (host, port):
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.connect(f"tcp://{host}:{port}")

        try:
            # Malformed request (<3 frames) -> INVALID_COMMAND
            req_id_1 = (101).to_bytes(4, "big")
            await sock.send_multipart([req_id_1])
            reply_1 = await sock.recv_multipart()
            assert reply_1[0] == req_id_1
            assert reply_1[1] == bytes([StatusCode.INVALID_COMMAND])

            # GET_CLIP_INFO with corrupt payload -> INVALID_PAYLOAD
            req_id_2 = (102).to_bytes(4, "big")
            await sock.send_multipart([req_id_2, bytes([Command.GET_CLIP_INFO.value]), b"\xff\xff\xff"])
            reply_2 = await sock.recv_multipart()
            assert reply_2[0] == req_id_2
            assert reply_2[1] == bytes([StatusCode.INVALID_PAYLOAD])

            # GET_FRAME with corrupt payload -> INVALID_PAYLOAD
            req_id_3 = (103).to_bytes(4, "big")
            await sock.send_multipart([req_id_3, bytes([Command.GET_FRAME.value]), b"\xff\xff\xff"])
            reply_3 = await sock.recv_multipart()
            assert reply_3[0] == req_id_3
            assert reply_3[1] == bytes([StatusCode.INVALID_PAYLOAD])

            # GET_FRAME with invalid output index -> NOT_FOUND
            req_id_4 = (104).to_bytes(4, "big")
            bad_frame_req = pack_payload({"output_index": 999, "n": 0, "compression": "zstd"})
            await sock.send_multipart([req_id_4, bytes([Command.GET_FRAME.value]), bad_frame_req])
            reply_4 = await sock.recv_multipart()
            assert reply_4[0] == req_id_4
            assert reply_4[1] == bytes([StatusCode.NOT_FOUND])

            # RELOAD with corrupt payload -> INVALID_PAYLOAD
            req_id_5 = (105).to_bytes(4, "big")
            await sock.send_multipart([req_id_5, bytes([Command.RELOAD.value]), b"\xff\xff\xff"])
            reply_5 = await sock.recv_multipart()
            assert reply_5[0] == req_id_5
            assert reply_5[1] == bytes([StatusCode.INVALID_PAYLOAD])

            # UNSUBSCRIBE_STREAM command
            req_id_6 = (106).to_bytes(4, "big")
            await sock.send_multipart([req_id_6, bytes([Command.UNSUBSCRIBE_STREAM.value]), b""])
            reply_6 = await sock.recv_multipart()
            assert reply_6[0] == req_id_6
            assert reply_6[1] == bytes([StatusCode.OK])

            # CLOSE command
            req_id_7 = (107).to_bytes(4, "big")
            await sock.send_multipart([req_id_7, bytes([Command.CLOSE.value]), b""])
            reply_7 = await sock.recv_multipart()
            assert reply_7[0] == req_id_7
            assert reply_7[1] == bytes([StatusCode.OK])
            assert reply_7[2] == b"BYE"
        finally:
            sock.close(linger=0)
            ctx.term()


def test_server_daemon_send_multipart_closed() -> None:
    daemon = ServerDaemon(ScriptRunner())
    with pytest.raises(TransportClosedError):
        asyncio.run(daemon._send_multipart([b"test"]))


def test_log_forwarder_arg_types_and_errors() -> None:
    events = list[StreamEvent]()
    forwarder = LogForwarder(events.append)

    # Non-serializable tuple args
    rec_tuple = logging.LogRecord("test", logging.INFO, "path", 1, "Tuple: %s", (object(),), None)
    forwarder.emit(rec_tuple)
    assert len(events) == 1
    assert isinstance(events[-1], RemoteLogRecord)

    # Dict args (serializable)
    rec_dict = logging.LogRecord("test", logging.INFO, "path", 1, "Dict", ({"k": "v"},), None)
    forwarder.emit(rec_dict)
    assert len(events) == 2

    # Dict args (non-serializable)
    rec_dict_unserializable = logging.LogRecord("test", logging.INFO, "path", 1, "Dict", ({"k": object()},), None)
    forwarder.emit(rec_dict_unserializable)
    assert len(events) == 3

    # Scalar non-tuple non-dict args
    rec_scalar = logging.LogRecord("test", logging.INFO, "path", 1, "Scalar", (), None)
    rec_scalar.args = 9999  # type: ignore[assignment]
    forwarder.emit(rec_scalar)
    assert len(events) == 4

    # exc_info without exc_text
    try:
        raise ValueError("Simulated failure")
    except ValueError:
        exc_info = sys.exc_info()
    rec_exc = logging.LogRecord("test", logging.ERROR, "path", 1, "Error", (), exc_info)
    forwarder.emit(rec_exc)
    assert len(events) == 5
    assert events[-1].exc_text is not None
    assert "Simulated failure" in events[-1].exc_text

    # Dispatch raising exception triggering handleError
    def broken_dispatch(evt: StreamEvent) -> None:
        raise RuntimeError("Dispatch failed")

    broken_forwarder = LogForwarder(broken_dispatch)
    broken_forwarder.emit(rec_tuple)


def test_script_runner_properties_and_errors() -> None:
    runner = ScriptRunner()

    # script_path is None
    assert runner.script_path is None

    # reload() without script raises ScriptNotLoadedError
    with pytest.raises(ScriptNotLoadedError, match="No script file is associated"):
        runner.reload()

    # get_clip(999) raises OutputNotFoundError
    with pytest.raises(OutputNotFoundError, match="Output index 999 not found"):
        runner.get_clip(999)

    # get_clip_info(999) raises OutputNotFoundError
    with pytest.raises(OutputNotFoundError, match="Output index 999 not found"):
        runner.get_clip_info(999)

    # _extract_outputs() raises ScriptNotLoadedError when _script is None
    with pytest.raises(ScriptNotLoadedError, match="Script doesn't exist"):
        runner._extract_outputs()

    # environment raises EnvironmentNotSetError when environment is None
    runner._environment = None
    with pytest.raises(EnvironmentNotSetError, match="No environment has been passed"):
        _ = runner.environment


@pytest.mark.vpy("no-policy")
def test_script_runner_audio_node_filtering(tmp_path: Path) -> None:
    script_file = tmp_path / "audio_video.vpy"
    script_file.write_text(
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=160, height=120, length=5)\n"
        "clip.set_output(0)\n"
        "if hasattr(core.std, 'BlankAudio'):\n"
        "    audio = core.std.BlankAudio()\n"
        "    audio.set_output(1)\n",
        encoding="utf-8",
    )

    runner = ScriptRunner.from_script(script_file)
    try:
        outputs = runner.list_outputs()
        assert len(outputs) == 1
        assert outputs[0].index == 0
        assert outputs[0].info.width == 160
    finally:
        runner.close()


def test_setup_logging_no_handlers() -> None:
    root = logging.getLogger()
    orig = list(root.handlers)
    orig_level = root.level
    root.handlers.clear()
    try:
        setup_logging(level=logging.INFO)
        assert len(root.handlers) >= 1
    finally:
        root.handlers = orig
        root.setLevel(orig_level)


def test_is_loopback_address() -> None:
    assert _is_loopback_address("tcp://127.0.0.1:5555") is True
    assert _is_loopback_address("tcp://127.0.1.1:5555") is True
    assert _is_loopback_address("tcp://localhost:5555") is True
    assert _is_loopback_address("tcp://[::1]:5555") is True
    assert _is_loopback_address("ipc:///tmp/vsremote.sock") is True
    assert _is_loopback_address("inproc://vsremote") is True

    assert _is_loopback_address("tcp://0.0.0.0:5555") is False
    assert _is_loopback_address("tcp://*:5555") is False
    assert _is_loopback_address("tcp://192.168.1.100:5555") is False
    assert _is_loopback_address("tcp://10.0.0.1:5555") is False
    assert _is_loopback_address("tcp://example.com:5555") is False


@pytest.mark.vpy("initial-core")
def test_remote_backlog_and_seeking(running_server: tuple[str, int]) -> None:
    """Test remote VideoNode with custom prefetch, backlog, and non-linear seeking."""
    host, port = running_server
    address = f"tcp://{host}:{port}"

    with RemoteClient(address, compression="zstd") as client:
        # Test custom prefetch and backlog parameters
        remote_clip = client.get_output(0, prefetch=3, backlog=6)

        # Sequential access
        assert remote_clip.get_frame(0).props["TestInt"] == 0
        assert remote_clip.get_frame(1).props["TestInt"] == 10
        assert remote_clip.get_frame(2).props["TestInt"] == 20

        # Forward seek (triggering stale pruning)
        assert remote_clip.get_frame(15).props["TestInt"] == 150
        assert remote_clip.get_frame(16).props["TestInt"] == 160

        # Backward seek (triggering stale pruning)
        assert remote_clip.get_frame(5).props["TestInt"] == 50


@pytest.mark.vpy("initial-core")
def test_remote_zero_prefetch(running_server: tuple[str, int]) -> None:
    """Test remote VideoNode with prefetch disabled (prefetch=0)."""
    host, port = running_server
    address = f"tcp://{host}:{port}"

    with RemoteClient(address, compression="none") as client:
        remote_clip = client.get_output(0, prefetch=0, backlog=0)
        assert remote_clip.get_frame(0).props["TestInt"] == 0
        assert remote_clip.get_frame(10).props["TestInt"] == 100
        assert remote_clip.get_frame(19).props["TestInt"] == 190


@pytest.mark.vpy("initial-core")
def test_client_future_cancellation(running_server: tuple[str, int]) -> None:
    """Test that cancelling a frame request future marks it cancelled and sends CANCEL_REQUEST."""
    host, port = running_server
    address = f"tcp://{host}:{port}"

    with ClientTransport(address) as trans:
        fut = trans.request_frame(0, 5)
        # Cancel future immediately
        assert fut.cancel() is True
        assert fut.cancelled() is True

        # Subsequent requests still work cleanly
        fut2 = trans.request_frame(0, 6)
        header, _ = fut2.result(timeout=5.0)
        assert header.status == StatusCode.OK
        assert header.n == 6


@pytest.mark.vpy("initial-core")
def test_server_cancel_in_flight_task(server: ServerFactory) -> None:
    """Test that server cleanly cancels an in-flight slow task without crashing."""

    clip = core.std.BlankClip(width=64, height=64, length=10)

    def slow_filter(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        if n == 3:
            time.sleep(0.3)
        return f

    slow_clip = clip.std.ModifyFrame(clip, slow_filter)

    with server([slow_clip]) as (host, port), ClientTransport(f"tcp://{host}:{port}") as trans:
        # Request slow frame and cancel it mid-evaluation
        slow_fut = trans.request_frame(0, 3)
        time.sleep(0.05)
        assert slow_fut.cancel() is True

        # Ensure server is still healthy and responds to subsequent requests
        normal_fut = trans.request_frame(0, 0)
        header, _ = normal_fut.result(timeout=5.0)
        assert header.status == StatusCode.OK
        assert header.n == 0
