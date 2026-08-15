from __future__ import annotations

import asyncio
import io
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import override

import pytest
import vapoursynth as vs
import vsengine.video
import zmq
from vsengine.policy import ManagedEnvironment, Policy

from vsremote.client import ClientTransport, RemoteClient, source
from vsremote.protocol.constants import Command, StatusCode
from vsremote.server import RemotePolicy, ScriptRunner, ServerDaemon

HOST = "127.0.0.1"


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


@pytest.mark.asyncio
@pytest.mark.vpy("initial-core")
async def test_async_client_operations(running_server: tuple[str, int]) -> None:
    host, port = running_server
    async with RemoteClient(f"tcp://{host}:{port}") as client:
        # 1. Async ping
        assert (await client.ping()) is True

        # 2. Async list outputs
        outputs = await client.list_outputs()
        assert len(outputs) == 2
        assert outputs[0].index == 0

        # 3. Async get clip info
        info = await client.get_clip_info(0)
        assert info.width == 128
        assert info.num_frames == 20

        # 4. Async request frame
        header, planes = await client.request_frame(0, 5)
        assert header.status == StatusCode.OK
        assert header.n == 5
        assert len(planes) == 3


@pytest.mark.vpy("initial-core")
def test_remote_source_frame_rendering(
    running_server: tuple[str, int],
    test_clip: vs.VideoNode,
) -> None:
    host, port = running_server
    address = f"tcp://{host}:{port}"

    # Connect proxy node using source() helper
    remote_clip = source(address, output=0, compression="zstd")

    assert remote_clip.width == test_clip.width
    assert remote_clip.height == test_clip.height
    assert remote_clip.num_frames == test_clip.num_frames
    assert remote_clip.format.id == test_clip.format.id

    # Test individual frames and verify bit-for-bit pixel matching and props
    for frame_num in [0, 5, 10, 19]:
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
def test_remote_source_10bit(running_server: tuple[str, int], test_clip_10bit: vs.VideoNode) -> None:
    host, port = running_server
    address = f"tcp://{host}:{port}"

    with RemoteClient(address, compression="none") as client:
        remote_clip = client.get_output(1)
        assert remote_clip.format.id == vs.YUV420P10
        assert remote_clip.width == 64
        assert remote_clip.num_frames == 10

        frame = remote_clip.get_frame(0)
        assert frame.format.id == vs.YUV420P10


@pytest.mark.vpy("initial-core")
def test_concurrent_frame_requests(running_server: tuple[str, int]) -> None:
    host, port = running_server
    address = f"tcp://{host}:{port}"

    with RemoteClient(address, compression="zstd") as client:
        remote_clip = client.get_output(0)

        # Concurrently request 20 frames across 8 worker threads
        def _fetch(n: int) -> int:
            f = remote_clip.get_frame(n % 20)
            return f.props["TestInt"]  # type: ignore[return-value]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(_fetch, range(40)))

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
        assert "out of bounds" in header.error_message

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


@pytest.mark.vpy("initial-core")
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

    env = vpy_env_factory()
    runner = ScriptRunner.from_script(script_file, environment=env)
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

    # 1. Unstarted transport should reject requests
    with pytest.raises(RuntimeError, match="Transport is not started"):
        transport.send_request(Command.PING).result()

    with pytest.raises(RuntimeError, match="Transport is not started"):
        transport.list_outputs().result()

    # 2. Start transport and verify communication
    transport.start()
    assert transport.ping().result() is True
    assert len(transport.list_outputs().result()) == 2

    # 3. Multiple start calls should be idempotent
    transport.start()
    assert transport.ping().result() is True

    # 4. Close transport and verify idempotency
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


@pytest.mark.asyncio
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
@pytest.mark.vpy("initial-core")
async def test_stream_structured_log_records(port: int, tmp_path: Path, vpy_policy: Policy) -> None:
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

    runner = ScriptRunner.from_script(script_file, environment=vpy_policy)
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}", compression="zstd")
    ready_event = asyncio.Event()
    daemon_task = asyncio.create_task(daemon.start(ready_event=ready_event))
    await ready_event.wait()

    collector = LogCollector()
    target_logger = logging.getLogger("test_custom_logger")
    target_logger.addHandler(collector)
    try:
        async with RemoteClient(f"tcp://{HOST}:{port}") as client:
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
        await daemon.stop()
        await daemon_task


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_stream_vapoursynth_log_message(port: int, tmp_path: Path, vpy_policy: Policy) -> None:
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

    runner = ScriptRunner.from_script(script_file, environment=vpy_policy)
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}", compression="zstd")
    ready_event = asyncio.Event()
    daemon_task = asyncio.create_task(daemon.start(ready_event=ready_event))
    await ready_event.wait()

    collector = LogCollector()
    vs_logger = logging.getLogger("vapoursynth")
    vs_logger.addHandler(collector)
    try:
        async with RemoteClient(f"tcp://{HOST}:{port}") as client:
            assert (await client.ping()) is True
            await asyncio.sleep(0.1)

        vs_records = [
            r for r in collector.records if r.name == "vapoursynth" and "VS core warning raised" in r.getMessage()
        ]
        assert len(vs_records) >= 1
        assert vs_records[0].levelno == logging.WARNING
    finally:
        vs_logger.removeHandler(collector)
        await daemon.stop()
        await daemon_task


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_stream_stdout_and_stderr(port: int, tmp_path: Path, vpy_policy: Policy) -> None:
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

    runner = ScriptRunner.from_script(script_file, environment=vpy_policy)
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}", compression="zstd")
    ready_event = asyncio.Event()
    daemon_task = asyncio.create_task(daemon.start(ready_event=ready_event))
    await ready_event.wait()

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        async with RemoteClient(
            f"tcp://{HOST}:{port}",
            stdout=stdout_buf,
            stderr=stderr_buf,
        ) as client:
            assert (await client.ping()) is True
            await asyncio.sleep(0.1)

        assert "Hello from remote stdout!" in stdout_buf.getvalue()
        assert "Direct stderr message" in stderr_buf.getvalue()
    finally:
        await daemon.stop()
        await daemon_task


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("no-policy")
async def test_stream_logs_not_duplicated_in_stderr(port: int, tmp_path: Path) -> None:
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

    collector = LogCollector()
    vs_logger = logging.getLogger("vapoursynth")
    vs_logger.addHandler(collector)

    try:
        runner = ScriptRunner.from_script(script_file)
        daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}", compression="zstd")
        ready_event = asyncio.Event()
        daemon_task = asyncio.create_task(daemon.start(ready_event=ready_event))
        await ready_event.wait()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            async with RemoteClient(f"tcp://{HOST}:{port}", stdout=stdout_buf, stderr=stderr_buf) as client:
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
            await daemon.stop()
            await daemon_task
    finally:
        vs_logger.removeHandler(collector)


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_forward_logs_disabled(port: int, tmp_path: Path, vpy_policy: Policy) -> None:
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

    runner = ScriptRunner.from_script(script_file, environment=vpy_policy)
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}", compression="zstd")
    ready_event = asyncio.Event()
    daemon_task = asyncio.create_task(daemon.start(ready_event=ready_event))
    await ready_event.wait()

    collector = LogCollector()
    target_logger = logging.getLogger("test_disabled_logger")
    target_logger.addHandler(collector)
    try:
        async with RemoteClient(f"tcp://{HOST}:{port}", forward_logs=False) as client:
            assert (await client.ping()) is True
            await asyncio.sleep(0.1)

        matching = [r for r in collector.records if r.name == "test_disabled_logger"]
        assert len(matching) == 0
    finally:
        target_logger.removeHandler(collector)
        await daemon.stop()
        await daemon_task


@pytest.mark.vpy("no-policy")
def test_remote_policy_logger_interception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RemotePolicy intercepts core.log_message via api.set_logger."""
    logs = list[tuple[int, str]]()
    policy = RemotePolicy()
    policy.register()

    monkeypatch.setattr(
        logging.getLogger("vapoursynth"),
        "log",
        lambda lvl, msg, *args, **kwargs: logs.append((lvl, msg)),
    )

    try:
        env = policy.new_environment()
        try:
            with env.use():
                env.core.log_message(vs.MESSAGE_TYPE_WARNING, "Direct core log from RemotePolicy")
            assert len(logs) == 1
            assert logs[0][0] == logging.WARNING
            assert logs[0][1] == "Direct core log from RemotePolicy"
        finally:
            env.dispose()
    finally:
        policy.unregister()


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_client_reload_script(tmp_path: Path, port: int, vpy_policy: Policy) -> None:
    """Test client triggering a server script reload after modifying the script on disk."""
    script_file = tmp_path / "reload_test.vpy"
    script_file.write_text(
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip = core.std.BlankClip(width=100, height=80, format=vs.YUV420P8, length=5)\n"
        "clip.set_output(0)\n",
        encoding="utf-8",
    )

    runner = ScriptRunner.from_script(script_file, environment=vpy_policy)
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}")
    ready_event = asyncio.Event()

    daemon_task = asyncio.create_task(daemon.start(ready_event=ready_event))
    await ready_event.wait()

    try:
        async with RemoteClient(f"tcp://{HOST}:{port}") as client:
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
    finally:
        await daemon.stop()
        await daemon_task


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_client_load_code_and_error_handling(port: int, vpy_policy: Policy) -> None:
    """Test dynamic code execution on the server via client.load_code and error resilience."""
    runner = ScriptRunner()
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}", allow_eval=True)
    ready_event = asyncio.Event()

    # Pre-configure policy on runner
    runner._ensure_policy(vpy_policy)

    daemon_task = asyncio.create_task(daemon.start(ready_event=ready_event))
    await ready_event.wait()

    try:
        async with RemoteClient(f"tcp://{HOST}:{port}") as client:
            # 1. Execute valid code string
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

            # 2. Execute invalid code - should return error without crashing server
            with pytest.raises(RuntimeError, match="Failed to load code"):
                await client.load_code("this is not valid python code !!!")

            # 3. Server remains healthy and can execute subsequent code
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
    finally:
        await daemon.stop()
        await daemon_task


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_client_load_script_switch(tmp_path: Path, port: int, vpy_policy: Policy) -> None:
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

    runner = ScriptRunner.from_script(script1, environment=vpy_policy)
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}", allow_eval=True)
    ready_event = asyncio.Event()

    daemon_task = asyncio.create_task(daemon.start(ready_event=ready_event))
    await ready_event.wait()

    try:
        async with RemoteClient(f"tcp://{HOST}:{port}") as client:
            outputs1 = await client.list_outputs()
            assert outputs1[0].info.width == 120

            # Switch script
            outputs2 = await client.load_script(script2)
            assert outputs2[0].info.width == 240
            assert outputs2[0].info.num_frames == 8

            # Try loading nonexistent script
            with pytest.raises(RuntimeError, match="Failed to load script"):
                await client.load_script(tmp_path / "nonexistent.vpy")
    finally:
        await daemon.stop()
        await daemon_task


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_permission_denied_when_allow_eval_disabled(port: int, vpy_policy: Policy, tmp_path: Path) -> None:
    """Test that dynamic code and script execution is blocked when allow_eval is False (default)."""
    script = tmp_path / "static_script.vpy"
    script.write_text(
        "import vapoursynth as vs\ncore = vs.core\n"
        "core.std.BlankClip(width=100, height=100, length=10).set_output(0)\n",
        encoding="utf-8",
    )

    runner = ScriptRunner.from_script(script, environment=vpy_policy)
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}", allow_eval=False)
    ready_event = asyncio.Event()

    daemon_task = asyncio.create_task(daemon.start(ready_event=ready_event))
    await ready_event.wait()

    try:
        async with RemoteClient(f"tcp://{HOST}:{port}") as client:
            # Listing outputs and fetching frames should work normally
            outputs = await client.list_outputs()
            assert len(outputs) == 1

            # Dynamic code loading must fail with permission denied error
            with pytest.raises(RuntimeError, match="Dynamic code evaluation is disabled on this server"):
                await client.load_code("core.std.BlankClip().set_output(0)")

            # Dynamic script switching must fail with permission denied error
            with pytest.raises(RuntimeError, match="Dynamic script loading is disabled on this server"):
                await client.load_script(script)
    finally:
        await daemon.stop()
        await daemon_task


@pytest.mark.asyncio(loop_factories=["custom"])
@pytest.mark.vpy("initial-core")
async def test_auth_token_security(port: int, test_clip: vs.VideoNode) -> None:
    """Test authentication token enforcement on ServerDaemon."""
    token = "secret_access_token_xyz"
    runner = ScriptRunner.from_clips([test_clip])
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}", auth_token=token)
    ready_event = asyncio.Event()

    daemon_task = asyncio.create_task(daemon.start(ready_event=ready_event))
    await ready_event.wait()

    try:
        # 1. Connecting without token -> should fail operations with unauthorized / ConnectionReset
        async with RemoteClient(f"tcp://{HOST}:{port}", auth_token=None) as client_unauthed:
            assert (await client_unauthed.ping()) is False
            with pytest.raises(RuntimeError, match="Failed to list outputs"):
                await client_unauthed.list_outputs()

        # 2. Connecting with wrong token -> should fail
        async with RemoteClient(f"tcp://{HOST}:{port}", auth_token="wrong_token_abc") as client_wrong:
            assert (await client_wrong.ping()) is False
            with pytest.raises(RuntimeError, match="Failed to list outputs"):
                await client_wrong.list_outputs()

        # 3. Connecting with valid token -> should succeed
        async with RemoteClient(f"tcp://{HOST}:{port}", auth_token=token) as client_authed:
            assert (await client_authed.ping()) is True
            outputs = await client_authed.list_outputs()
            assert len(outputs) == 1
            header, _ = await client_authed.request_frame(0, 0)
            assert header.status == StatusCode.OK
    finally:
        await daemon.stop()
        await daemon_task


@pytest.mark.vpy("initial-core")
def test_curvezmq_end_to_end_encryption(port: int, test_clip: vs.VideoNode) -> None:
    """Test ZeroMQ CurveZMQ end-to-end encrypted connection and frame streaming."""
    # Generate server Curve keypair
    server_public, server_secret = zmq.curve_keypair()

    runner = ScriptRunner.from_clips([test_clip])
    daemon = ServerDaemon(
        runner,
        address=f"tcp://{HOST}:{port}",
        curve_secret_key=server_secret,
        curve_public_key=server_public,
    )
    ready_event = threading.Event()
    loop = asyncio.SelectorEventLoop()

    thread = threading.Thread(
        target=lambda: loop.run_until_complete(daemon.start(ready_event=ready_event)),
        name="CurveServerThread",
        daemon=True,
    )
    thread.start()
    assert ready_event.wait(timeout=5.0)

    try:
        address = f"tcp://{HOST}:{port}"
        # Connect client using server's public key (client ephemeral keypair is generated automatically)
        with RemoteClient(address, curve_server_key=server_public) as client:
            assert client.ping().result() is True
            outputs = client.list_outputs().result()
            assert len(outputs) == 1

            # Fetch encrypted frame proxy via source()
            proxy = source(address, output=0, curve_server_key=server_public)
            frame = proxy.get_frame(0)
            assert frame.width == test_clip.width
            assert bytes(frame[0]) == bytes(test_clip.get_frame(0)[0])
    finally:
        fut = asyncio.run_coroutine_threadsafe(daemon.stop(), loop)
        fut.result(timeout=5.0)
        thread.join(timeout=2.0)
        if not loop.is_closed():
            loop.close()


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
def test_remote_traceback_propagation(port: int) -> None:
    """Test that server evaluation errors include formatted remote traceback."""
    core = vs.core
    # Create a clip that raises an error on frame 1
    src = core.std.BlankClip(width=64, height=64, length=5)

    def faulty_filter(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        if n == 1:
            raise ValueError("Intentional error inside custom filter")
        return f

    faulty_clip = core.std.ModifyFrame(src, src, faulty_filter)
    runner = ScriptRunner.from_clips([faulty_clip])
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}")
    ready_event = threading.Event()
    loop = asyncio.SelectorEventLoop()

    thread = threading.Thread(
        target=lambda: loop.run_until_complete(daemon.start(ready_event=ready_event)),
        daemon=True,
    )
    thread.start()
    assert ready_event.wait(timeout=5.0)

    try:
        with RemoteClient(f"tcp://{HOST}:{port}") as client:
            clip = client.get_output(0)
            # Frame 0 succeeds
            assert clip.get_frame(0).width == 64
            # Frame 1 fails and includes remote traceback
            with pytest.raises((vs.Error, RuntimeError)) as exc_info:
                clip.get_frame(1)
            err_msg = str(exc_info.value)
            assert "Intentional error inside custom filter" in err_msg
            assert "[Remote Traceback]" in err_msg
    finally:
        fut = asyncio.run_coroutine_threadsafe(daemon.stop(), loop)
        fut.result(timeout=5.0)
        thread.join(timeout=2.0)
        if not loop.is_closed():
            loop.close()


@pytest.mark.vpy("initial-core")
def test_strided_copy_non_aligned(port: int) -> None:
    """Test that clips with unaligned dimensions requiring strided line-by-line copies work perfectly."""
    core = vs.core
    # Odd width on RGB24 creates stride > row_size on standard aligned blank clips
    unaligned_clip = core.std.BlankClip(width=157, height=93, format=vs.RGB24, length=5, color=[120, 80, 200])
    runner = ScriptRunner.from_clips([unaligned_clip])
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}")
    ready_event = threading.Event()
    loop = asyncio.SelectorEventLoop()

    thread = threading.Thread(
        target=lambda: loop.run_until_complete(daemon.start(ready_event=ready_event)),
        daemon=True,
    )
    thread.start()
    assert ready_event.wait(timeout=5.0)

    try:
        with RemoteClient(f"tcp://{HOST}:{port}", compression="none") as client:
            clip = client.get_output(0)
            assert clip.width == 157
            assert clip.height == 93
            frame = clip.get_frame(0)
            orig_frame = unaligned_clip.get_frame(0)
            for p in range(3):
                assert bytes(frame[p]) == bytes(orig_frame[p])
    finally:
        fut = asyncio.run_coroutine_threadsafe(daemon.stop(), loop)
        fut.result(timeout=5.0)
        thread.join(timeout=2.0)
        if not loop.is_closed():
            loop.close()
