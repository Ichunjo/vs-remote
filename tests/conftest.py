from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Callable, Generator

import pytest
import vapoursynth as vs

from vsremote.server.daemon import ServerDaemon
from vsremote.server.runner import ScriptRunner

core = vs.core

HOST = "127.0.0.1"


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> dict[str, Callable[[], asyncio.AbstractEventLoop]]:
    return {"stdlib": asyncio.new_event_loop, "custom": asyncio.SelectorEventLoop}


@pytest.fixture
def port() -> int:
    """Find an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return int(s.getsockname()[1])


@pytest.fixture
def test_clip() -> vs.VideoNode:
    """Create a sample YUV420P8 test clip with distinctive plane patterns and frame properties."""
    clip = core.std.BlankClip(width=128, height=128, format=vs.YUV420P8, length=20, fpsnum=24, fpsden=1)

    def _set_props(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f.copy()
        fout.props["_SARNum"] = 1
        fout.props["_SARDen"] = 1
        fout.props["_Matrix"] = 1
        fout.props["TestString"] = f"Frame_{n}"
        fout.props["TestInt"] = n * 10
        fout.props["TestFloat"] = n * 0.5
        fout.props["TestBytes"] = b"raw_data_bytes"
        return fout

    return core.std.ModifyFrame(clip, clip, _set_props)


@pytest.fixture
def test_clip_10bit() -> vs.VideoNode:
    """Create a sample 10-bit YUV420P10 test clip."""
    return core.std.BlankClip(width=64, height=64, format=vs.YUV420P10, length=10, fpsnum=30, fpsden=1)


@pytest.fixture
def running_server(
    port: int, test_clip: vs.VideoNode, test_clip_10bit: vs.VideoNode
) -> Generator[tuple[str, int], None, None]:
    """Run a temporary ServerDaemon in a background thread for testing."""
    runner = ScriptRunner.from_clips([test_clip, test_clip_10bit])
    daemon = ServerDaemon(runner, address=f"tcp://{HOST}:{port}", compression="zstd", max_workers=4)

    ready_event = threading.Event()
    loop = asyncio.SelectorEventLoop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(daemon.start(ready_event=ready_event))

    thread = threading.Thread(target=_run, name="TestServerThread", daemon=True)
    thread.start()
    assert ready_event.wait(timeout=5.0), "Server daemon failed to start"

    yield HOST, port

    fut = asyncio.run_coroutine_threadsafe(daemon.stop(), loop)
    fut.result(timeout=5.0)

    thread.join(timeout=2.0)
    if not loop.is_closed():
        loop.close()
