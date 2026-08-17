from __future__ import annotations

import asyncio
import os
import socket
import threading
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Any, Protocol

import pytest
import vapoursynth as vs
from vsengine.policy import Policy

from vsremote.cli import serve
from vsremote.protocol import Compression
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
    return core.std.BlankClip(width=64, height=64, format=vs.YUV420P10, length=10, fpsnum=30, fpsden=1)


class ServerContext:
    def __init__(
        self,
        target: str | os.PathLike[str] | ScriptRunner | Sequence[vs.VideoNode] | None = None,
        *,
        port: int,
        default_clips: Sequence[vs.VideoNode] | None = None,
        default_clips_factory: Callable[[], Sequence[vs.VideoNode]] | None = None,
        compression: Compression = "zstd",
        allow_eval: bool = False,
        auth_token: str | None = None,
        environment: Policy | None = None,
        **daemon_kwargs: Any,
    ) -> None:
        self.target = target
        self.port = port
        self.default_clips = list(default_clips) if default_clips else []
        self.default_clips_factory = default_clips_factory
        self.compression: Compression = compression
        self.allow_eval = allow_eval
        self.auth_token = auth_token
        self.environment = environment
        self.daemon_kwargs = daemon_kwargs

        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sync_daemon: ServerDaemon | None = None

        self._async_daemon: ServerDaemon | None = None
        self._async_task: asyncio.Task[None] | None = None
        self._runner: ScriptRunner | None = None

    def __enter__(self) -> tuple[str, int]:
        ready_event = threading.Event()
        self._stop_event = threading.Event()

        if isinstance(self.target, (str, os.PathLike)):
            self._thread = threading.Thread(
                target=serve,
                args=(self.target,),
                kwargs={
                    "address": f"tcp://{HOST}:{self.port}",
                    "compression": self.compression,
                    "allow_eval": self.allow_eval,
                    "auth_token": self.auth_token,
                    "ready_event": ready_event,
                    "stop_event": self._stop_event,
                    "environment": self.environment,
                    **self.daemon_kwargs,
                },
                daemon=True,
            )
        else:
            self._sync_daemon = ServerDaemon(
                self._resolve_runner(),
                address=f"tcp://{HOST}:{self.port}",
                compression=self.compression,
                allow_eval=self.allow_eval,
                auth_token=self.auth_token,
                **self.daemon_kwargs,
            )
            self._loop = asyncio.SelectorEventLoop()

            def run() -> None:
                assert self._loop
                assert self._sync_daemon
                asyncio.set_event_loop(self._loop)
                self._loop.run_until_complete(self._sync_daemon.start(ready_event=ready_event))

            self._thread = threading.Thread(target=run, name="ServerContextThread", daemon=True)

        self._thread.start()
        assert ready_event.wait(timeout=5.0), "Server daemon failed to start"
        return HOST, self.port

    def __exit__(self, *_: object) -> None:
        if self._stop_event:
            self._stop_event.set()

        if self._sync_daemon and self._loop and not self._loop.is_closed():
            fut = asyncio.run_coroutine_threadsafe(self._sync_daemon.stop(), self._loop)
            fut.result(timeout=5.0)

        if self._thread:
            self._thread.join(timeout=3.0)

        if self._loop and not self._loop.is_closed():
            self._loop.close()

        if self._runner:
            self._runner.close()

    async def __aenter__(self) -> tuple[str, int]:
        self._async_daemon = ServerDaemon(
            self._resolve_runner(),
            address=f"tcp://{HOST}:{self.port}",
            compression=self.compression,
            allow_eval=self.allow_eval,
            auth_token=self.auth_token,
            **self.daemon_kwargs,
        )
        ready_event = asyncio.Event()
        self._async_task = asyncio.create_task(self._async_daemon.start(ready_event=ready_event))
        await ready_event.wait()
        return HOST, self.port

    async def __aexit__(self, *_: object) -> None:
        if self._async_daemon:
            await self._async_daemon.stop()
        if self._async_task:
            await self._async_task
        if self._runner:
            self._runner.close()

    def _resolve_runner(self) -> ScriptRunner:
        if self._runner is not None:
            return self._runner

        match self.target:
            case ScriptRunner():
                self._runner = self.target
            case str() | os.PathLike():
                self._runner = ScriptRunner.from_script(self.target, environment=self.environment)
            case Sequence():
                self._runner = ScriptRunner.from_clips(self.target)  # type: ignore[arg-type]
            case _ if self.default_clips_factory:
                self._runner = ScriptRunner.from_clips(self.default_clips_factory())
            case _:
                self._runner = ScriptRunner.from_clips(self.default_clips)
        return self._runner


class ServerFactory(Protocol):
    def __call__(
        self,
        target: Path | str | ScriptRunner | Sequence[vs.VideoNode] | None = None,
        *,
        compression: Compression = "zstd",
        allow_eval: bool = False,
        auth_token: str | None = None,
        environment: Policy | None = None,
        **daemon_kwargs: Any,
    ) -> ServerContext: ...


@pytest.fixture
def server(port: int, request: pytest.FixtureRequest) -> ServerFactory:
    def factory(
        target: Path | str | ScriptRunner | Sequence[vs.VideoNode] | None = None,
        *,
        compression: Compression = "zstd",
        allow_eval: bool = False,
        auth_token: str | None = None,
        environment: Policy | None = None,
        **daemon_kwargs: Any,
    ) -> ServerContext:
        return ServerContext(
            target,
            port=port,
            default_clips_factory=lambda: [
                request.getfixturevalue("test_clip"),
                request.getfixturevalue("test_clip_10bit"),
            ],
            compression=compression,
            allow_eval=allow_eval,
            auth_token=auth_token,
            environment=environment,
            **daemon_kwargs,
        )

    return factory


@pytest.fixture
def running_server(server: ServerFactory) -> Generator[tuple[str, int], None, None]:
    """Run a temporary ServerDaemon in a background thread for testing."""
    with server(compression="zstd", max_workers=4) as server_info:
        yield server_info
