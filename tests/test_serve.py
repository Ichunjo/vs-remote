from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest
import vapoursynth as vs
from vsengine.policy import Policy

from vsremote.client import RemoteClient, source
from vsremote.server.cli import app, keygen, serve

core = vs.core

HOST = "127.0.0.1"


@pytest.mark.vpy("initial-core")
def test_serve_lifecycle(port: int, tmp_path: Path, vpy_policy: Policy) -> None:
    """Test full serve lifecycle with multi-output VapourSynth script, frame fetching, and clean shutdown."""
    script_file = tmp_path / "test_serve.vpy"
    script_file.write_text(
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip0 = core.std.BlankClip(width=160, height=120, format=vs.YUV420P8, length=10)\n"
        "clip1 = core.std.BlankClip(width=80, height=60, format=vs.RGB24, length=5)\n"
        "clip0.set_output(0)\n"
        "clip1.set_output(1)\n",
        encoding="utf-8",
    )

    ready_event = threading.Event()
    stop_event = threading.Event()

    server_thread = threading.Thread(
        target=serve,
        args=(script_file,),
        kwargs={
            "address": f"tcp://{HOST}:{port}",
            "compression": "zstd",
            "ready_event": ready_event,
            "stop_event": stop_event,
            "environment": vpy_policy,
        },
        name="ServeLifecycleThread",
        daemon=True,
    )
    server_thread.start()
    assert ready_event.wait(timeout=5.0), "serve() failed to signal ready"

    try:
        address = f"tcp://{HOST}:{port}"
        with RemoteClient(address) as client:
            assert client.ping().result() is True

            outputs = client.list_outputs().result()
            assert len(outputs) == 2
            assert outputs[0].index == 0
            assert outputs[0].info.width == 160
            assert outputs[0].info.height == 120
            assert outputs[0].info.num_frames == 10

            assert outputs[1].index == 1
            assert outputs[1].info.width == 80
            assert outputs[1].info.height == 60
            assert outputs[1].info.num_frames == 5

        # Test proxy VideoNode via source()
        proxy0 = source(address, output=0)
        assert proxy0.width == 160
        assert proxy0.num_frames == 10
        frame0 = proxy0.get_frame(0)
        assert frame0.width == 160

        proxy1 = source(address, output=1)
        assert proxy1.width == 80
        assert proxy1.format.id == vs.RGB24
        frame1 = proxy1.get_frame(0)
        assert frame1.width == 80
    finally:
        stop_event.set()
        server_thread.join(timeout=3.0)
        assert not server_thread.is_alive(), "serve thread did not terminate cleanly"


def test_serve_file_not_found() -> None:
    """Test that serve raises FileNotFoundError when passed a nonexistent script."""
    with pytest.raises(FileNotFoundError, match="Script not found"):
        serve("nonexistent_path_xyz123.vpy")


def test_serve_cli_command_dispatch(port: int, tmp_path: Path) -> None:
    """Test Cyclopts CLI command dispatch and parameter parsing for the serve command."""
    script_file = tmp_path / "cli_test.vpy"
    script_file.write_text("import vapoursynth as vs\n", encoding="utf-8")

    func, bound, _ = app.parse_args(
        [
            "serve",
            str(script_file),
            "--address",
            f"tcp://{HOST}:{port}",
            "--compression",
            "none",
            "--max-workers",
            "4",
            "--allow-eval",
            "--auth-token",
            "test_secret_token",
            "--curve",
            "--curve-secret-key",
            "curve_sec_key_123",
        ]
    )

    assert func == serve
    assert bound.arguments["script_path"] == str(script_file)
    assert bound.arguments["address"] == f"tcp://{HOST}:{port}"
    assert bound.arguments["compression"] == "none"
    assert bound.arguments["max_workers"] == 4
    assert bound.arguments["allow_eval"] is True
    assert bound.arguments["auth_token"] == "test_secret_token"
    assert bound.arguments["curve"] is True
    assert bound.arguments["curve_secret_key"] == "curve_sec_key_123"


def test_keygen_cli_command() -> None:
    """Test that keygen CLI command executes and outputs public/secret keys."""
    func, _, _ = app.parse_args(["keygen"])
    assert func == keygen
    keygen()


@pytest.mark.vpy("no-policy")
def test_serve_keyboard_interrupt(port: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that serve handles KeyboardInterrupt cleanly."""
    script_file = tmp_path / "test_interrupt.vpy"
    script_file.write_text(
        "import vapoursynth as vs\ncore = vs.core\nclip = core.std.BlankClip()\nclip.set_output(0)\n",
        encoding="utf-8",
    )

    original_run_until_complete = asyncio.BaseEventLoop.run_until_complete

    def mock_run_until_complete(self: asyncio.BaseEventLoop, future: Awaitable[Any]) -> object:
        coro = (
            future.get_coro() if isinstance(future, asyncio.Task) else future if inspect.iscoroutine(future) else None
        )
        if coro and (cr_code := getattr(coro, "cr_code", None)) and cr_code.co_name == "run":
            if inspect.iscoroutine(future):
                future.close()
            raise KeyboardInterrupt
        return original_run_until_complete(self, future)

    monkeypatch.setattr(asyncio.BaseEventLoop, "run_until_complete", mock_run_until_complete)

    serve(script_file, address=f"tcp://{HOST}:{port}")
