from __future__ import annotations

import asyncio
import ctypes
import inspect
import signal
import sys
import threading
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import vapoursynth as vs
import zmq
from vsengine.policy import Policy

from vsremote.cli import app, keygen, serve
from vsremote.client import RemoteClient, source

if TYPE_CHECKING:
    from conftest import ServerFactory

core = vs.core

HOST = "127.0.0.1"


@pytest.mark.vpy("unique-core")
def test_serve_lifecycle(server: ServerFactory, tmp_path: Path, vpy_policy: Policy) -> None:
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

    with server(script_file, compression="zstd", environment=vpy_policy) as (host, port):
        address = f"tcp://{host}:{port}"
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
            "--curve-allowed-keys",
            "key1",
            "key2",
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
    assert bound.arguments["curve_allowed_keys"] == ["key1", "key2"]


def test_serve_cli_curve_allowed_keys_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parsing curve_allowed_keys from environment variable."""
    monkeypatch.setenv("VSREMOTE_CURVE_ALLOWED_KEYS", "key_alpha key_beta")
    func, bound, _ = app.parse_args(["serve"])
    assert func == serve
    assert bound.arguments["curve_allowed_keys"] == ["key_alpha", "key_beta"]


def test_keygen_cli_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that keygen CLI command executes and outputs public/secret keys safely."""
    func, _, _ = app.parse_args(["keygen"])
    assert func == keygen
    keygen()

    problematic_pairs = [
        (b"[/pCXlglk3Wz!Zjur&YLUe1$hd4$a?Ml<s{t1fYb", b"]^RR@@Le*+@?H&{}&X&gtDt<1/IAORgTJ2n}O@a/"),
        (b"[bold red]test[cyan]", b"[/bold red]end[yellow]"),
        (b"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[", b"]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"),
    ]
    for pub, sec in problematic_pairs:
        monkeypatch.setattr(zmq, "curve_keypair", lambda p=pub, s=sec: (p, s))
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


@pytest.mark.vpy("no-core")
def test_serve_in_thread(tmp_path: Path, vpy_policy: Policy, port: int) -> None:
    """Test that serve runs cleanly in a background thread without signal handler errors."""
    script_file = tmp_path / "thread_test.vpy"
    script_file.write_text(
        "import vapoursynth as vs\ncore = vs.core\nclip = core.std.BlankClip()\nclip.set_output(0)\n",
        encoding="utf-8",
    )

    ready_event = threading.Event()
    stop_event = threading.Event()

    def run_serve() -> None:
        serve(
            script_file,
            address=f"tcp://{HOST}:{port}",
            ready_event=ready_event,
            stop_event=stop_event,
            environment=vpy_policy,
        )

    t = threading.Thread(target=run_serve, daemon=True)
    t.start()
    assert ready_event.wait(timeout=5.0)
    stop_event.set()
    t.join(timeout=3.0)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Console Ctrl Handler test")
@pytest.mark.asyncio
async def test_windows_ctrl_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Windows console control handler registration, signal dispatching, and cleanup."""
    called = False

    def on_stop() -> None:
        nonlocal called
        called = True

    handlers = list[Any]()

    def mock_set_console_ctrl_handler(handler: Any, add: bool) -> bool:
        if add:
            handlers.append(handler)
        return True

    if sys.platform == "win32":
        from vsremote.cli import _windows_ctrl_handler

        monkeypatch.setattr(ctypes.windll.kernel32, "SetConsoleCtrlHandler", mock_set_console_ctrl_handler)
        cleanup = _windows_ctrl_handler(asyncio.get_running_loop(), on_stop)
        assert len(handlers) == 1
        handler = handlers[0]

        # Simulate CTRL_C_EVENT (0)
        assert bool(handler(signal.CTRL_C_EVENT)) is True
        await asyncio.sleep(0)
        assert called

        # Ctrl+Break returns 0/False so default handler can force-terminate
        assert bool(handler(signal.CTRL_BREAK_EVENT)) is False

        # Non-exit event (e.g. 3) returns 0/False
        assert bool(handler(3)) is False

        cleanup()
