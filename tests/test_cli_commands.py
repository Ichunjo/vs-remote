from __future__ import annotations

import io
import sys
import threading
from pathlib import Path

import pytest
from vsengine.policy import Policy

from vsremote.server.cli import ClientConfig, app, info, keygen, ping, pipe, serve

HOST = "127.0.0.1"


@pytest.mark.vpy("no-core")
def test_cli_subcommands(port: int, tmp_path: Path, vpy_policy: Policy, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ping, info, and pipe CLI commands against a live running vsremote server."""
    script_file = tmp_path / "test_cli_commands.vpy"
    script_file.write_text(
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip0 = core.std.BlankClip(width=160, height=120, format=vs.YUV420P8, length=3, fpsnum=24, fpsden=1)\n"
        "clip0.set_output(0)\n",
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
        daemon=True,
    )
    server_thread.start()
    assert ready_event.wait(timeout=5.0)

    try:
        address = f"tcp://{HOST}:{port}"
        client_cfg = ClientConfig(address=address)

        # 1. Test ping command
        ping(client_cfg)

        # 2. Test info command
        info(client_cfg)

        # 3. Test pipe command with Y4M header
        buf = io.BytesIO()
        monkeypatch.setattr(sys.stdout, "buffer", buf)
        pipe(client_cfg, output=0, y4m=True, prefetch=2, environment=vpy_policy)

        output_bytes = buf.getvalue()
        assert output_bytes.startswith(b"YUV4MPEG2 C420 W160 H120 F24:1 Ip A0:0 XLENGTH=3\n")
        assert b"FRAME\n" in output_bytes

        # Verify frame count in streamed Y4M
        frame_markers = output_bytes.count(b"FRAME\n")
        assert frame_markers == 3

        # 4. Test pipe command raw without Y4M header
        raw_buf = io.BytesIO()
        monkeypatch.setattr(sys.stdout, "buffer", raw_buf)
        pipe(client_cfg, output=0, y4m=False, prefetch=2)

        raw_bytes = raw_buf.getvalue()
        assert not raw_bytes.startswith(b"YUV4MPEG2")
        # 160x120 YUV420: Y=160*120=19200, U=80*60=4800, V=80*60=4800 -> 28800 bytes per frame * 3 frames = 86400 bytes
        expected_size = (160 * 120 + 80 * 60 + 80 * 60) * 3
        assert len(raw_bytes) == expected_size

    finally:
        stop_event.set()
        server_thread.join(timeout=3.0)


def test_cli_dispatch_subcommands() -> None:
    """Test Cyclopts CLI parser dispatching to subcommands correctly."""
    func, bound, _ = app.parse_args(["ping", "tcp://127.0.0.1:5555"])
    assert func == ping
    assert bound.arguments["config"].address == "tcp://127.0.0.1:5555"

    func, bound, _ = app.parse_args(["info", "tcp://127.0.0.1:5555"])
    assert func == info
    assert bound.arguments["config"].address == "tcp://127.0.0.1:5555"

    func, bound, _ = app.parse_args(["pipe", "tcp://127.0.0.1:5555", "--output", "1", "--no-y4m"])
    assert func == pipe
    assert bound.arguments["config"].address == "tcp://127.0.0.1:5555"
    assert bound.arguments["output"] == 1
    assert bound.arguments["y4m"] is False

    func, bound, _ = app.parse_args(["keygen"])
    assert func == keygen
