from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from vsengine.policy import Policy

from vsremote.server.cli import ClientConfig, app, info, keygen, ping, pipe

if TYPE_CHECKING:
    from conftest import ServerFactory


@pytest.mark.vpy("no-core")
def test_cli_subcommands(
    server: ServerFactory,
    tmp_path: Path,
    vpy_policy: Policy,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_file = tmp_path / "test_cli_commands.vpy"
    script_file.write_text(
        "import vapoursynth as vs\n"
        "core = vs.core\n"
        "clip0 = core.std.BlankClip(width=160, height=120, format=vs.YUV420P8, length=3, fpsnum=24, fpsden=1)\n"
        "clip0.set_output(0)\n",
        encoding="utf-8",
    )

    with server(script_file, compression="zstd", environment=vpy_policy) as (host, port):
        address = f"tcp://{host}:{port}"
        client_cfg = ClientConfig(address=address)

        # Clear any startup logs before testing command outputs
        capsys.readouterr()

        # Test ping command
        ping(client_cfg)
        ping_err = capsys.readouterr().err
        assert "OK" in ping_err
        assert "Successfully connected to" in ping_err
        assert address in ping_err
        assert "RTT:" in ping_err

        # Test info command
        info(client_cfg)
        info_err = capsys.readouterr().err
        assert f"Remote Outputs for {address}" in info_err
        assert "Index" in info_err
        assert "Name" in info_err
        assert "Resolution" in info_err
        assert "FPS" in info_err
        assert "Format" in info_err
        assert "Frames" in info_err
        assert "0" in info_err
        assert "Output 0" in info_err
        assert "160x120" in info_err
        assert "24.000 (24/1)" in info_err
        assert "YUV420P8" in info_err
        assert "3" in info_err

        # Test pipe command with Y4M header
        buf = io.BytesIO()
        monkeypatch.setattr(sys.stdout, "buffer", buf)
        pipe(client_cfg, output=0, y4m=True, prefetch=2, environment=vpy_policy)

        output_bytes = buf.getvalue()
        assert output_bytes.startswith(b"YUV4MPEG2 C420 W160 H120 F24:1 Ip A0:0 XLENGTH=3\n")
        assert b"FRAME\n" in output_bytes

        # Verify frame count in streamed Y4M
        frame_markers = output_bytes.count(b"FRAME\n")
        assert frame_markers == 3

        # Test pipe command raw without Y4M header
        raw_buf = io.BytesIO()
        monkeypatch.setattr(sys.stdout, "buffer", raw_buf)
        pipe(client_cfg, output=0, y4m=False, prefetch=2)

        raw_bytes = raw_buf.getvalue()
        assert not raw_bytes.startswith(b"YUV4MPEG2")
        # 160x120 YUV420: Y=160*120=19200, U=80*60=4800, V=80*60=4800 -> 28800 bytes per frame * 3 frames = 86400 bytes
        expected_size = (160 * 120 + 80 * 60 + 80 * 60) * 3
        assert len(raw_bytes) == expected_size


def test_cli_dispatch_subcommands() -> None:
    """Test Cyclopts CLI parser dispatching to subcommands correctly."""
    func, bound, _ = app.parse_args(["ping", "tcp://127.0.0.1:5555"])
    assert func == ping
    assert bound.arguments["config"].address == "tcp://127.0.0.1:5555"

    func, bound, _ = app.parse_args(["info", "tcp://127.0.0.1:5555"])
    assert func == info
    assert bound.arguments["config"].address == "tcp://127.0.0.1:5555"

    func, bound, _ = app.parse_args(["pipe", "tcp://127.0.0.1:5555", "--output", "1"])
    assert func == pipe
    assert bound.arguments["config"].address == "tcp://127.0.0.1:5555"
    assert bound.arguments["output"] == 1

    func, bound, _ = app.parse_args(["keygen"])
    assert func == keygen
