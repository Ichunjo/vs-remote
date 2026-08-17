from __future__ import annotations

import io
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import vapoursynth as vs
from vsengine.futures import UnifiedFuture
from vsengine.policy import ManagedEnvironment, Policy

from vsremote.cli import ClientConfig, _get_y4m_header, app, info, keygen, ping, pipe, serve
from vsremote.client import ClientTransport
from vsremote.protocol import ClipInfo, FrameHeader, PlaneInfo, StatusCode

if TYPE_CHECKING:
    from conftest import ServerFactory

core = vs.core


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
        "\n"
        "clip0 = vs.core.std.BlankClip(width=160, height=120, format=vs.YUV420P8, length=3, fpsnum=24, fpsden=1)\n"
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


def test_ping_command_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test ping command failure exiting with SystemExit(1)."""
    bad_cfg = ClientConfig(address="tcp://127.0.0.1:1")
    mock_fut = UnifiedFuture[bool]()
    mock_fut.set_result(False)

    monkeypatch.setattr(ClientTransport, "ping", lambda self: mock_fut)

    with pytest.raises(SystemExit) as exc_info:
        ping(bad_cfg)
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "FAIL" in err
    assert "Ping failed for" in err


def test_pipe_command_frame_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pipe command failure when a frame fetch returns an error status."""
    cfg = ClientConfig(address="tcp://127.0.0.1:5555")

    mock_info_fut = UnifiedFuture[ClipInfo]()
    info_data = ClipInfo(
        width=100,
        height=100,
        fps_num=24,
        fps_den=1,
        num_frames=1,
        format_id=int(vs.YUV420P8),
        format_name="YUV420P8",
        num_planes=3,
        bytes_per_sample=1,
        subsampling_w=1,
        subsampling_h=1,
        planes=[PlaneInfo(100, 100, 1, 10000), PlaneInfo(50, 50, 1, 2500), PlaneInfo(50, 50, 1, 2500)],
    )
    mock_info_fut.set_result(info_data)

    mock_frame_fut = UnifiedFuture[tuple[FrameHeader, list[bytes]]]()
    mock_frame_fut.set_result(
        (
            FrameHeader(
                status=StatusCode.ERROR,
                request_id=1,
                n=0,
                output_index=0,
                compression="zstd",
                error_message="Simulated frame rendering error",
            ),
            [],
        )
    )

    monkeypatch.setattr(ClientTransport, "get_clip_info", lambda self, output: mock_info_fut)
    monkeypatch.setattr(ClientTransport, "request_frame", lambda self, output, n, compression="zstd": mock_frame_fut)

    with pytest.raises(RuntimeError, match="Failed to fetch frame 0: Simulated frame rendering error"):
        pipe(cfg, output=0)


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


@pytest.mark.vpy("initial-core")
def test_get_y4m_header_subsamplings(vpy_policy: Policy, vpy_env_factory: Callable[[], ManagedEnvironment]) -> None:
    # 1 plane: Mono GRAY8 & GRAY10
    clip_gray8 = core.std.BlankClip(width=160, height=120, format=vs.GRAY8, length=5)
    info_gray8 = ClipInfo.from_clip(clip_gray8)
    h_gray8 = _get_y4m_header(info_gray8, vpy_policy)
    assert h_gray8.startswith(b"YUV4MPEG2 Cmono W160 H120")

    clip_gray10 = core.std.BlankClip(width=160, height=120, format=vs.GRAY10, length=5)
    info_gray10 = ClipInfo.from_clip(clip_gray10)
    h_gray10 = _get_y4m_header(info_gray10, vpy_policy)
    assert h_gray10.startswith(b"YUV4MPEG2 Cmonop10 W160 H120")

    # 3 planes: 420
    clip_420 = core.std.BlankClip(width=160, height=120, format=vs.YUV420P8, length=5)
    info_420 = ClipInfo.from_clip(clip_420)
    assert _get_y4m_header(info_420, vpy_policy).startswith(b"YUV4MPEG2 C420 ")

    # 3 planes: 422
    clip_422 = core.std.BlankClip(width=160, height=120, format=vs.YUV422P8, length=5)
    info_422 = ClipInfo.from_clip(clip_422)
    assert _get_y4m_header(info_422, vpy_policy).startswith(b"YUV4MPEG2 C422 ")

    # 3 planes: 444
    clip_444 = core.std.BlankClip(width=160, height=120, format=vs.YUV444P8, length=5)
    info_444 = ClipInfo.from_clip(clip_444)
    assert _get_y4m_header(info_444, vpy_policy).startswith(b"YUV4MPEG2 C444 ")

    # 3 planes: 410 (subsampling_w=2, subsampling_h=2)
    clip_410 = core.std.BlankClip(width=160, height=120, format=vs.YUV410P8, length=5)
    info_410 = ClipInfo.from_clip(clip_410)
    assert _get_y4m_header(info_410, vpy_policy).startswith(b"YUV4MPEG2 C410 ")

    # 3 planes: 411 (subsampling_w=2, subsampling_h=0)
    clip_411 = core.std.BlankClip(width=160, height=120, format=vs.YUV411P8, length=5)
    info_411 = ClipInfo.from_clip(clip_411)
    assert _get_y4m_header(info_411, vpy_policy).startswith(b"YUV4MPEG2 C411 ")

    # 3 planes: 440 (subsampling_w=0, subsampling_h=1)
    clip_440 = core.std.BlankClip(width=160, height=120, format=vs.YUV440P8, length=5)
    info_440 = ClipInfo.from_clip(clip_440)
    assert _get_y4m_header(info_440, vpy_policy).startswith(b"YUV4MPEG2 C440 ")

    # ManagedEnvironment as environment argument
    assert _get_y4m_header(info_420, vpy_env_factory()).startswith(b"YUV4MPEG2 C420 ")

    # None as environment argument
    assert _get_y4m_header(info_420, None).startswith(b"YUV4MPEG2 C420 ")


def test_get_y4m_header_error_cases() -> None:
    # Unsupported number of planes
    info_2planes = ClipInfo(
        width=100,
        height=100,
        fps_num=24,
        fps_den=1,
        num_frames=10,
        format_id=int(vs.YUV420P8),
        format_name="Test2Planes",
        num_planes=2,
        bytes_per_sample=1,
        subsampling_w=1,
        subsampling_h=1,
        planes=[],
    )
    with pytest.raises(ValueError, match="Unsupported number of planes for Y4M: 2"):
        _get_y4m_header(info_2planes, None)

    # Unsupported subsampling for 3 planes
    info_bad_sub = ClipInfo(
        width=100,
        height=100,
        fps_num=24,
        fps_den=1,
        num_frames=10,
        format_id=int(vs.YUV420P8),
        format_name="TestBadSub",
        num_planes=3,
        bytes_per_sample=1,
        subsampling_w=3,
        subsampling_h=3,
        planes=[],
    )
    with pytest.raises(ValueError, match=r"Unsupported subsampling for Y4M: \(3, 3\)"):
        _get_y4m_header(info_bad_sub, None)


def test_clean_help_formatter() -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["--help"])
    assert exc_info.value.code == 0

    with pytest.raises(SystemExit) as exc_info_serve:
        app(["serve", "--help"])
    assert exc_info_serve.value.code == 0


@pytest.mark.vpy("initial-core")
def test_serve_curve_auto_keygen(tmp_path: Path, vpy_policy: Policy, port: int) -> None:
    """Test serve with curve=True generating an ephemeral keypair."""
    script_file = tmp_path / "curve_test.vpy"
    script_file.write_text("import vapoursynth as vs\n\nvs.core.std.BlankClip().set_output(0)\n", encoding="utf-8")

    ready_event = threading.Event()
    stop_event = threading.Event()

    # Trigger stop immediately after ready
    def stopper() -> None:
        ready_event.wait(timeout=5.0)
        stop_event.set()

    thread = threading.Thread(target=stopper, daemon=True)
    thread.start()

    serve(
        script_file,
        address=f"tcp://127.0.0.1:{port}",
        curve=True,
        ready_event=ready_event,
        stop_event=stop_event,
        environment=vpy_policy,
    )
    thread.join(timeout=2.0)
    assert ready_event.is_set()
