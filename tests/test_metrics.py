from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from vsengine.policy import Policy

from vsremote.cli import ClientConfig, top
from vsremote.client import ClientTransport, RemoteClient
from vsremote.protocol import ServerStats
from vsremote.server.metrics import ServerMetricsCollector
from vsremote.tui import _format_bytes, _format_uptime, build_dashboard

if TYPE_CHECKING:
    from conftest import ServerFactory


def test_tui_helpers() -> None:
    # Test _format_uptime
    assert _format_uptime(45) == "45s"
    assert _format_uptime(125) == "2m 5s"
    assert _format_uptime(3665) == "1h 1m 5s"
    assert _format_uptime(90065) == "1d 1h 1m 5s"

    # Test _format_bytes
    assert "B" in _format_bytes(500)
    assert "KiB" in _format_bytes(2048)
    assert "MiB" in _format_bytes(1024 * 1024 * 5)
    assert "GiB" in _format_bytes(1024 * 1024 * 1024 * 2)


@pytest.mark.vpy("initial-core")
def test_metrics_collector_unit() -> None:
    # Instantiated cleanly without any daemon or runner mocks
    collector = ServerMetricsCollector(window_seconds=2.0)

    # Initial snapshot
    snap0 = collector.snapshot(active_script=None, outputs=(), compression_mode="zstd", in_flight_requests=0)
    assert snap0.total_frame_requests == 0
    assert snap0.completed_frames == 0
    assert snap0.fps == 0.0
    assert snap0.in_flight_requests == 0
    assert snap0.vs_max_cache_mb > 0
    assert snap0.vs_used_cache_mb >= 0.0
    assert snap0.vs_threads > 0

    # Simulate client activity
    ident = b"client_1"
    collector.on_request_start(ident)
    collector.on_frame_request()

    collector.on_frame_completed(
        uncompressed_bytes=1000,
        compressed_bytes=500,
        render_time_ms=10.0,
        compress_time_ms=2.0,
    )
    collector.on_request_end(ident)

    snap1 = collector.snapshot(
        active_script="test_script.vpy",
        outputs=(),
        compression_mode="zstd",
        in_flight_requests=1,
    )
    assert snap1.active_script == "test_script.vpy"
    assert snap1.in_flight_requests == 1
    assert snap1.total_frame_requests == 1
    assert snap1.completed_frames == 1
    assert snap1.total_uncompressed_bytes == 1000
    assert snap1.total_compressed_bytes == 500
    assert snap1.compression_ratio == 2.0
    assert snap1.avg_render_time_ms == 10.0
    assert snap1.avg_compress_time_ms == 2.0
    assert len(snap1.active_clients) == 1
    assert snap1.active_clients[0].identity == "client_1"
    assert snap1.active_clients[0].total_requests == 1
    assert snap1.active_clients[0].active_requests == 0

    # Test binary non-printable identity falls back to hex
    raw_binary_ident = b"\x00\x12\x34\xab\xcd"
    collector.on_request_start(raw_binary_ident)
    snap2 = collector.snapshot(active_script=None, outputs=(), compression_mode="zstd", in_flight_requests=0)
    binary_client = next(c for c in snap2.active_clients if c.identity == "001234abcd")
    assert binary_client.active_requests == 1
    collector.on_request_end(raw_binary_ident)

    # Test build_dashboard renders without error (both normal and reconnecting alert states)
    layout = build_dashboard(snap1, "tcp://127.0.0.1:5555", 1.0)
    assert layout is not None

    alert_layout = build_dashboard(snap1, "tcp://127.0.0.1:5555", 1.0, status_msg="Server unreachable (timed out)")
    assert alert_layout is not None


@pytest.mark.vpy("no-core")
def test_metrics_integration(
    server: ServerFactory,
    tmp_path: Path,
    vpy_policy: Policy,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_file = tmp_path / "test_metrics_script.vpy"
    script_file.write_text(
        "import vapoursynth as vs\n"
        "clip = vs.core.std.BlankClip(width=160, height=120, format=vs.YUV420P8, length=10)\n"
        "clip.set_output(0)\n",
        encoding="utf-8",
    )

    with server(script_file, compression="zstd", environment=vpy_policy) as (host, port):
        address = f"tcp://{host}:{port}"

        # 1. Test get_stats via ClientTransport with custom client_id
        with ClientTransport(address, client_id="test_worker#1", subscribe_streams=False) as transport:
            stats = transport.get_stats().result(timeout=5.0)
            assert isinstance(stats, ServerStats)
            assert stats.num_outputs == 1
            assert stats.active_script == str(script_file.resolve())
            assert stats.compression_mode == "zstd"

            # Request a frame to generate metrics
            transport.request_frame(0, 0).result(timeout=5.0)

            stats_after = transport.get_stats().result(timeout=5.0)
            assert stats_after.completed_frames >= 1
            assert stats_after.total_compressed_bytes > 0
            assert stats_after.total_uncompressed_bytes > 0
            assert any(c.identity == "test_worker#1" for c in stats_after.active_clients)

        # 2. Test get_stats via RemoteClient
        with RemoteClient(address, subscribe_streams=False) as client:
            client_stats = client.get_stats().result(timeout=5.0)
            assert isinstance(client_stats, ServerStats)
            assert client_stats.completed_frames >= 1

        # 3. Test CLI top --json
        capsys.readouterr()  # Flush
        client_cfg = ClientConfig(address=address)
        top(client_cfg, json_output=True)
        captured = capsys.readouterr()

        json_output = captured.out

        data = json.loads(json_output)
        assert data["num_outputs"] == 1
        assert data["completed_frames"] >= 1
        assert "fps" in data
        assert "vs_version" in data
        assert data["vs_max_cache_mb"] > 0
        assert data["vs_used_cache_mb"] >= 0.0
        assert "outputs" in data
        assert len(data["outputs"]) == 1
