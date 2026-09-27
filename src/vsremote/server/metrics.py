from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass

import psutil
import vapoursynth as vs

from ..protocol import ClientSessionStats, Compression, OutputItem, ServerStats


@dataclass(slots=True)
class ClientSession:
    identity_bytes: bytes
    identity_str: str
    active_requests: int = 0
    total_requests: int = 0
    last_active: float = 0.0


@dataclass(slots=True)
class FrameWindowEntry:
    timestamp: float
    frames: int
    compressed_bytes: int


class ServerMetricsCollector:
    """Thread-safe metrics aggregator for vs-remote server performance telemetry."""

    def __init__(self, window_seconds: float = 2.0, max_latencies: int = 100) -> None:
        self.window_seconds = window_seconds
        self.start_time = time.time()
        self._lock = threading.Lock()

        # Cumulative counters
        self.total_frame_requests = 0
        self.completed_frames = 0
        self.failed_frames = 0
        self.cancelled_frames = 0
        self.total_uncompressed_bytes = 0
        self.total_compressed_bytes = 0

        # Rolling windows for throughput rate calculations
        self._throughput_window = deque[FrameWindowEntry]()

        # Rolling latencies (in milliseconds)
        self._render_latencies_ms = deque[float](maxlen=max_latencies)
        self._compress_latencies_ms = deque[float](maxlen=max_latencies)

        # Client sessions
        self._clients = dict[bytes, ClientSession]()

        # Process monitor
        try:
            self._process: psutil.Process | None = psutil.Process()
            # Initial call to cpu_percent to initialize internal counter
            self._process.cpu_percent()
        except psutil.Error:
            self._process = None

    def on_request_start(self, identity: bytes) -> None:
        """Record the initiation of a request from a client identity."""
        now = time.time()
        with self._lock:
            session = self._clients.get(identity)
            if session is None:
                if not identity:
                    id_str = "unknown"
                else:
                    try:
                        decoded = identity.decode("utf-8")
                        id_str = decoded if decoded.isprintable() else identity.hex()
                    except UnicodeDecodeError:
                        id_str = identity.hex()
                session = ClientSession(identity_bytes=identity, identity_str=id_str)
                self._clients[identity] = session
            session.active_requests += 1
            session.total_requests += 1
            session.last_active = now

    def on_request_end(self, identity: bytes) -> None:
        """Record the completion of a request from a client identity."""
        now = time.time()
        with self._lock:
            if session := self._clients.get(identity):
                session.active_requests = max(0, session.active_requests - 1)
                session.last_active = now

    def on_frame_request(self) -> None:
        """Increment total frame request counter."""
        with self._lock:
            self.total_frame_requests += 1

    def on_frame_completed(
        self,
        uncompressed_bytes: int,
        compressed_bytes: int,
        render_time_ms: float,
        compress_time_ms: float,
    ) -> None:
        """Record successful frame rendering and compression."""
        now = time.time()
        with self._lock:
            self.completed_frames += 1
            self.total_uncompressed_bytes += uncompressed_bytes
            self.total_compressed_bytes += compressed_bytes

            self._throughput_window.append(FrameWindowEntry(timestamp=now, frames=1, compressed_bytes=compressed_bytes))
            self._render_latencies_ms.append(render_time_ms)
            self._compress_latencies_ms.append(compress_time_ms)
            self._prune_window(now)

    def on_frame_failed(self) -> None:
        """Record failed frame render."""
        with self._lock:
            self.failed_frames += 1

    def on_frame_cancelled(self) -> None:
        """Record cancelled frame render."""
        with self._lock:
            self.cancelled_frames += 1

    def snapshot(
        self,
        *,
        active_script: str | os.PathLike[str] | None = None,
        outputs: Sequence[OutputItem] = (),
        compression_mode: Compression = "zstd",
        in_flight_requests: int = 0,
    ) -> ServerStats:
        """Generate a point-in-time ServerStats telemetry struct."""
        now = time.time()
        uptime = now - self.start_time

        with self._lock:
            self._prune_window(now)

            # Compute windowed rates
            total_window_frames = sum(entry.frames for entry in self._throughput_window)
            total_window_bytes = sum(entry.compressed_bytes for entry in self._throughput_window)

            if self._throughput_window:
                earliest = self._throughput_window[0].timestamp
                time_span = max(now - earliest, 0.1)
                fps = total_window_frames / time_span
                bandwidth_mbps = (total_window_bytes * 8.0) / (time_span * 1_000_000.0)
            else:
                fps = 0.0
                bandwidth_mbps = 0.0

            # Compute average latencies
            avg_render_ms = (
                sum(self._render_latencies_ms) / len(self._render_latencies_ms) if self._render_latencies_ms else 0.0
            )
            avg_compress_ms = (
                sum(self._compress_latencies_ms) / len(self._compress_latencies_ms)
                if self._compress_latencies_ms
                else 0.0
            )

            # Compression ratio (uncompressed / compressed)
            if self.total_compressed_bytes > 0:
                compression_ratio = self.total_uncompressed_bytes / self.total_compressed_bytes
            else:
                compression_ratio = 1.0

            # Client sessions snapshot (clean up clients idle for > 1 hour)
            client_stats = list[ClientSessionStats]()
            stale_clients = list[bytes]()
            for ident, sess in self._clients.items():
                idle = now - sess.last_active
                if idle > 3600.0 and sess.active_requests == 0:
                    stale_clients.append(ident)
                    continue
                client_stats.append(
                    ClientSessionStats(
                        identity=sess.identity_str,
                        active_requests=sess.active_requests,
                        total_requests=sess.total_requests,
                        last_active_seconds_ago=idle,
                    )
                )

            for stale in stale_clients:
                self._clients.pop(stale, None)

            # Sort clients by active requests then total requests
            client_stats.sort(key=lambda c: (c.active_requests, c.total_requests), reverse=True)

        # Query OS process memory & CPU
        process_rss_mb = None
        process_cpu_percent = None
        if self._process is not None:
            with suppress(psutil.Error):
                mem = self._process.memory_info()
                process_rss_mb = mem.rss / (1024.0 * 1024.0)
                process_cpu_percent = self._process.cpu_percent()

        vs_version = str(vs.core.core_version)
        vs_threads = vs.core.num_threads
        # vs.core.used_cache_size is in bytes; convert to MiB
        vs_used_cache_mb = round(vs.core.used_cache_size / (1024.0 * 1024.0), 1)
        # vs.core.max_cache_size is already in MiB
        vs_max_cache_mb = vs.core.max_cache_size

        return ServerStats(
            uptime_seconds=uptime,
            active_script=str(active_script),
            num_outputs=len(outputs),
            total_frame_requests=self.total_frame_requests,
            completed_frames=self.completed_frames,
            failed_frames=self.failed_frames,
            cancelled_frames=self.cancelled_frames,
            in_flight_requests=in_flight_requests,
            fps=round(fps, 2),
            bandwidth_mbps=round(bandwidth_mbps, 3),
            avg_render_time_ms=round(avg_render_ms, 2),
            avg_compress_time_ms=round(avg_compress_ms, 2),
            compression_mode=compression_mode,
            total_uncompressed_bytes=self.total_uncompressed_bytes,
            total_compressed_bytes=self.total_compressed_bytes,
            compression_ratio=round(compression_ratio, 2),
            vs_version=vs_version,
            vs_threads=vs_threads,
            vs_used_cache_mb=vs_used_cache_mb,
            vs_max_cache_mb=vs_max_cache_mb,
            process_rss_mb=round(process_rss_mb, 1) if process_rss_mb is not None else None,
            process_cpu_percent=round(process_cpu_percent, 1) if process_cpu_percent is not None else None,
            outputs=list(outputs),
            active_clients=client_stats,
        )

    def _prune_window(self, now: float) -> None:
        # Prune throughput window entries older than the window duration (lock must be held).
        cutoff = now - self.window_seconds
        while self._throughput_window and self._throughput_window[0].timestamp < cutoff:
            self._throughput_window.popleft()
