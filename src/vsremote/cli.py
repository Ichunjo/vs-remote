from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import msgspec
import vapoursynth as vs
import zmq
from cyclopts import App, Parameter
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text
from vsengine import ManagedEnvironment, Policy, UnifiedFuture

from .client.transport import ClientTransport
from .exceptions import RemoteError, TransportError, UnsupportedFormatError
from .protocol import DEFAULT_ADDRESS, ClipInfo, Compression, FrameHeader, ServerStats, StatusCode, decompress_plane
from .server import ScriptRunner, ServerDaemon
from .tui import build_dashboard
from .utils import console, setup_logging

logger = logging.getLogger(__name__)
app = App("vsremote", console=console, default_parameter=Parameter(negative=()))


@Parameter(name="*")
@dataclass(frozen=True, slots=True)
class ClientConfig:
    """Connection and authentication parameters for remote server operations."""

    address: str = DEFAULT_ADDRESS
    """Remote server address (e.g. tcp://127.0.0.1:5555 or ipc:///tmp/vsremote.sock)."""

    auth_token: Annotated[str | None, Parameter(env_var="VSREMOTE_AUTH_TOKEN")] = None
    """Optional shared secret authentication token."""

    curve_server_key: Annotated[str | None, Parameter(env_var="VSREMOTE_CURVE_SERVER_KEY")] = None
    """Optional CurveZMQ server public key."""

    curve_public_key: Annotated[str | None, Parameter(env_var="VSREMOTE_CURVE_PUBLIC_KEY")] = None
    """Optional CurveZMQ client public key."""

    curve_secret_key: Annotated[str | None, Parameter(env_var="VSREMOTE_CURVE_SECRET_KEY")] = None
    """Optional CurveZMQ client secret key."""

    client_id: Annotated[str | None, Parameter(env_var="VSREMOTE_CLIENT_ID")] = None
    """Optional human-readable identity for this client (e.g. 'worker-1' or 'encoder')."""

    def create_transport(
        self,
        *,
        subscribe_streams: bool = False,
        default_client_id: str | None = None,
    ) -> ClientTransport:
        return ClientTransport(
            self.address,
            auth_token=self.auth_token,
            curve_server_key=self.curve_server_key,
            curve_public_key=self.curve_public_key,
            curve_secret_key=self.curve_secret_key,
            client_id=self.client_id or default_client_id,
            subscribe_streams=subscribe_streams,
        )


DEFAULT_CLIENT_CONFIG = ClientConfig()


@app.command
def serve(
    script_path: str | os.PathLike[str] | None = None,
    /,
    *,
    address: str = DEFAULT_ADDRESS,
    compression: Compression = "zstd",
    max_workers: Annotated[int | None, Parameter(env_var="VSREMOTE_MAX_WORKERS")] = None,
    allow_eval: Annotated[bool, Parameter(env_var="VSREMOTE_ALLOW_EVAL")] = False,
    auth_token: Annotated[str | None, Parameter(env_var="VSREMOTE_AUTH_TOKEN")] = None,
    curve: bool = False,
    curve_secret_key: Annotated[str | None, Parameter(env_var="VSREMOTE_CURVE_SECRET_KEY")] = None,
    curve_public_key: Annotated[str | None, Parameter(env_var="VSREMOTE_CURVE_PUBLIC_KEY")] = None,
    curve_allowed_keys: Annotated[
        Sequence[str] | None,
        Parameter(env_var="VSREMOTE_CURVE_ALLOWED_KEYS", consume_multiple=True),
    ] = None,
    # Not exposed to the CLI
    ready_event: Annotated[threading.Event | asyncio.Event | None, Parameter(show=False)] = None,
    stop_event: Annotated[threading.Event | asyncio.Event | None, Parameter(show=False)] = None,
    environment: Annotated[Policy | ManagedEnvironment | None, Parameter(show=False)] = None,
) -> None:
    """
    Host a VapourSynth script on the network.

    Args:
        script_path: Path to the .vpy script file.
        address: Network or IPC address to bind (e.g. tcp://127.0.0.1:5555 or ipc:///tmp/vsremote.sock).
        compression: Compression mode for video frames.
        max_workers: Worker thread pool size for compression.
        allow_eval: Allow remote clients to execute dynamic Python code or switch scripts.
        auth_token: Optional shared secret authentication token.
        curve: Automatically generate an ephemeral CurveZMQ keypair for this session.
        curve_secret_key: Optional CurveZMQ server secret key for end-to-end encryption.
        curve_public_key: Optional CurveZMQ server public key.
        curve_allowed_keys: Optional sequence of authorized CurveZMQ client public keys.
    """
    if curve and not curve_secret_key:
        pub, sec = zmq.curve_keypair()
        curve_public_key = pub.decode("ascii")
        curve_secret_key = sec.decode("ascii")
        logger.info("CurveZMQ encryption enabled. Server public key: %s", curve_public_key)

    runner = (
        ScriptRunner.from_script(script_path, environment=environment)
        if script_path
        else ScriptRunner(environment=environment)
    )

    with runner:
        daemon = ServerDaemon(
            runner,
            address=address,
            compression=compression,
            max_workers=max_workers,
            allow_eval=allow_eval,
            auth_token=auth_token,
            curve_secret_key=curve_secret_key,
            curve_public_key=curve_public_key,
            curve_allowed_keys=curve_allowed_keys,
        )

        async def run() -> None:
            if sys.platform != "win32" and threading.current_thread() is threading.main_thread():
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(daemon.stop()))
                loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(daemon.stop()))

                wakeup_task = None
            else:
                wakeup_task = asyncio.create_task(_wakeup(), name="wakeup")

            stop_task = asyncio.create_task(_watch_stop(daemon, stop_event), name="watch_stop") if stop_event else None
            try:
                await daemon.start(ready_event=ready_event)
            finally:
                if wakeup_task:
                    wakeup_task.cancel()
                if stop_task:
                    stop_task.cancel()

        try:
            asyncio.run(run(), loop_factory=asyncio.SelectorEventLoop)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down server...")


@app.command
def ping(*, config: ClientConfig = DEFAULT_CLIENT_CONFIG, timeout: float = 10.0) -> None:
    """
    Check connectivity and liveness to a remote vs-remote server.

    Args:
        timeout: The number of seconds to wait for the ping result.
    """
    with config.create_transport(subscribe_streams=False) as transport:
        t0 = time.perf_counter()
        ok = transport.ping().result(timeout=timeout)
        lat = (time.perf_counter() - t0) * 1000.0

    if ok:
        console.print(
            f"[bold green]OK[/bold green] - Successfully connected to [cyan]{config.address}[/cyan] "
            f"(RTT: [yellow]{lat:.2f}ms[/yellow])"
        )
    else:
        console.print(f"[bold red]FAIL[/bold red] - Ping failed for [cyan]{config.address}[/cyan]")
        raise SystemExit(1)


@app.command
def info(*, config: ClientConfig = DEFAULT_CLIENT_CONFIG, timeout: float = 10.0) -> None:
    """
    Query and display metadata for all outputs available on the remote server.

    Args:
        timeout: The number of seconds to wait for the info result.
    """
    with config.create_transport(subscribe_streams=False) as transport:
        outputs = transport.list_outputs().result(timeout=timeout)

    table = Table(title=f"Remote Outputs for {config.address}")
    table.add_column("Index", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="magenta")
    table.add_column("Resolution", justify="center", style="green")
    table.add_column("FPS", justify="center")
    table.add_column("Format", style="yellow")
    table.add_column("Frames", justify="right", style="blue")

    for item in outputs:
        clip_info = item.info
        fps_str = f"{clip_info.fps_num / clip_info.fps_den:.3f}" if clip_info.fps_den else f"{clip_info.fps_num}"
        table.add_row(
            str(item.index),
            item.name,
            f"{clip_info.width}x{clip_info.height}",
            f"{fps_str} ({clip_info.fps_num}/{clip_info.fps_den})",
            clip_info.format_name,
            str(clip_info.num_frames),
        )

    console.print(table)


@app.command
def pipe(
    *,
    config: ClientConfig = DEFAULT_CLIENT_CONFIG,
    output: int = 0,
    y4m: bool = False,
    prefetch: int = 8,
    backlog: int | None = None,
    compression: Compression = "zstd",
    timeout: float = 10.0,
) -> None:
    """
    Stream video frames directly from the remote server to stdout in Y4M format or raw planes.

    Args:
        output: Output clip index on the remote server.
        y4m: Output standard Y4M (YUV4MPEG2) header and frame tags.
        prefetch: Number of frames to prefetch ahead concurrently (0 to disable).
        backlog: Maximum number of in-flight and prefetched frame requests buffered
            (defaults to max(prefetch * 3, prefetch)).
        compression: Frame transport compression.
        timeout: Maximum time in seconds to wait for operations.
    """
    with config.create_transport(subscribe_streams=False) as transport:
        clip_info = transport.get_clip_info(output).result(timeout=timeout)

        stdout_buf = sys.stdout.buffer

        if y4m:
            stdout_buf.write(_get_y4m_header(clip_info))
            stdout_buf.flush()

        prefetch_count = max(0, prefetch)
        backlog_count = max(prefetch_count, backlog if backlog is not None else prefetch_count * 3)

        inflight = dict[int, UnifiedFuture[tuple[FrameHeader, list[bytes]]]]()

        for n in range(clip_info.num_frames):
            if prefetch_count > 0:
                while len(inflight) < backlog_count:
                    next_to_request = n + len(inflight)
                    if next_to_request >= clip_info.num_frames or next_to_request > n + prefetch_count:
                        break
                    if next_to_request not in inflight:
                        inflight[next_to_request] = transport.request_frame(output, next_to_request, compression)
                    else:
                        break

            if (fut := inflight.pop(n, None)) is None:
                fut = transport.request_frame(output, n, compression=compression)

            header, plane_parts = fut.result(timeout=timeout)

            if header.status != StatusCode.OK:
                header.status.raise_for_status(f"Failed to fetch frame {n}: {header.error_message}")

            if y4m:
                stdout_buf.write(b"FRAME\n")

            for p, compressed in enumerate(plane_parts):
                decompressed = decompress_plane(compressed, clip_info.planes[p].size_bytes, header.compression)
                stdout_buf.write(decompressed)

            stdout_buf.flush()


@app.command
def top(
    *,
    config: ClientConfig = DEFAULT_CLIENT_CONFIG,
    interval: float = 1.0,
    json_output: Annotated[bool, Parameter("--json")] = False,
    timeout: float = 3.0,
) -> None:
    """
    Monitor real-time performance, throughput, and memory metrics of a vs-remote server.

    Args:
        interval: Polling interval in seconds.
        json_output: Output a single JSON snapshot and exit.
        timeout: Maximum time in seconds to wait for the stats result.
    """
    with (
        contextlib.suppress(KeyboardInterrupt),
        config.create_transport(subscribe_streams=False, default_client_id="top") as transport,
    ):
        if json_output:
            stats = transport.get_stats().result(timeout=timeout)
            raw = msgspec.json.encode(stats)
            # Pretty-print formatted JSON
            parsed = json.loads(raw)
            print(json.dumps(parsed, indent=2))
            return

        last_known_stats: ServerStats | None = None
        error_msg: str | None = None

        with Live(console=console, screen=True, refresh_per_second=int(max(1.0 / interval, 2.0))) as live:
            live.update(Status("Starting top..."))
            while True:
                try:
                    stats = transport.get_stats().result(timeout=max(interval * 2.0, timeout))
                    last_known_stats = stats
                    error_msg = None
                except TimeoutError:
                    error_msg = f"Server unreachable at {config.address} (timed out)"
                except (TransportError, RemoteError, Exception) as exc:
                    err_str = str(exc).strip()
                    error_type = type(exc).__name__
                    error_msg = f"Connection error: {err_str}" if err_str else f"Connection error ({error_type})"

                if last_known_stats is not None:
                    dashboard = build_dashboard(last_known_stats, config.address, interval, status_msg=error_msg)
                    live.update(dashboard)
                else:
                    # Initial connection attempt failed
                    connecting_panel = Panel(
                        Text(f"Connecting to {config.address}...\n{error_msg or ''}", style="bold yellow"),
                        title="[bold cyan]vsremote top[/bold cyan]",
                    )
                    live.update(connecting_panel)

                time.sleep(interval)


@app.command
def keygen() -> None:
    """
    Generate a new Curve25519 keypair for CurveZMQ transport encryption.
    """
    pub, sec = zmq.curve_keypair()
    pub_str = escape(pub.decode("ascii"))
    sec_str = escape(sec.decode("ascii"))

    console.print("[bold green]Generated CurveZMQ Keypair:[/bold green]\n")
    console.print(f"  [bold]Public Key:[/bold]  [cyan]{pub_str}[/cyan]")
    console.print(f"  [bold]Secret Key:[/bold]  [yellow]{sec_str}[/yellow]\n")

    console.print("[bold dim]Usage (Server Encryption):[/bold dim]")
    console.print(f'  Server:  vsremote serve script.vpy --curve-secret-key "{sec_str}"')
    console.print(f'  Client:  vsremote.source("tcp://...", curve_server_key="{pub_str}")\n')

    console.print("[bold dim]Usage (Client Authentication):[/bold dim]")
    console.print(
        f'  Server:  vsremote serve script.vpy --curve-secret-key "<SERVER_SEC>" --curve-allowed-keys "{pub_str}"'
    )
    console.print(
        f'  Client:  vsremote.source("tcp://...", curve_server_key="<SERVER_PUB>", '
        f'curve_public_key="{pub_str}", curve_secret_key="{sec_str}")'
    )
    console.print(
        f'  CLI:     vsremote pipe --curve-server-key "<SERVER_PUB>" '
        f'--curve-public-key "{pub_str}" --curve-secret-key "{sec_str}"\n'
    )


@app.meta.default
def main_meta(*tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)], verbose: bool = False) -> None:
    """
    High-performance remote execution server for VapourSynth

    Args:
        verbose: Enable debug logging.
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    app(tokens)


def main() -> None:
    app.meta()


async def _watch_stop(daemon: ServerDaemon, stop_event: threading.Event | asyncio.Event) -> None:
    while not stop_event.is_set():  # noqa: ASYNC110
        await asyncio.sleep(0.5)
    await daemon.stop()


async def _wakeup() -> None:
    # Heartbeat required on Windows to process SIGINT
    while True:  # noqa: ASYNC110
        await asyncio.sleep(0.5)


def _get_y4m_header(info: ClipInfo) -> bytes:
    if info.color_family not in (vs.ColorFamily.GRAY, vs.ColorFamily.YUV):
        raise UnsupportedFormatError(
            f"Unsupported format for Y4M: {info.format_name} (Y4M only supports YUV and Gray formats)"
        )

    if info.sample_type != vs.SampleType.INTEGER:
        raise UnsupportedFormatError(
            f"Unsupported sample type for Y4M: {info.format_name} (Y4M only supports integer formats)"
        )

    if info.bits_per_sample not in (8, 9, 10, 12, 14, 16):
        raise UnsupportedFormatError(
            f"Unsupported bit depth for Y4M: {info.bits_per_sample} (Y4M only supports 8, 9, 10, 12, 14, 16 bits)"
        )

    if info.num_planes == 1:
        y4mformat = "mono" if info.bits_per_sample == 8 else f"mono{info.bits_per_sample}"
    elif info.num_planes == 3:
        match info.subsampling_w, info.subsampling_h:
            case 1, 1:
                y4mformat = "420"
            case 1, 0:
                y4mformat = "422"
            case 0, 0:
                y4mformat = "444"
            case 2, 2:
                y4mformat = "410"
            case 2, 0:
                y4mformat = "411"
            case 0, 1:
                y4mformat = "440"
            case _:
                raise UnsupportedFormatError(
                    f"Unsupported subsampling for Y4M: ({info.subsampling_w}, {info.subsampling_h})"
                )

        if info.bits_per_sample > 8:
            y4mformat += f"p{info.bits_per_sample}"
    else:
        raise UnsupportedFormatError(f"Unsupported number of planes for Y4M: {info.num_planes}")

    header = (
        f"YUV4MPEG2 C{y4mformat} W{info.width} H{info.height} "
        f"F{info.fps_num}:{info.fps_den} Ip A0:0 XLENGTH={info.num_frames}\n"
    )
    return header.encode("ascii")


if __name__ == "__main__":
    main()
