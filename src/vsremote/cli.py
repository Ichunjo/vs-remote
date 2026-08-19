from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Annotated, override

import vapoursynth as vs
import zmq
from cyclopts import App, Parameter
from cyclopts.help import DefaultFormatter, HelpPanel
from rich.console import Console, ConsoleOptions
from rich.table import Table
from vsengine import ManagedEnvironment, Policy, UnifiedFuture

from .client.transport import ClientTransport
from .exceptions import UnsupportedFormatError
from .protocol import DEFAULT_ADDRESS, ClipInfo, Compression, FrameHeader, StatusCode, decompress_plane
from .server import ScriptRunner, ServerDaemon
from .utils import console, setup_logging

logger = logging.getLogger(__name__)


class CleanHelpFormatter(DefaultFormatter):
    @override
    def __call__(self, console: Console, options: ConsoleOptions, panel: HelpPanel) -> None:
        panel.entries = [
            entry.copy(positive_names=entry.positive_names[1:])  # type: ignore[no-untyped-call]
            if len(entry.positive_names) > 1 and not entry.positive_names[0].startswith("-")
            else entry
            for entry in panel.entries
        ]
        super().__call__(console, options, panel)


app = App("vsremote", console=console, default_parameter=Parameter(negative=()), help_formatter=CleanHelpFormatter())


@Parameter(name="*")
@dataclass(frozen=True)
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

    def create_transport(self, *, subscribe_streams: bool = False) -> ClientTransport:
        return ClientTransport(
            self.address,
            auth_token=self.auth_token,
            curve_server_key=self.curve_server_key,
            curve_public_key=self.curve_public_key,
            curve_secret_key=self.curve_secret_key,
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
    ready_event: Annotated[threading.Event | None, Parameter(show=False)] = None,
    stop_event: Annotated[threading.Event | None, Parameter(show=False)] = None,
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
        logger.info("CurveZMQ encryption enabled. Client public key: %s", curve_public_key)

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
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, lambda: asyncio.create_task(daemon.stop()))

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
def ping(config: ClientConfig = DEFAULT_CLIENT_CONFIG) -> None:
    """Check connectivity and liveness to a remote vs-remote server."""
    with config.create_transport(subscribe_streams=False) as transport:
        t0 = time.perf_counter()
        ok = transport.ping().result(timeout=10.0)
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
def info(config: ClientConfig = DEFAULT_CLIENT_CONFIG) -> None:
    """Query and display metadata for all outputs available on the remote server."""
    with config.create_transport(subscribe_streams=False) as transport:
        outputs = transport.list_outputs().result(timeout=10.0)

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
    config: ClientConfig = DEFAULT_CLIENT_CONFIG,
    *,
    output: int = 0,
    y4m: bool = False,
    prefetch: int = 8,
    backlog: int | None = None,
    compression: Compression = "zstd",
    environment: Annotated[Policy | ManagedEnvironment | None, Parameter(show=False)] = None,
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
    """
    with config.create_transport(subscribe_streams=False) as transport:
        clip_info = transport.get_clip_info(output).result(timeout=10.0)

        stdout_buf = sys.stdout.buffer

        if y4m:
            stdout_buf.write(_get_y4m_header(clip_info, environment))
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
                        inflight[next_to_request] = transport.request_frame(
                            output, next_to_request, compression=compression
                        )
                    else:
                        break

            if (fut := inflight.pop(n, None)) is None:
                fut = transport.request_frame(output, n, compression=compression)

            header, plane_parts = fut.result(timeout=30.0)

            if header.status != StatusCode.OK:
                header.status.raise_for_status(f"Failed to fetch frame {n}: {header.error_message}")

            if y4m:
                stdout_buf.write(b"FRAME\n")

            for p, compressed in enumerate(plane_parts):
                decompressed = decompress_plane(compressed, clip_info.planes[p].size_bytes, header.compression)
                stdout_buf.write(decompressed)

            stdout_buf.flush()


@app.command
def keygen() -> None:
    """
    Generate a new Curve25519 keypair for CurveZMQ transport encryption.
    """
    pub, sec = zmq.curve_keypair()
    pub_str = pub.decode("ascii")
    sec_str = sec.decode("ascii")

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

    if tokens:
        app(tokens)


def main() -> None:
    app.meta()


async def _watch_stop(daemon: ServerDaemon, stop_event: threading.Event) -> None:
    await asyncio.get_running_loop().run_in_executor(None, stop_event.wait)
    await daemon.stop()


async def _wakeup() -> None:
    # Heartbeat required on Windows to process SIGINT
    while True:  # noqa: ASYNC110
        await asyncio.sleep(0.5)


def _get_y4m_header(info: ClipInfo, environment: Policy | ManagedEnvironment | None) -> bytes:
    if info.num_planes == 1:
        y4mformat = "mono"
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
    else:
        raise UnsupportedFormatError(f"Unsupported number of planes for Y4M: {info.num_planes}")

    if isinstance(policy := environment, Policy):
        ctx = policy.new_environment().use()
    elif isinstance(environment, ManagedEnvironment):
        ctx = environment.use()
    else:
        ctx = nullcontext()

    with ctx:
        bits = vs.core.get_video_format(info.format_id).bits_per_sample

    if bits > 8:
        y4mformat += f"p{bits}"

    header = (
        f"YUV4MPEG2 C{y4mformat} W{info.width} H{info.height} "
        f"F{info.fps_num}:{info.fps_den} Ip A0:0 XLENGTH={info.num_frames}\n"
    )
    return header.encode("ascii")


if __name__ == "__main__":
    main()
