from __future__ import annotations

import time
from pathlib import Path

from rich import box
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from .protocol import ServerStats


def _format_uptime(seconds: float) -> str:
    total_seconds = int(seconds)
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_bytes(num_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    val = float(num_bytes)
    for unit in units:
        if abs(val) < 1024.0 or unit == units[-1]:
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} TiB"


def build_dashboard(stats: ServerStats, address: str, interval: float, status_msg: str | None = None) -> Layout:
    """
    Construct an interactive, rich-native terminal dashboard layout.

    Args:
        stats: Point-in-time server telemetry snapshot.
        address: Connected server address.
        interval: Client refresh rate in seconds.
        status_msg: Optional error or reconnection status message.

    Returns:
        A Rich Layout instance ready for Live rendering.
    """
    header_size = 5 if status_msg else 4
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=header_size),
        Layout(name="kpis", size=9),
        Layout(name="outputs", ratio=1),
        Layout(name="clients", size=7),
        Layout(name="footer", size=1),
    )

    # 1. Header (2-column expansive grid)
    header_grid = Table.grid(expand=True)
    header_grid.add_column(justify="left", ratio=3)
    header_grid.add_column(justify="right", ratio=2)

    script_str = Path(stats.active_script).name if stats.active_script else "<none>"
    uptime_str = _format_uptime(stats.uptime_seconds)

    left_header = Text()
    left_header.append("vs-remote server monitor", style="bold cyan")
    left_header.append("  |  Address: ", style="dim")
    left_header.append(address, style="bold yellow")
    left_header.append("  |  Active Script: ", style="dim")
    left_header.append(script_str, style="bold green")

    right_header = Text()
    right_header.append("Uptime: ", style="dim")
    right_header.append(uptime_str, style="bold white")
    right_header.append("  |  Engine: ", style="dim")
    right_header.append(f"VapourSynth {stats.vs_version}", style="bold magenta")
    right_header.append(f" ({stats.vs_threads} threads)", style="dim")

    header_grid.add_row(left_header, right_header)

    if status_msg:
        alert_text = Text(f"● RECONNECTING: {status_msg}", style="bold red")
        header_panel = Panel(
            Group(header_grid, alert_text),
            title="[bold red] DISCONNECTED / RECONNECTING [/bold red]",
            border_style="bold red",
        )
    else:
        header_panel = Panel(header_grid, border_style="cyan")

    layout["header"].update(header_panel)

    # 2. KPIs Section (3 columns)
    layout["kpis"].split_row(
        Layout(name="throughput"),
        Layout(name="latency_compress"),
        Layout(name="resources"),
    )

    # 2a. Throughput Panel (Grid)
    fps_style = "bold green" if stats.fps > 0 else "white"
    bw_style = "bold cyan" if stats.bandwidth_mbps > 0 else "white"
    mb_s = stats.bandwidth_mbps / 8.0

    tp_grid = Table.grid(padding=(0, 1))
    tp_grid.add_column(style="dim", justify="right")
    tp_grid.add_column()

    tp_grid.add_row("Rate:", Text(f"{stats.fps:.2f} fps", style=fps_style))
    tp_grid.add_row(
        "Throughput:",
        Text.assemble((f"{stats.bandwidth_mbps:.3f} Mbps ", bw_style), (f"({mb_s:.2f} MB/s)", "dim")),
    )
    q_style = "bold yellow" if stats.in_flight_requests > 0 else "white"
    tp_grid.add_row("In-Flight Queue:", Text(str(stats.in_flight_requests), style=q_style))
    tp_grid.add_row("Frames Done:", Text(f"{stats.completed_frames:,} / {stats.total_frame_requests:,}", style="white"))
    tp_grid.add_row(
        "Errors / Cancel:",
        Text.assemble(
            (f"{stats.failed_frames}", "bold red" if stats.failed_frames > 0 else "dim"),
            (" / ", "dim"),
            (f"{stats.cancelled_frames}", "bold yellow" if stats.cancelled_frames > 0 else "dim"),
        ),
    )

    layout["kpis"]["throughput"].update(
        Panel(tp_grid, title="[bold]Throughput & Requests[/bold]", border_style="green")
    )

    # 2b. Latency & Compression Panel (Grid)
    render_style = (
        "bold green"
        if stats.avg_render_time_ms < 15.0
        else ("bold yellow" if stats.avg_render_time_ms < 50.0 else "bold red")
    )
    comp_style = (
        "bold green"
        if stats.avg_compress_time_ms < 5.0
        else ("bold yellow" if stats.avg_compress_time_ms < 20.0 else "bold red")
    )

    lc_grid = Table.grid(padding=(0, 1))
    lc_grid.add_column(style="dim", justify="right")
    lc_grid.add_column()

    lc_grid.add_row("Avg Render:", Text(f"{stats.avg_render_time_ms:.2f} ms", style=render_style))
    lc_grid.add_row("Avg Compress:", Text(f"{stats.avg_compress_time_ms:.2f} ms", style=comp_style))
    lc_grid.add_row("Algorithm:", Text(stats.compression_mode, style="bold yellow"))
    lc_grid.add_row("Ratio:", Text(f"{stats.compression_ratio:.2f}:1", style="bold cyan"))

    comp_fmt = _format_bytes(stats.total_compressed_bytes)
    raw_fmt = _format_bytes(stats.total_uncompressed_bytes)
    lc_grid.add_row(
        "Transferred:",
        Text.assemble((comp_fmt, "white"), (" ", ""), (f"({raw_fmt} raw)", "dim")),
    )

    layout["kpis"]["latency_compress"].update(
        Panel(lc_grid, title="[bold]Latency & Codec[/bold]", border_style="yellow")
    )

    # 2c. Resources Panel (Grid with ProgressBar for Cache & CPU)
    cache_pct = (stats.vs_used_cache_mb / stats.vs_max_cache_mb * 100.0) if stats.vs_max_cache_mb > 0 else 0.0
    cache_color = "green" if cache_pct < 70.0 else ("yellow" if cache_pct < 90.0 else "red")
    cache_bar = ProgressBar(total=100.0, completed=cache_pct, width=10, complete_style=cache_color)

    res_grid = Table.grid(padding=(0, 1))
    res_grid.add_column(style="dim", justify="right")
    res_grid.add_column()

    cache_val_grid = Table.grid(padding=(0, 1))
    cache_val_grid.add_row(cache_bar, Text(f"{cache_pct:.1f}%", style=cache_color))
    res_grid.add_row("VS Cache:", cache_val_grid)
    res_grid.add_row(
        "Cache Used:",
        Text(f"{stats.vs_used_cache_mb:,.1f} / {stats.vs_max_cache_mb:,} MiB", style="white"),
    )

    rss_text = (
        Text(f"{stats.process_rss_mb:.1f} MiB", style="bold magenta")
        if stats.process_rss_mb is not None
        else Text("N/A", style="dim")
    )
    res_grid.add_row("Process RSS:", rss_text)

    if stats.process_cpu_percent is not None:
        cpu_pct = min(100.0, stats.process_cpu_percent)
        if stats.process_cpu_percent < 100.0:
            cpu_color = "green"
        elif stats.process_cpu_percent < 400.0:
            cpu_color = "yellow"
        else:
            cpu_color = "red"

        cpu_bar = ProgressBar(total=100.0, completed=cpu_pct, width=10, complete_style=cpu_color)
        cpu_val_grid = Table.grid(padding=(0, 1))
        cpu_val_grid.add_row(cpu_bar, Text(f"{stats.process_cpu_percent:.1f}%", style=cpu_color))
        res_grid.add_row("Process CPU:", cpu_val_grid)
    else:
        res_grid.add_row("Process CPU:", Text("N/A", style="dim"))

    layout["kpis"]["resources"].update(Panel(res_grid, title="[bold]Memory & Host[/bold]", border_style="magenta"))

    # 3. Outputs Table (box.SIMPLE_HEAD)
    out_table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
    out_table.add_column("Index", style="cyan", justify="right", width=6)
    out_table.add_column("Name", style="bold magenta")
    out_table.add_column("Resolution", style="green", justify="center")
    out_table.add_column("FPS", justify="center")
    out_table.add_column("Format", style="yellow")
    out_table.add_column("Frames", justify="right", style="blue")

    for item in stats.outputs:
        info = item.info
        fps_str = f"{info.fps_num / info.fps_den:.3f}" if info.fps_den else f"{info.fps_num}"
        out_table.add_row(
            str(item.index),
            item.name,
            f"{info.width}x{info.height}",
            f"{fps_str} ({info.fps_num}/{info.fps_den})",
            info.format_name,
            f"{info.num_frames:,}",
        )

    if not stats.outputs:
        out_table.add_row("-", "No active outputs registered", "-", "-", "-", "-")

    layout["outputs"].update(
        Panel(out_table, title=f"[bold]Active Outputs ({len(stats.outputs)})[/bold]", border_style="blue")
    )

    # 4. Connected Clients Table (box.SIMPLE_HEAD)
    client_table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
    client_table.add_column("Identity", style="bold cyan", width=24)
    client_table.add_column("Active Requests", justify="right", style="yellow", width=16)
    client_table.add_column("Lifetime Requests", justify="right", style="white", width=18)
    client_table.add_column("Last Seen", justify="right", style="dim")

    for cl in stats.active_clients:
        if cl.last_active_seconds_ago < 60:
            idle_str = f"{cl.last_active_seconds_ago:.1f}s ago"
        else:
            idle_str = f"{_format_uptime(cl.last_active_seconds_ago)} ago"

        client_table.add_row(
            cl.identity[:20] + "..." if len(cl.identity) > 23 else cl.identity,
            str(cl.active_requests),
            f"{cl.total_requests:,}",
            idle_str,
        )

    if not stats.active_clients:
        client_table.add_row("-", "No active clients", "-", "-")

    layout["clients"].update(
        Panel(client_table, title=f"[bold]Connected Clients ({len(stats.active_clients)})[/bold]", border_style="cyan")
    )

    # 5. Footer (2-column grid)
    footer_grid = Table.grid(expand=True)
    footer_grid.add_column(justify="left")
    footer_grid.add_column(justify="right")

    if status_msg:
        footer_left = Text.assemble(
            ("● RECONNECTING: ", "bold red"),
            (f"{status_msg} ", "red"),
            (" |  Press Ctrl+C to exit", "dim"),
        )
    else:
        footer_left = Text(f"Polling interval: {interval:.1f}s  |  Press Ctrl+C to exit", style="dim")

    footer_grid.add_row(
        footer_left,
        Text(time.strftime("%H:%M:%S"), style="dim"),
    )
    layout["footer"].update(footer_grid)

    return layout
