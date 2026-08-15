from __future__ import annotations

import asyncio
import gc
from logging import getLogger

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from vsengine.adapters.asyncio import AsyncIOLoop
from vsengine.loops import NO_LOOP, get_loop, set_loop

logger = getLogger(__name__)


def ensure_vsengine_loop(loop: asyncio.AbstractEventLoop | None = None) -> None:
    if get_loop() is NO_LOOP:
        set_loop(AsyncIOLoop(loop))


def gc_collect() -> None:
    logger.debug("Running garbage collection")

    for i in range(3):
        gc.collect(generation=i)

    for _ in range(3):
        gc.collect()


console = Console(stderr=True)


def setup_logging(level: int) -> None:
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        log_time_format=lambda dt: Text(f"[{dt:%H:%M:%S}.{dt.microsecond // 1000:03d}]"),
    )

    root_logger = getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        root_logger.addHandler(handler)
