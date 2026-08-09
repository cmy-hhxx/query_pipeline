from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


async def run_concurrent(
    items: list[Any],
    worker: Callable[[Any], Awaitable[Any]],
    *,
    concurrency: int,
    description: str = "Processing",
) -> list[Any]:
    """Run a worker over items with bounded concurrency.

    Progress reporting is intentionally absent: the pipeline logs per-stage
    summaries instead of live progress bars (audit-friendly, log/output live
    in the same directory).
    """
    # clamp: concurrency<=0 would deadlock every worker on a 0-permit semaphore.
    # LLMClient already clamps to 1; this is the single shared choke point for all stages.
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[Any] = [None] * len(items)

    async def wrapped(index: int, item: Any) -> None:
        async with semaphore:
            results[index] = await worker(item)

    await asyncio.gather(*(wrapped(i, item) for i, item in enumerate(items)))
    return results
