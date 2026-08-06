from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from rich.progress import Progress


async def run_concurrent(
    items: list[Any],
    worker: Callable[[Any], Awaitable[Any]],
    *,
    concurrency: int,
    description: str = "Processing",
    show_progress: bool = True,
    progress: Progress | None = None,
) -> list[Any]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[Any] = [None] * len(items)

    async def wrapped(index: int, item: Any) -> None:
        async with semaphore:
            results[index] = await worker(item)

    if progress is not None:
        # Attach a transient task to an existing Progress (e.g. one per
        # session inside the session loop) instead of owning a bar.
        task_id = progress.add_task(description, total=len(items))

        async def tracked(index: int, item: Any) -> None:
            await wrapped(index, item)
            progress.advance(task_id)

        try:
            await asyncio.gather(*(tracked(i, item) for i, item in enumerate(items)))
        finally:
            progress.remove_task(task_id)
        return results

    if not show_progress:
        await asyncio.gather(*(wrapped(i, item) for i, item in enumerate(items)))
        return results

    with Progress() as progress:
        task_id = progress.add_task(description, total=len(items))
        async def tracked(index: int, item: Any) -> None:
            await wrapped(index, item)
            progress.advance(task_id)

        await asyncio.gather(*(tracked(i, item) for i, item in enumerate(items)))
    return results
