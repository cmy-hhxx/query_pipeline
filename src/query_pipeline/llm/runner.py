from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Awaitable, Callable
from typing import Any

from rich.progress import Progress


async def run_concurrent(
    items: list[Any],
    worker: Callable[[Any], Awaitable[Any]],
    *,
    concurrency: int,
    description: str = "Processing",
) -> list[Any]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[Any] = [None] * len(items)

    async def wrapped(index: int, item: Any) -> None:
        async with semaphore:
            results[index] = await worker(item)

    with Progress() as progress:
        task_id = progress.add_task(description, total=len(items))
        async def tracked(index: int, item: Any) -> None:
            await wrapped(index, item)
            progress.advance(task_id)

        await asyncio.gather(*(tracked(i, item) for i, item in enumerate(items)))
    return results


def question_length_without_punctuation(question: str) -> int:
    count = 0
    for ch in question:
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "S", "Z")):
            continue
        count += 1
    return count
