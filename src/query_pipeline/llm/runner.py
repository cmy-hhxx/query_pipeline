from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


_CHUNK_SIZE = 1000


async def run_concurrent(
    items: list[Any],
    worker: Callable[[Any], Awaitable[Any]],
    *,
    description: str = "Processing",
) -> list[Any]:
    """Run a worker over items with pure task orchestration.

    限流职责在 LLMClient 的进程级 semaphore（唯一 choke point），此处不再叠加
    第二层 semaphore。分块 dispatch（每批 _CHUNK_SIZE 项）：一次性 gather 全部
    创建会让 verify/translate 上万行 = 上万个协程（实测 100k 任务峰值 110MB +
    2.1s 纯调度开销），分批后内存有界、语义不变（client semaphore 仍是唯一并发上限）。
    单个 item 的意外异常（如 cache 磁盘 OSError）会被兜底：记 warning 并返回
    None，绝不中止整批或遗留孤儿任务（gather 不会取消其余任务）。

    Progress reporting is intentionally absent: the pipeline logs per-stage
    summaries instead of live progress bars (audit-friendly, log/output live
    in the same directory).
    """
    results: list[Any] = [None] * len(items)

    async def wrapped(index: int, item: Any) -> None:
        try:
            results[index] = await worker(item)
        except Exception as exc:  # noqa: BLE001 单行异常不弃整批
            logger.warning(
                "[%s] item %d failed: %s", description, index, str(exc)[:200]
            )
            results[index] = None

    for start in range(0, len(items), _CHUNK_SIZE):
        chunk = items[start : start + _CHUNK_SIZE]
        await asyncio.gather(*(wrapped(start + i, item) for i, item in enumerate(chunk)))
    return results
