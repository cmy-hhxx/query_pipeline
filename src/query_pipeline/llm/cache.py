from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    skipped = 0
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Torn trailing line from a hard-killed run: drop it and let
                # the next run re-do that one LLM call instead of crashing.
                skipped += 1
                continue
            key = row.get("cache_key")
            label = row.get("label")
            if not isinstance(key, str) or not isinstance(label, dict):
                skipped += 1
                continue
            cache[key] = label
    if skipped:
        logger.warning("llm cache: dropped %d unparseable line(s) in %s", skipped, cache_path)
    return cache


def append_cache(cache_path: Path, cache_key: str, label: dict[str, Any], *, meta: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"cache_key": cache_key, "label": label, **meta}
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


async def put_cache(
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    cache_key: str,
    label: dict[str, Any],
    *,
    meta: dict[str, Any],
    lock: asyncio.Lock,
) -> None:
    """Atomically publish a cache entry to memory + disk under a shared lock.

    Concurrent sessions share one cache file; without serialization, appends of
    long JSON lines can tear. If another writer already stored the same key,
    leave their entry (avoid duplicate appends for the common race).
    """
    async with lock:
        if cache_key in cache:
            return
        cache[cache_key] = label
        append_cache(cache_path, cache_key, label, meta=meta)


def make_cache_key(question: str, *, step: str, model: str, prompt: str = "") -> str:
    # Include the system prompt so a prompt change invalidates stale cached
    # results (otherwise old labels are reused for the new instructions).
    material = (prompt + "\n") + question if prompt else question
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{step}:{model}:{digest}"
