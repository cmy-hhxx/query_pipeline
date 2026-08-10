from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def src_hash() -> str:
    """Hash of all pipeline source code; behavior fixes must invalidate LLM cache.

    Hashes the whole src/query_pipeline tree (not a per-stage list) so the
    fingerprint can never silently forget the modules a stage depends on.
    checkpoint 的 stage_fingerprint 复用同一哈希，缓存与断点失效策略一致。
    """
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()


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
    # Include the source fingerprint so "改代码不改 prompt" 的修复（parse 规则、
    # taxonomy 映射、难度判定等）也不会跨运行静默复用旧 label——与 checkpoint
    # 的失效策略一致（README：配置/输入/源码变化自动失效）。
    material = f"{src_hash()}\n{prompt}\n{question}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{step}:{model}:{digest}"
