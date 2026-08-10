from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX (Windows)
    fcntl = None

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


@contextlib.contextmanager
def _file_lock(cache_path: Path):
    """Cross-process advisory lock serializing cache rewrite vs append.

    Pipeline runner 与 QC CLI 共用同一 cache 文件；孤儿代 rewrite（tmp+replace）
    与另一进程的 append 并发时：replace 会换掉旧 inode，旧 inode 上刚 append 的
    条目静默丢失，且固定 tmp 名会让两个进程互相截断对方的 rewrite。flock 是
    进程级锁（lock 文件），rewrite 与 append 都持锁执行，互斥后无丢失窗口。
    """
    lock_path = cache_path.with_name(cache_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    # 读 + rewrite 全程持跨进程锁：另一进程的 append 等待，rewrite 的 replace
    # 不会丢掉并发 append 的条目（B 的 replace 只能在 A 的 append 之后生效）。
    with _file_lock(cache_path):
        return _load_cache_locked(cache_path)


def _load_cache_locked(cache_path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    skipped = 0
    orphaned = 0
    current = src_hash()
    rows: list[dict[str, Any]] = []
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
            if row.get("src") != current:
                # 孤儿代：src_hash 已进 cache key，任何代码改动都会让全部旧 key
                # 永久失效；缓存是 append-only，永不清理则文件只增不减、每次 run
                # 全量解析死代条目。启动时检测到孤儿代就一次性 rewrite 压缩。
                orphaned += 1
                continue
            cache[key] = label
            rows.append(row)
    if skipped:
        logger.warning("llm cache: dropped %d unparseable line(s) in %s", skipped, cache_path)
    if orphaned:
        logger.warning(
            "llm cache: %d orphaned entry(ies) from an outdated source generation dropped; rewriting %s",
            orphaned,
            cache_path,
        )
        _rewrite_cache(cache_path, rows)
    return cache


def _rewrite_cache(cache_path: Path, rows: list[dict[str, Any]]) -> None:
    """Rewrite the cache file atomically (tmp + replace) keeping only live rows.

    调用方必须已持有 _file_lock（load_cache 全程持锁）；tmp 名带 pid+随机后缀，
    即使两个进程同时走到 rewrite（理论上被锁互斥），也不会共用同一 tmp 互相截断。
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(tmp, cache_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def append_cache(cache_path: Path, cache_key: str, label: dict[str, Any], *, meta: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # src 代标记：load_cache 靠它识别孤儿代并触发压缩（旧代条目 key 永不命中）。
    row = {"cache_key": cache_key, "label": label, "src": src_hash(), **meta}
    # 跨进程锁：与另一进程的孤儿代 rewrite 互斥，append 永不写进将被 replace 的旧 inode。
    with _file_lock(cache_path):
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
