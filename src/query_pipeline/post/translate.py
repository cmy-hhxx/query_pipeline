from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from query_pipeline.config.models import LLMConfig
from query_pipeline.io.checkpoint import Checkpoint, content_key
from query_pipeline.llm.cache import make_cache_key, put_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.session import parse_json_object
from query_pipeline.prompts import resolve_prompt

_CJK = re.compile(r"[一-鿿]")


def needs_translation(text: str, *, cjk_ratio: float = 0.3) -> bool:
    if not text.strip():
        return False
    return len(_CJK.findall(text)) / len(text) < cjk_ratio


def _row_text(row: dict[str, Any]) -> str:
    inp = row.get("input")
    if isinstance(inp, dict):
        return str(inp.get("text") or "")
    return ""


async def translate_rows(
    rows: list[dict[str, Any]],
    *,
    client: LLMClient,
    llm_cfg: LLMConfig,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    checkpoint: Checkpoint | None = None,
    cache_lock: asyncio.Lock | None = None,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    system_prompt = resolve_prompt("translate")
    lock = cache_lock or asyncio.Lock()
    # counts[...] += 1 is de-facto atomic: no `await` between load and store, and asyncio
    # is single-threaded/cooperative, so the increments can never interleave. No lock needed.
    counts = {"translated": 0, "translate_skipped": 0, "translate_failed": 0}
    checkpoint = checkpoint or Checkpoint.disabled()
    completion_errors: list[Exception] = []

    def complete(row: dict[str, Any]) -> None:
        if on_complete is None:
            return
        try:
            on_complete(row)
        except Exception as exc:  # preserved and re-raised after concurrent workers settle
            completion_errors.append(exc)
            raise

    def put(row: dict[str, Any], translation: str | None, *, failed: bool = False) -> None:
        # translation 是顶层唯一字段：只有非中文问句翻译成功才填译文；
        # 原文已是中文、或翻译失败 → null（见 templates/filter_out.jsonc）。
        row["translation"] = translation
        if failed:
            # 失败标记（meta 逃生字段）：QC 的 meta 规则据此区分"翻译失败"
            # （fail-open 可接受）与"从未翻译"（null 且无标记 → fail）。
            meta = row.get("meta")
            if not isinstance(meta, dict):
                meta = {}
                row["meta"] = meta
            meta["translate_failed"] = True

    async def worker(row: dict[str, Any]) -> bool:
        # 返回 True 表示处理完成（成功/跳过/失败均已计数）；
        # run_concurrent 兜底网捕获的意外异常返回 None，由调用方补记 failed。
        text = _row_text(row)
        key = content_key(text)
        record = checkpoint.get(key)
        if record is not None:
            put(row, record["translation"])
            counts["translate_skipped" if record["skipped"] else "translated"] += 1
            complete(row)
            return True
        if not needs_translation(text):
            put(row, None)  # 原文已是中文：不需要翻译 → null
            counts["translate_skipped"] += 1
            try:
                await checkpoint.mark(key, translation=None, skipped=True)
            except OSError as exc:
                # 行已处理完，checkpoint 失败只影响下次重跑（llm_cache/规则可复用）：
                # 不得把已计数行再补记 failed（双计数）。
                logger.warning("translate checkpoint mark failed: %s", exc)
            complete(row)
            return True
        user_prompt = "请翻译以下用户问句，只输出严格 JSON：\n" + json.dumps(
            {"text": text}, ensure_ascii=False, separators=(",", ":")
        )
        cache_key = make_cache_key(user_prompt, step="translate", model=llm_cfg.model, prompt=system_prompt)
        try:
            translation: str | None = None
            if cache_key in cache:
                cached = cache[cache_key].get("translation")
                if isinstance(cached, str) and cached.strip():
                    translation = cached
                else:
                    # 坏缓存 label：驱逐并重调，避免每次运行重复失败（与 segment 自愈一致）
                    logger.warning("cached translate label invalid, re-calling LLM")
                    cache.pop(cache_key, None)
            if translation is None:
                raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
                translation = parse_translation(raw)
                await put_cache(
                    cache,
                    cache_path,
                    cache_key,
                    {"translation": translation},
                    meta={"step": "translate", "model": llm_cfg.model, "text": text[:120]},
                    lock=lock,
                )
            put(row, translation)
            try:
                await checkpoint.mark(key, translation=translation, skipped=False)
            except OSError as exc:
                # 同 skipped 分支：行已翻译完成，checkpoint 失败不得触发双计数
                # （旧行为：OSError 穿透到 run_concurrent 兜底网 → translated + failed 同增）。
                logger.warning("translate checkpoint mark failed: %s", exc)
            counts["translated"] += 1
        except (ValueError, RuntimeError):
            # 翻译失败：已有译文必须保留——无条件覆盖为 null 会让"缓存丢失后重跑
            # 失败"静默抹掉上次成功译文，且 QC 因 translate_failed 标记放行，
            # 译文丢失被掩盖。仅当本行没有译文时才置 null + 失败标记
            # （fail-open 保留行，下轮重试）。
            if not isinstance(row.get("translation"), str) or not row["translation"].strip():
                put(row, None, failed=True)
            counts["translate_failed"] += 1
        complete(row)
        return True

    results = await run_concurrent(rows, worker, description="LLM translate")
    if completion_errors:
        raise completion_errors[0]
    for row, result in zip(rows, results):
        if result is None:  # 兜底网捕获的意外异常（如 cache 磁盘 OSError）
            if not isinstance(row.get("translation"), str) or not row["translation"].strip():
                put(row, None, failed=True)  # 已成功的行不覆盖译文，只补记失败
            counts["translate_failed"] += 1
            complete(row)
    return counts


def parse_translation(raw: str) -> str:
    translation = parse_json_object(raw).get("translation")
    if not isinstance(translation, str) or not translation.strip():
        raise ValueError("missing translation in response")
    return translation
