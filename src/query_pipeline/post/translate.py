from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

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
) -> dict[str, int]:
    system_prompt = resolve_prompt("translate")
    lock = cache_lock or asyncio.Lock()
    # counts[...] += 1 is de-facto atomic: no `await` between load and store, and asyncio
    # is single-threaded/cooperative, so the increments can never interleave. No lock needed.
    counts = {"translated": 0, "translate_skipped": 0, "translate_failed": 0}
    checkpoint = checkpoint or Checkpoint.disabled()

    def put(row: dict[str, Any], translation: str | None) -> None:
        # translation 是顶层唯一字段：只有非中文问句翻译成功才填译文；
        # 原文已是中文、或翻译失败 → null（见 templates/filter_out.jsonc）。
        row["translation"] = translation

    async def worker(row: dict[str, Any]) -> None:
        text = _row_text(row)
        key = content_key(text)
        record = checkpoint.get(key)
        if record is not None:
            put(row, record["translation"])
            counts["translate_skipped" if record["skipped"] else "translated"] += 1
            return
        if not needs_translation(text):
            put(row, None)  # 原文已是中文：不需要翻译 → null
            counts["translate_skipped"] += 1
            await checkpoint.mark(key, translation=None, skipped=True)
            return
        user_prompt = "请翻译以下用户问句，只输出严格 JSON：\n" + json.dumps(
            {"text": text}, ensure_ascii=False, separators=(",", ":")
        )
        cache_key = make_cache_key(user_prompt, step="translate", model=llm_cfg.model, prompt=system_prompt)
        try:
            if cache_key in cache:
                translation = cache[cache_key].get("translation")
                if not isinstance(translation, str) or not translation.strip():
                    raise ValueError("cached translation is invalid")
            else:
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
            counts["translated"] += 1
            await checkpoint.mark(key, translation=translation, skipped=False)
        except (ValueError, RuntimeError):
            put(row, None)  # 翻译失败：无译文 → null（fail-open 保留行，下轮重试）
            counts["translate_failed"] += 1

    await run_concurrent(
        rows, worker, concurrency=llm_cfg.concurrency, description="LLM translate", show_progress=True
    )
    return counts


def parse_translation(raw: str) -> str:
    translation = parse_json_object(raw).get("translation")
    if not isinstance(translation, str) or not translation.strip():
        raise ValueError("missing translation in response")
    return translation
