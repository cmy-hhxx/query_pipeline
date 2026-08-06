from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from query_pipeline.config.models import LLMStageConfig, TranslateConfig
from query_pipeline.llm.cache import append_cache, make_cache_key
from query_pipeline.llm.client import LLMClient
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.prompts import resolve_prompt

_CJK = re.compile(r"[一-鿿]")

_TARGET_LABELS: dict[str, str] = {"zh": "简体中文", "en": "英文"}


def needs_translation(text: str, *, cjk_ratio: float = 0.3) -> bool:
    """True unless the text is empty or already predominantly CJK (target zh)."""
    if not text.strip():
        return False
    return len(_CJK.findall(text)) / len(text) < cjk_ratio


def _target_label(target: str) -> str:
    return _TARGET_LABELS.get(target, target)


async def translate_rows(
    rows: list[dict[str, Any]],
    *,
    client: LLMClient,
    llm_cfg: LLMStageConfig,
    translate_cfg: TranslateConfig,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
) -> dict[str, int]:
    """Fill meta.translation for each row's input.text.

    Already-Chinese rows are kept as-is (no LLM call); LLM failures fall back
    to the original text so downstream always has a meta.translation value.
    Returns {translated, translate_skipped, translate_failed}.
    """
    system_prompt = resolve_prompt("translate").format(target=_target_label(translate_cfg.target))
    lock = asyncio.Lock()
    counts = {"translated": 0, "translate_skipped": 0, "translate_failed": 0}

    def put(row: dict[str, Any], translation: str) -> None:
        row.setdefault("meta", {})["translation"] = translation

    async def worker(row: dict[str, Any]) -> None:
        text = row.get("input", {}).get("text", "") if isinstance(row.get("input"), dict) else ""
        if not needs_translation(text):
            put(row, text)
            counts["translate_skipped"] += 1
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
                async with lock:
                    cache[cache_key] = {"translation": translation}
                    append_cache(
                        cache_path,
                        cache_key,
                        {"translation": translation},
                        meta={
                            "step": "translate",
                            "target": translate_cfg.target,
                            "model": llm_cfg.model,
                            "text": text[:120],
                        },
                    )
            put(row, translation)
            counts["translated"] += 1
        except (ValueError, RuntimeError):
            put(row, text)
            counts["translate_failed"] += 1

    await run_concurrent(
        rows, worker, concurrency=llm_cfg.concurrency, description="LLM translate", show_progress=False
    )
    return counts


def parse_translation(raw: str) -> str:
    translation = json.loads(raw).get("translation")
    if not isinstance(translation, str) or not translation.strip():
        raise ValueError("missing translation in response")
    return translation
