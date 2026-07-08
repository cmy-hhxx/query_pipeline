from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

from query_pipeline.llm.cache import append_cache, load_cache, make_cache_key
from query_pipeline.llm.client import LLMClient
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.records import CoreLabelResult, parse_core_label_payload, parse_core_label_response
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.pipeline.records import normalized_text, set_pipeline_output
from query_pipeline.prompts import resolve_prompt


def run_llm_label_stage(ctx: PipelineContext) -> PipelineContext:
    if not ctx.config.llm_stage.enabled:
        ctx.records = [set_pipeline_output(record, status="accepted") for record in ctx.records]
        ctx.stats["llm_label_rows"] = 0
        ctx.stats["llm_extra_field_rows"] = 0
        ctx.stats["llm_extra_fields"] = {}
        return ctx

    if not ctx.records:
        ctx.stats["llm_label_rows"] = 0
        ctx.stats["llm_extra_field_rows"] = 0
        ctx.stats["llm_extra_fields"] = {}
        return ctx

    ctx.records = asyncio.run(_label_records(ctx, ctx.records))
    ctx.stats["llm_label_rows"] = len(ctx.records)
    return ctx


async def _label_records(ctx: PipelineContext, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = ctx.config.llm_stage
    client = LLMClient(cfg)
    system_prompt = resolve_prompt(cfg.prompt_id)
    cache = load_cache(cfg.cache)
    lock = asyncio.Lock()
    extra_field_counts: Counter[str] = Counter()
    extra_field_rows = 0

    async def record_extra_fields(label: CoreLabelResult) -> None:
        nonlocal extra_field_rows
        if not label.extra_fields:
            return
        async with lock:
            extra_field_rows += 1
            extra_field_counts.update(label.extra_fields)

    async def worker(record: dict[str, Any]) -> dict[str, Any]:
        text = normalized_text(record)
        cache_key = make_cache_key(text, step=f"core_label:{cfg.prompt_id}")
        if cache_key in cache:
            parsed = parse_core_label_payload(cache[cache_key])
            await record_extra_fields(parsed)
            label = parsed.to_output()
        else:
            raw = await client.complete(system_prompt=system_prompt, user_prompt=_build_user_payload(record))
            parsed = parse_core_label_response(raw)
            await record_extra_fields(parsed)
            label = parsed.to_output()
            cache_label = parsed.to_cache_label()
            async with lock:
                cache[cache_key] = cache_label
                append_cache(
                    cfg.cache,
                    cache_key,
                    cache_label,
                    meta={"step": "core_label", "prompt_id": cfg.prompt_id, "text": text[:120]},
                )
        return set_pipeline_output(record, status="accepted", llm_label=label)

    try:
        labeled = await run_concurrent(records, worker, concurrency=cfg.concurrency, description="LLM core label")
        ctx.stats["llm_extra_field_rows"] = extra_field_rows
        ctx.stats["llm_extra_fields"] = dict(sorted(extra_field_counts.items()))
        return labeled
    finally:
        await client.close()


def _build_user_payload(record: dict[str, Any]) -> str:
    payload = {
        "normalized_text": normalized_text(record),
    }
    return "请标注以下 JSON 记录，只输出严格 JSON：\n" + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
