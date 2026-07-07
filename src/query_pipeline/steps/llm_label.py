from __future__ import annotations

import asyncio
import json
from typing import Any

from query_pipeline.llm.cache import append_cache, load_cache, make_cache_key
from query_pipeline.llm.client import LLMClient
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.records import parse_unified_label_response
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.pipeline.records import normalized_text, set_pipeline_output
from query_pipeline.prompts import resolve_prompt


def run_llm_label_stage(ctx: PipelineContext) -> PipelineContext:
    if not ctx.config.llm_stage.enabled:
        ctx.records = [set_pipeline_output(record, status="accepted") for record in ctx.records]
        ctx.stats["llm_label_rows"] = 0
        return ctx

    if not ctx.records:
        ctx.stats["llm_label_rows"] = 0
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

    async def worker(record: dict[str, Any]) -> dict[str, Any]:
        text = normalized_text(record)
        cache_key = make_cache_key(text, step=f"unified_label:{cfg.prompt_id}")
        if cache_key in cache:
            label = cache[cache_key]
        else:
            raw = await client.complete(system_prompt=system_prompt, user_prompt=_build_user_payload(record))
            label = parse_unified_label_response(raw).to_output()
            async with lock:
                cache[cache_key] = label
                append_cache(
                    cfg.cache,
                    cache_key,
                    label,
                    meta={"step": "unified_label", "prompt_id": cfg.prompt_id, "text": text[:120]},
                )
        return set_pipeline_output(record, status="accepted", llm_label=label)

    try:
        return await run_concurrent(records, worker, concurrency=cfg.concurrency, description="LLM unified label")
    finally:
        await client.close()


def _build_user_payload(record: dict[str, Any]) -> str:
    payload = {
        "normalized_text": normalized_text(record),
    }
    nlu_reference = record.get("nlu_reference", record.get("nlu", record.get("NLU")))
    if nlu_reference is not None:
        payload["nlu_reference"] = nlu_reference
    return "请标注以下 JSON 记录，只输出严格 JSON：\n" + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
