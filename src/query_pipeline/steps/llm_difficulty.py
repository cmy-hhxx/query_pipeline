from __future__ import annotations

import asyncio
import json
from typing import Any

from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.cache import append_cache, load_cache, make_cache_key
from query_pipeline.llm.client import LLMClient, load_prompt
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.records import parse_difficulty_response
from query_pipeline.pipeline.context import PipelineContext


def _build_user_payload(record: dict[str, Any]) -> str:
    useful = {
        "source": record.get("source"),
        "line_number": record.get("line_number"),
        "question": record.get("question"),
        "category_id": record.get("category_id"),
        "category_name": record.get("category_name"),
        "judge_reason": record.get("judge_reason"),
    }
    return "请标注以下 JSON 记录，并输出严格 json：\n" + json.dumps(
        useful,
        ensure_ascii=False,
        separators=(",", ":"),
    )


async def _label_difficulty(ctx: PipelineContext, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ctx.config.llm.difficulty.enabled:
        return records

    client = LLMClient(ctx.config.llm)
    system_prompt = load_prompt(ctx.config.llm.difficulty.prompt)
    cache = load_cache(ctx.config.llm.cache)
    lock = asyncio.Lock()

    async def worker(record: dict[str, Any]) -> dict[str, Any]:
        question = record["question"]
        cache_key = make_cache_key(question, step="difficulty")
        if cache_key in cache:
            result_fields = cache[cache_key]
        else:
            raw = await client.complete(
                system_prompt=system_prompt,
                user_prompt=_build_user_payload(record),
            )
            result_fields = parse_difficulty_response(raw).to_record_fields()
            async with lock:
                cache[cache_key] = result_fields
                append_cache(
                    ctx.config.llm.cache,
                    cache_key,
                    result_fields,
                    meta={"step": "difficulty", "question": question[:120]},
                )
        return {**record, **result_fields}

    try:
        return await run_concurrent(
            records,
            worker,
            concurrency=ctx.config.llm.concurrency,
            description="LLM difficulty",
        )
    finally:
        await client.close()


def run_llm_difficulty_step(ctx: PipelineContext) -> PipelineContext:
    eligible = [r for r in ctx.records if "skip_reason" not in r]
    skipped_only = [r for r in ctx.records if "skip_reason" in r]

    if eligible:
        labeled = asyncio.run(_label_difficulty(ctx, eligible))
        ctx.records = labeled + skipped_only
    else:
        ctx.records = skipped_only

    ctx.stats["llm_difficulty_rows"] = len(eligible)
    return ctx
