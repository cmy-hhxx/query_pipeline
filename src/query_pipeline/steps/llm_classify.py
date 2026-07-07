from __future__ import annotations

import asyncio
import json
from typing import Any

from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.cache import append_cache, load_cache, make_cache_key
from query_pipeline.llm.client import LLMClient, load_prompt
from query_pipeline.llm.runner import question_length_without_punctuation, run_concurrent
from query_pipeline.models.records import parse_classify_response
from query_pipeline.pipeline.context import PipelineContext


def _eligible_for_classify(ctx: PipelineContext, record: dict[str, Any]) -> bool:
    cfg = ctx.config.llm.classify
    if record.get("complexity_score", 0) < cfg.min_complexity_score:
        return False
    return question_length_without_punctuation(record["question"]) >= cfg.min_question_length


async def _classify_records(ctx: PipelineContext, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ctx.config.llm.classify.enabled:
        return records

    client = LLMClient(ctx.config.llm)
    system_prompt = load_prompt(ctx.config.llm.classify.prompt)
    cache = load_cache(ctx.config.llm.cache)
    lock = asyncio.Lock()

    async def worker(record: dict[str, Any]) -> dict[str, Any]:
        question = record["question"]
        cache_key = make_cache_key(question, step="classify")
        if cache_key in cache:
            result_fields = cache[cache_key]
        else:
            raw = await client.complete(system_prompt=system_prompt, user_prompt=question)
            result_fields = parse_classify_response(raw).to_record_fields()
            async with lock:
                cache[cache_key] = result_fields
                append_cache(
                    ctx.config.llm.cache,
                    cache_key,
                    result_fields,
                    meta={"step": "classify", "question": question[:120]},
                )
        return {**record, **result_fields}

    try:
        results = await run_concurrent(
            records,
            worker,
            concurrency=ctx.config.llm.concurrency,
            description="LLM classify",
        )
        return results
    finally:
        await client.close()


def run_llm_classify_step(ctx: PipelineContext) -> PipelineContext:
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in ctx.records:
        if _eligible_for_classify(ctx, record):
            eligible.append(record)
        else:
            skipped.append({**record, "skip_reason": "low_complexity_score_or_short"})
    ctx.skipped.extend(skipped)

    if eligible:
        classified = asyncio.run(_classify_records(ctx, eligible))
        ctx.records = classified + skipped
    else:
        ctx.records = skipped

    ctx.stats["llm_classify_eligible_rows"] = len(eligible)
    ctx.stats["llm_classify_skipped_rows"] = len(skipped)
    write_jsonl(ctx.path("llm_classify.jsonl"), ctx.records)
    return ctx
