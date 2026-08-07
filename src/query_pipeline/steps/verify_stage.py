from __future__ import annotations

import asyncio
import json
from typing import Any

from query_pipeline.io.checkpoint import content_key, stage_checkpoint
from query_pipeline.llm.cache import load_cache, make_cache_key, put_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.session import parse_verify_payload, parse_verify_response
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.prompts import resolve_prompt


def run_verify_stage(ctx: PipelineContext) -> PipelineContext:
    """Second-pass verification: judge each exported question standalone.

    Pass 1 judged questions *with* same-segment context, which lets
    connective short turns ride on rich context. This stage re-judges the
    bare question (no context) so only questions that are complex on their
    own survive. LLM failures keep the row (fail-open), are counted, and are
    checkpointed sticky so a resume cannot later flip a kept row to rejected.
    """
    cfg = ctx.config
    if not cfg.verify_stage.enabled or not cfg.llm_stage.enabled or not ctx.rows:
        return ctx

    kept, counts, debug = asyncio.run(_verify_all(ctx, ctx.rows))
    ctx.rows = kept
    ctx.stats["verify_kept"] = counts["kept"]
    ctx.stats["verify_rejected"] = counts["rejected"]
    ctx.stats["verify_failed"] = counts["failed"]
    if debug:
        from query_pipeline.io.jsonl import write_jsonl

        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(ctx.path("verified.jsonl"), debug)
    return ctx


async def _verify_all(
    ctx: PipelineContext, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    cfg = ctx.config
    system_prompt = resolve_prompt(cfg.verify_stage.prompt_id)
    client = LLMClient(cfg.llm_stage)
    cache = load_cache(cfg.llm_stage.cache)
    checkpoint = stage_checkpoint(cfg, "verify")
    lock = asyncio.Lock()
    counts = {"kept": 0, "rejected": 0, "failed": 0}
    debug: list[dict[str, Any]] = []

    async def worker(row: dict[str, Any]) -> dict[str, Any]:
        inp = row.get("input")
        question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
        key = content_key(str(row.get("source_case_id", "")), str(row.get("trace_id", "")), question)
        record = checkpoint.get(key)
        if record is not None:
            return {"keep": record["keep"], "reason": record["reason"], "error": record.get("error")}
        user_prompt = "请判断以下单个问句是否属于复杂金融问句，只输出严格 JSON：\n" + json.dumps(
            {"question": question}, ensure_ascii=False, separators=(",", ":")
        )
        cache_key = make_cache_key(
            user_prompt, step=f"verify:{cfg.verify_stage.prompt_id}", model=cfg.llm_stage.model, prompt=system_prompt
        )
        try:
            if cache_key in cache:
                parsed = parse_verify_payload(cache[cache_key])
            else:
                raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
                parsed = parse_verify_response(raw)
                await put_cache(
                    cache,
                    cfg.llm_stage.cache,
                    cache_key,
                    parsed.to_cache_label(),
                    meta={
                        "step": "verify",
                        "prompt_id": cfg.verify_stage.prompt_id,
                        "model": cfg.llm_stage.model,
                        "question": question[:120],
                    },
                    lock=lock,
                )
            result = {"keep": parsed.is_complex, "reason": parsed.reason, "error": None}
            await checkpoint.mark(key, keep=result["keep"], reason=result["reason"], error=None)
            return result
        except (ValueError, RuntimeError) as exc:
            # Sticky fail-open: persist keep=True so resume cannot drop a row
            # that a completed prior run already emitted.
            error = str(exc)[:200]
            await checkpoint.mark(key, keep=True, reason=None, error=error)
            return {"keep": True, "reason": None, "error": error}

    results = await run_concurrent(
        rows,
        worker,
        concurrency=cfg.llm_stage.concurrency,
        description="LLM verify",
        show_progress=True,
    )
    await client.close()

    kept: list[dict[str, Any]] = []
    for row, result in zip(rows, results):
        keep, reason, error = result["keep"], result["reason"], result["error"]
        inp = row.get("input")
        question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
        debug.append(
            {
                "source_case_id": row.get("source_case_id", ""),
                "trace_id": row.get("trace_id", ""),
                "category": row.get("category", ""),
                "question": question[:200],
                "is_complex": keep,
                "reason": reason,
                "error": error,
            }
        )
        if error is not None:
            counts["failed"] += 1
            kept.append(row)
        elif keep:
            counts["kept"] += 1
            kept.append(row)
        else:
            counts["rejected"] += 1
    return kept, counts, debug
