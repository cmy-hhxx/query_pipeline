from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

from query_pipeline.io.checkpoint import content_key, stage_checkpoint
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.cache import make_cache_key, put_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.session import parse_verify_payload, parse_verify_response
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.prompts import resolve_prompt


def _prior_questions(row: dict[str, Any]) -> list[str]:
    context = row.get("context") or []
    return [str(t.get("question") or "") for t in context if isinstance(t, dict)]


async def run_verify_stage(
    ctx: PipelineContext,
    client: LLMClient | None,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    """Context-aware multi-round re-check, rounds per difficulty.

    hard rows must stay complex in every round; normal rows must stay
    non-complex. LLM failures drop the row (fail-closed, admission bar is
    high) and are checkpointed so a resumed run replays the same verdict.
    """
    cfg = ctx.config
    ctx.prune_debug_artifacts("verified.jsonl")
    if not cfg.verify.enabled or client is None or not ctx.rows:
        return ctx

    checkpoint = stage_checkpoint(cfg, "verify")
    counts = {"kept": 0, "rejected": 0, "failed": 0}
    debug: list[dict[str, Any]] = []

    async def worker(row: dict[str, Any]) -> dict[str, Any]:
        inp = row.get("input")
        question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
        difficulty = row.get("difficulty_level", "hard")
        max_rounds = cfg.verify.max_rounds_hard if difficulty == "hard" else cfg.verify.max_rounds_normal
        key = content_key(str(row.get("source_case_id", "")), str(row.get("trace_id", "")), question)
        record = checkpoint.get(key)
        if record is not None:
            return {
                "keep": record["keep"],
                "reason": record["reason"],
                "error": record.get("error"),
                "rounds": record.get("rounds", []),
            }
        user_prompt = "请复核以下问句，只输出严格 JSON：\n" + json.dumps(
            {"prior_questions": _prior_questions(row), "question": question},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        rounds: list[dict[str, Any]] = []
        error: str | None = None
        reason: str | None = None
        expected = difficulty == "hard"
        keep = False
        for round_no in range(1, max_rounds + 1):
            # 单视角级联从严：round 1 主判定，round >= 2 逐轮从严复核。
            # 实测：双视角制衡（复杂度判定 + 简单识别器）对灰色地带的表现
            # 不如单一"宁缺毋滥"从严判定，故不引入第二视角。
            if round_no == 1:
                prompt_id = cfg.verify.prompt_id
                system_prompt = resolve_prompt(prompt_id)
            else:
                prompt_id = "verify_recheck"
                system_prompt = resolve_prompt(prompt_id).format(round_no=round_no)
            cache_key = make_cache_key(
                user_prompt, step=f"verify:{prompt_id}", model=cfg.llm.model, prompt=system_prompt
            )
            try:
                if cache_key in cache:
                    parsed = parse_verify_payload(cache[cache_key])
                else:
                    raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
                    parsed = parse_verify_response(raw)
                    await put_cache(
                        cache,
                        cfg.cache_path,
                        cache_key,
                        parsed.to_cache_label(),
                        meta={
                            "step": "verify",
                            "prompt_id": prompt_id,
                            "round": round_no,
                            "model": cfg.llm.model,
                            "question": question[:120],
                        },
                        lock=cache_lock,
                    )
            except (ValueError, RuntimeError) as exc:
                error = str(exc)[:200]
                break
            reason = parsed.reason
            rounds.append(
                {
                    "round": round_no,
                    "prompt_id": prompt_id,
                    "is_complex": parsed.is_complex,
                    "reason": parsed.reason,
                }
            )
            if parsed.is_complex != expected:
                break
            keep = round_no == max_rounds
        # Fail-closed: errored rows are dropped (keep=False) and checkpointed so a
        # resumed run replays the drop instead of re-verifying.
        await checkpoint.mark(key, keep=keep, reason=reason, error=error, rounds=rounds)
        return {"keep": keep, "reason": reason, "error": error, "rounds": rounds}

    results = await run_concurrent(
        ctx.rows,
        worker,
        concurrency=cfg.llm.concurrency,
        description="LLM verify",
    )

    kept: list[dict[str, Any]] = []
    for row, result in zip(ctx.rows, results):
        keep, reason, error = result["keep"], result["reason"], result["error"]
        inp = row.get("input")
        question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
        debug.append(
            {
                "source_case_id": row.get("source_case_id", ""),
                "trace_id": row.get("trace_id", ""),
                "difficulty": row.get("difficulty_level", ""),
                "category": row.get("category", ""),
                "question": question[:200],
                "is_complex": keep,
                "reason": reason,
                "error": error,
                "rounds": result.get("rounds", []),
            }
        )
        if error is not None:
            counts["failed"] += 1
        elif keep:
            counts["kept"] += 1
            kept.append(row)
        else:
            counts["rejected"] += 1

    ctx.rows = kept
    logger.info(
        "[verify] kept=%d rejected=%d failed=%d",
        counts["kept"], counts["rejected"], counts["failed"],
    )
    ctx.stats["verify_kept"] = counts["kept"]
    ctx.stats["verify_rejected"] = counts["rejected"]
    ctx.stats["verify_failed"] = counts["failed"]
    if debug and cfg.debug.dump_intermediates:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(ctx.path("verified.jsonl"), debug)
    return ctx
