from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

from query_pipeline.llm.cache import make_cache_key, put_cache
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.session import parse_json_object
from query_pipeline.quality.aggregate import record_key
from query_pipeline.quality.prompts import QC_STEP, build_judge_system_prompt, build_judge_user_prompt


class JudgeResult(BaseModel):
    question_quality: str
    label_ok: bool
    reason: str = ""

    @field_validator("question_quality")
    @classmethod
    def _validate_quality(cls, value: str) -> str:
        if value not in {"high", "low"}:
            raise ValueError(f"invalid question_quality: {value!r}")
        return value


def select_sample(records: list[dict[str, Any]], *, ratio: float, seed: int) -> list[int]:
    """Deterministic uniform-random sample of record indices (empty when ratio<=0)."""
    if ratio <= 0 or not records:
        return []
    n = max(1, round(len(records) * ratio))
    n = min(n, len(records))
    return sorted(random.Random(seed).sample(range(len(records)), n))


async def judge_one(
    row: dict[str, Any],
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
    cache_path: Any,
    *,
    system_prompt: str,
    model: str,
) -> dict[str, Any]:
    """One LLM call for a sampled record; never raises for LLM/parse failures."""
    # join key must match aggregate.record_key so verdicts fold onto the right rows
    key = record_key(row)
    trace_id = str(row.get("trace_id") or "")
    user_prompt = build_judge_user_prompt(row)
    cache_key = make_cache_key(user_prompt, step=QC_STEP, model=model, prompt=system_prompt)
    try:
        parsed: JudgeResult | None = None
        if cache_key in cache:
            try:
                parsed = JudgeResult.model_validate(cache[cache_key])
            except (ValueError, RuntimeError) as exc:
                # 坏缓存 label（跨版本 schema/手改缓存）：驱逐并重调，避免每次运行重复误判
                logger.warning("cached QC judge label invalid, re-calling LLM: %s", str(exc)[:120])
                cache.pop(cache_key, None)
        if parsed is None:
            raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            parsed = JudgeResult.model_validate(parse_json_object(raw))
            await put_cache(
                cache,
                cache_path,
                cache_key,
                {
                    "question_quality": parsed.question_quality,
                    "label_ok": parsed.label_ok,
                    "reason": parsed.reason,
                },
                meta={"step": QC_STEP, "model": model, "trace_id": trace_id},
                lock=cache_lock,
            )
        return {
            "trace_id": key,
            "question_quality": parsed.question_quality,
            "label_ok": parsed.label_ok,
            "reason": parsed.reason,
            "error": None,
        }
    except (ValueError, RuntimeError) as exc:
        return {
            "trace_id": key,
            "question_quality": None,
            "label_ok": None,
            "reason": "",
            "error": str(exc)[:200],
        }


async def run_llm_judge(
    records: list[dict[str, Any]],
    client: Any,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
    cache_path: Any,
    *,
    ratio: float,
    seed: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Sample records and LLM-judge each; returns (sample_indices, verdicts)."""
    sample_indices = select_sample(records, ratio=ratio, seed=seed)
    if not sample_indices:
        return [], []
    sampled_rows = [records[i] for i in sample_indices]
    system_prompt = build_judge_system_prompt()
    model = client.config.model
    verdicts = await run_concurrent(
        sampled_rows,
        lambda row: judge_one(
            row, client, cache, cache_lock, cache_path,
            system_prompt=system_prompt, model=model,
        ),
        description="LLM 抽检判定",
    )
    # run_concurrent 兜底网返回 None（judge_one 意外异常）：不得静默丢弃——否则
    # sample_set 仍含该行而 verdict 缺失，build_results 会把它算成 pass（fail-open），
    # 与 judge_one 内部异常（error → needs_review）结论相反。映射为 error 语义。
    verdicts = [
        v
        if v is not None
        else {
            "trace_id": record_key(row),
            "question_quality": None,
            "label_ok": None,
            "reason": "",
            "error": "judge 意外失败（run_concurrent 兜底网）",
        }
        for v, row in zip(verdicts, sampled_rows)
    ]
    return sample_indices, verdicts
