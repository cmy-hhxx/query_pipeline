from __future__ import annotations

import asyncio
import random
from typing import Any

from pydantic import BaseModel, field_validator

from query_pipeline.llm.cache import make_cache_key, put_cache
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.session import parse_json_object
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
    trace_id = str(row.get("trace_id") or "")
    user_prompt = build_judge_user_prompt(row)
    cache_key = make_cache_key(user_prompt, step=QC_STEP, model=model, prompt=system_prompt)
    try:
        if cache_key in cache:
            parsed = JudgeResult.model_validate(cache[cache_key])
        else:
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
            "trace_id": trace_id,
            "question_quality": parsed.question_quality,
            "label_ok": parsed.label_ok,
            "reason": parsed.reason,
            "error": None,
        }
    except (ValueError, RuntimeError) as exc:
        return {
            "trace_id": trace_id,
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
    concurrency: int,
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
        concurrency=concurrency,
        description="LLM 抽检判定",
    )
    return sample_indices, verdicts
