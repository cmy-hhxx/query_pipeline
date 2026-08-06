from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from query_pipeline.config.models import LLMStageConfig, Step2Config
from query_pipeline.llm.cache import append_cache, make_cache_key
from query_pipeline.llm.client import LLMClient
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.session import Segment, Step2Result, parse_step2_payload, parse_step2_response, prior_indices
from query_pipeline.prompts import resolve_prompt


def segment_of(segments: list[Segment], idx: int) -> Segment:
    for seg in segments:
        if seg.start <= idx <= seg.end:
            return seg
    raise IndexError(f"index {idx} not covered by any segment")


def build_judge_payload(turns: list[dict[str, Any]], segment: Segment, idx: int) -> dict[str, Any]:
    prior = [turns[k].get("question", "") for k in prior_indices(segment, idx)]
    return {"prior_questions": prior, "current_question": turns[idx].get("question", "")}


async def judge_candidates(
    *,
    client: LLMClient,
    turns: list[dict[str, Any]],
    segments: list[Segment],
    candidates: list[int],
    llm_cfg: LLMStageConfig,
    step2_cfg: Step2Config,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Judge each candidate turn via LLM. Returns one dict per candidate:
    {idx, is_complex, category_id, reason, error}. error is set on LLM/parse
    failure; is_complex is None then.
    """
    system_prompt = resolve_prompt(step2_cfg.prompt_id)
    lock = asyncio.Lock()

    async def worker(idx: int) -> dict[str, Any]:
        segment = segment_of(segments, idx)
        payload = build_judge_payload(turns, segment, idx)
        user_prompt = "请标注以下会话问句，只输出严格 JSON：\n" + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        )
        cache_key = make_cache_key(user_prompt, step=f"complex_judge:{step2_cfg.prompt_id}", model=llm_cfg.model)
        try:
            if cache_key in cache:
                parsed = parse_step2_payload(cache[cache_key])
            else:
                raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
                parsed = parse_step2_response(raw)
                label = parsed.to_cache_label()
                async with lock:
                    cache[cache_key] = label
                    append_cache(
                        cache_path,
                        cache_key,
                        label,
                        meta={
                            "step": "complex_judge",
                            "prompt_id": step2_cfg.prompt_id,
                            "model": llm_cfg.model,
                            "current_question": payload["current_question"][:120],
                        },
                    )
            return {
                "idx": idx,
                "is_complex": parsed.is_complex,
                "category_id": parsed.category_id,
                "reason": parsed.reason,
                "error": None,
            }
        except (ValueError, RuntimeError) as exc:
            return {"idx": idx, "is_complex": None, "category_id": None, "reason": None, "error": str(exc)[:200]}

    return await run_concurrent(
        candidates,
        worker,
        concurrency=llm_cfg.concurrency,
        description="LLM complex judge",
        show_progress=False,
    )


def is_complex_result(judged: dict[str, Any]) -> bool:
    return judged.get("is_complex") is True
