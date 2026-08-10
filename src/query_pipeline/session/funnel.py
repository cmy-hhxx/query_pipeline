"""Decoupled funnel per candidate: value gate -> complexity gate -> classify.

Each step is a separate LLM call with its own cache key. Failures drop the
candidate (fail-closed); the funnel only assembles rows for survivors.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

from openai import APIStatusError
from pydantic import BaseModel, field_validator

from query_pipeline.config.models import LLMConfig
from query_pipeline.llm.cache import make_cache_key, put_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.models.session import parse_json_object
from query_pipeline.prompts import resolve_prompt
from query_pipeline.session.judge import build_judge_payload
from query_pipeline.taxonomy import load_taxonomy
from query_pipeline.models.turn import Turn


class ValueResult(BaseModel):
    is_valuable: bool
    reason: str | None = None


class ComplexityResult(BaseModel):
    is_complex: bool
    reason: str | None = None


class ClassifyResult(BaseModel):
    category_id: str
    reason: str | None = None

    OTHER: ClassVar[str] = "other"

    @field_validator("category_id")
    @classmethod
    def validate_category_id(cls, value: str) -> str:
        if value != ClassifyResult.OTHER and value not in load_taxonomy().complex and value not in load_taxonomy().normal:
            raise ValueError(f"invalid category_id: {value}")
        return value


def _as_dict(raw: str | dict[str, Any]) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else parse_json_object(raw)


def parse_value_response(raw: str | dict[str, Any]) -> ValueResult:
    # 严格类型校验：LLM 输出 "is_valuable": "false"（字符串）会被 pydantic 正确解析为
    # False，而手工 bool("false") == True 会静默放行；非法值抛 ValidationError
    # （ValueError 子类）→ 候选 fail-closed 丢弃。
    return ValueResult.model_validate(_as_dict(raw))


def parse_complexity_response(raw: str | dict[str, Any]) -> ComplexityResult:
    return ComplexityResult.model_validate(_as_dict(raw))


def parse_classify_response(raw: str | dict[str, Any]) -> ClassifyResult:
    data = _as_dict(raw)
    return ClassifyResult(category_id=str(data.get("category_id") or ""), reason=data.get("reason"))


async def _call(
    client: LLMClient,
    llm_cfg: LLMConfig,
    cache: dict[str, dict[str, Any]],
    cache_path: Any,
    cache_lock: Any,
    *,
    step: str,
    prompt_id: str,
    payload: dict[str, Any],
    parse: Any,
) -> Any:
    system_prompt = resolve_prompt(prompt_id)
    user_prompt = "请标注以下会话问句，只输出严格 JSON：\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    cache_key = make_cache_key(user_prompt, step=step, model=llm_cfg.model, prompt=system_prompt)
    if cache_key in cache:
        try:
            return parse(cache[cache_key])
        except (ValueError, RuntimeError) as exc:
            # 坏缓存 label（跨版本 schema / 手改缓存）：驱逐并重调，避免每次运行重复丢弃
            # （与 segment 的自愈策略一致）。
            logger.warning("cached %s label invalid, re-calling LLM: %s", step, str(exc)[:120])
            cache.pop(cache_key, None)
    raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    parsed = parse(raw)
    await put_cache(
        cache,
        cache_path,
        cache_key,
        parsed.model_dump(),
        meta={"step": step, "prompt_id": prompt_id, "model": llm_cfg.model, "question": payload["current_question"][:120]},
        lock=cache_lock,
    )
    return parsed


async def funnel_candidate(
    *,
    client: LLMClient,
    turns: list[Turn],
    segments: list[Any],
    idx: int,
    llm_cfg: LLMConfig,
    cache: dict[str, dict[str, Any]],
    cache_path: Any,
    cache_lock: Any,
) -> dict[str, Any]:
    """Run the value -> complexity -> classify funnel for one candidate turn.

    Returns a dict with keys: idx, value, complexity, classify, difficulty,
    category_id, reason, error. On any gate failure the candidate is dropped
    (error set); classify runs only for survivors.
    """
    from query_pipeline.session.judge import segment_of

    segment = segment_of(segments, idx)
    payload = build_judge_payload(turns, segment, idx)
    try:
        value = await _call(
            client, llm_cfg, cache, cache_path, cache_lock,
            step="value_gate", prompt_id="value_gate", payload=payload, parse=parse_value_response,
        )
        if not value.is_valuable:
            return {"idx": idx, "value": value, "dropped": "value", "error": None}
        complexity = await _call(
            client, llm_cfg, cache, cache_path, cache_lock,
            step="complexity_gate", prompt_id="complexity_gate", payload=payload, parse=parse_complexity_response,
        )
        if complexity.is_complex:
            classify = await _call(
                client, llm_cfg, cache, cache_path, cache_lock,
                step="classify_complex", prompt_id="classify_complex", payload=payload, parse=parse_classify_response,
            )
            if classify.category_id == ClassifyResult.OTHER:
                raise ValueError("classify_complex returned 'other' — complex taxonomy has no fallback")
            difficulty = "hard"
        else:
            classify = await _call(
                client, llm_cfg, cache, cache_path, cache_lock,
                step="classify_normal", prompt_id="classify_normal", payload=payload, parse=parse_classify_response,
            )
            difficulty = "normal"
        return {
            "idx": idx,
            "value": value,
            "complexity": complexity,
            "classify": classify,
            "difficulty": difficulty,
            "category_id": classify.category_id,
            "reason": classify.reason or complexity.reason,
            "error": None,
        }
    except (ValueError, RuntimeError, APIStatusError) as exc:
        # APIStatusError（含 4xx/5xx 永久错误）也按候选失败处理（fail-closed 丢弃），
        # 不重试（client 已处理）也不炸掉整个会话。
        return {"idx": idx, "error": str(exc)[:200]}
