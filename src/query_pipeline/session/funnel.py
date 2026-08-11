"""Decoupled funnel per candidate: value gate -> complexity gate -> classify.

Each step is a separate LLM call with its own cache key. Failures drop the
candidate (fail-closed); the funnel only assembles rows for survivors.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

# 模型输出是概率性的：解析/校验失败（非 API 错误）重调，避免一次坏输出就 fail-closed 丢候选。
# 语义上等于"请按契约重新输出"，与 segment 的解析自愈一致。
PARSE_MAX_ATTEMPTS = 5

from openai import APIStatusError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from query_pipeline.config.models import LLMConfig
from query_pipeline.llm.cache import make_cache_key, put_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.models.session import parse_json_object
from query_pipeline.models.complexity import (
    ComplexFeature,
    ComplexRoute,
    ExclusionReason,
    PolicyEvidence,
    clean_policy_fields,
    evidence_is_grounded,
)
from query_pipeline.prompts import resolve_prompt
from query_pipeline.session.judge import build_judge_payload
from query_pipeline.taxonomy import load_taxonomy
from query_pipeline.models.turn import Turn


class ValueResult(BaseModel):
    is_valuable: bool
    has_executable_task: bool = True
    self_contained: bool = True
    template_severity: Literal["none", "light", "severe"] = "none"
    contains_embedded_prompt: bool = False
    reason: str | None = None

    @property
    def admissible(self) -> bool:
        """Semantic value admission; no keyword or length heuristics."""
        return (
            self.is_valuable
            and self.has_executable_task
            and self.self_contained
            and self.template_severity != "severe"
            and not self.contains_embedded_prompt
        )

    @property
    def rejection_kind(self) -> str:
        if self.contains_embedded_prompt or self.template_severity == "severe":
            return "template"
        if not self.has_executable_task:
            return "no_task"
        if not self.self_contained:
            return "not_self_contained"
        return "other"


class SemanticSignature(BaseModel):
    """Entity/value-independent description of the task's reasoning program."""

    goal: str
    subject_type: str
    operations: list[str] = Field(default_factory=list)
    data_dimensions: list[str] = Field(default_factory=list)
    temporal_shape: str
    output_shape: list[str] = Field(default_factory=list)

    @field_validator("goal", "subject_type", "temporal_shape")
    @classmethod
    def validate_scalar(cls, value: str) -> str:
        text = value.strip().lower()
        if not text:
            raise ValueError("semantic signature scalar fields must be non-empty")
        return text

    @field_validator("operations", "data_dimensions", "output_shape")
    @classmethod
    def normalize_list(cls, value: list[str]) -> list[str]:
        # 排序+去重让同一语义签名稳定序列化；实体/数字由 prompt 负责不写入。
        return sorted(set(value))


class ComplexityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    route: ComplexRoute
    complex_features: list[ComplexFeature] = Field(default_factory=list)
    exclusion_reasons: list[ExclusionReason] = Field(default_factory=list)
    evidence: list[PolicyEvidence] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    question_quality: Literal["low", "medium", "high"]
    semantic_signature: SemanticSignature
    reason: str | None = None

    @field_validator("complex_features", "exclusion_reasons")
    @classmethod
    def normalize_policy_lists(cls, value: list[Any]) -> list[Any]:
        return sorted(set(value))

    @model_validator(mode="after")
    def validate_route_consistency(self) -> "ComplexityResult":
        criteria = {item.criterion for item in self.evidence}
        if self.route == "complex":
            if not self.complex_features or self.exclusion_reasons:
                raise ValueError("complex route requires features and forbids exclusions")
            if self.confidence == "low" or self.question_quality == "low":
                raise ValueError("low confidence/quality cannot route to complex")
            # 覆盖匹配：每条声明的 feature 都要有对应证据，容忍多余的 evidence
            # （精确集合相等对模型过于苛刻，任何多/漏都会 fail-closed 丢候选）。
            if set(self.complex_features) - criteria:
                raise ValueError("every complex feature requires matching evidence")
        elif self.route == "normal":
            if self.complex_features or not self.exclusion_reasons:
                raise ValueError("normal route requires exclusions and forbids complex features")
            if set(self.exclusion_reasons) & {"eval_template", "embedded_prompt"}:
                raise ValueError("template/prompt exclusions must route to reject")
            # 负向声明（排除原因）不强制证据：路由本身已是权威判定。
        else:
            if self.complex_features:
                raise ValueError("reject route forbids complex features")
            if not set(self.exclusion_reasons) & {"eval_template", "embedded_prompt"}:
                raise ValueError("reject route requires eval_template or embedded_prompt")
        return self

    @property
    def admissible_hard(self) -> bool:
        """Business-policy hard admission before source quote containment."""
        return self.route == "complex"

    def admits_hard_for(self, question: str) -> bool:
        return self.admissible_hard and self.evidence_is_grounded_for(question)

    def evidence_is_grounded_for(self, question: str) -> bool:
        # 只有 complex 的正向声明要求逐字证据；normal/reject 的排除原因为负向声明，不强制。
        if self.route != "complex":
            return True
        return evidence_is_grounded(self.evidence, question)


class ClassifyResult(BaseModel):
    category_id: str
    reason: str

    @field_validator("category_id")
    @classmethod
    def validate_category_id(cls, value: str) -> str:
        if value not in load_taxonomy().complex and value not in load_taxonomy().normal:
            raise ValueError(f"invalid category_id: {value}")
        return value

    @field_validator("reason")
    @classmethod
    def validate_classify_reason(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("classify reason must be non-empty")
        return text


def _as_dict(raw: str | dict[str, Any]) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else parse_json_object(raw)


def parse_value_response(raw: str | dict[str, Any]) -> ValueResult:
    # 严格类型校验：LLM 输出 "is_valuable": "false"（字符串）会被 pydantic 正确解析为
    # False，而手工 bool("false") == True 会静默放行；非法值抛 ValidationError
    # （ValueError 子类）→ 候选 fail-closed 丢弃。
    return ValueResult.model_validate(_as_dict(raw))


def parse_complexity_response(raw: str | dict[str, Any]) -> ComplexityResult:
    return ComplexityResult.model_validate(clean_policy_fields(_as_dict(raw)))


def parse_classify_response(raw: str | dict[str, Any]) -> ClassifyResult:
    data = _as_dict(raw)
    return ClassifyResult(
        category_id=str(data.get("category_id") or ""),
        reason=str(data.get("reason") or ""),
    )


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
    parsed: Any | None = None
    for attempt in range(1, PARSE_MAX_ATTEMPTS + 1):
        raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        try:
            parsed = parse(raw)
            break
        except (ValueError, RuntimeError) as exc:
            if attempt == PARSE_MAX_ATTEMPTS:
                raise
            logger.warning(
                "%s label invalid (attempt %d/%d), re-calling LLM: %s",
                step, attempt, PARSE_MAX_ATTEMPTS, str(exc)[:120],
            )
    # 循环内要么成功赋值要么 raise，不可能走到这里仍为 None
    assert parsed is not None
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
    current_only_payload = {"current_question": payload["current_question"]}

    def parse_grounded_complexity(raw: str | dict[str, Any]) -> ComplexityResult:
        parsed = parse_complexity_response(raw)
        if not parsed.evidence_is_grounded_for(current_only_payload["current_question"]):
            raise ValueError("complexity evidence quote must be copied from current_question")
        return parsed

    try:
        value = await _call(
            client, llm_cfg, cache, cache_path, cache_lock,
            step="value_gate", prompt_id="value_gate", payload=current_only_payload,
            parse=parse_value_response,
        )
        if not value.admissible:
            return {"idx": idx, "value": value, "dropped": "value", "error": None}
        complexity = await _call(
            client, llm_cfg, cache, cache_path, cache_lock,
            step="complexity_gate", prompt_id="complexity_gate", payload=current_only_payload,
            parse=parse_grounded_complexity,
        )
        if complexity.route == "reject":
            return {
                "idx": idx,
                "value": value,
                "complexity": complexity,
                "dropped": "complexity_reject",
                "error": None,
            }
        if complexity.admits_hard_for(current_only_payload["current_question"]):
            classify = await _call(
                client, llm_cfg, cache, cache_path, cache_lock,
                step="classify_complex", prompt_id="classify_complex",
                payload=current_only_payload, parse=parse_classify_response,
            )
            if classify.category_id not in load_taxonomy().complex:
                raise ValueError(
                    f"classify_complex returned invalid complex category {classify.category_id!r}"
                )
            difficulty = "hard"
        else:
            classify = await _call(
                client, llm_cfg, cache, cache_path, cache_lock,
                step="classify_normal", prompt_id="classify_normal", payload=payload, parse=parse_classify_response,
            )
            if classify.category_id not in load_taxonomy().normal:
                raise ValueError(
                    f"classify_normal returned invalid normal category {classify.category_id!r}"
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
