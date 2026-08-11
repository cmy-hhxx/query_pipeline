from __future__ import annotations

import re
from typing import Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, field_validator


ComplexRoute = Literal["complex", "normal", "reject"]

ComplexFeature = Literal[
    "natural_multi_condition_screen",
    "position_context_decision",
    "multi_dimension_attribution",
    "cross_period_entity_research",
    "strategy_scenario_evaluation",
    "event_policy_impact",
    "multi_method_technical_analysis",
    "macro_industry_transmission",
    "historical_simulation_statistics",
    "stateful_tracking_execution",
    "artifact_action",
]

ExclusionReason = Literal[
    "simple_lookup",
    "simple_filter_ranking",
    "single_formula",
    "generic_recommendation",
    "absolute_unverifiable",
    "insufficient_depth",
    "eval_template",
    "embedded_prompt",
]

PolicyCriterion: TypeAlias = ComplexFeature | ExclusionReason


class PolicyEvidence(BaseModel):
    """Direct provenance for a complex feature or exclusion reason."""

    model_config = ConfigDict(extra="forbid", strict=True)

    criterion: PolicyCriterion
    quote: str

    @field_validator("quote")
    @classmethod
    def validate_quote(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("policy evidence quote must be non-empty")
        return text


def _grounding_norm(text: str) -> str:
    """逐字证据宽松匹配用：去掉所有空白 + 全角标点转半角。"""
    for full, half in (("，", ","), ("。", "."), ("、", ","), ("：", ":"), ("；", ";")):
        text = text.replace(full, half)
    return re.sub(r"\s+", "", text)


def evidence_is_grounded(evidence: list[PolicyEvidence], question: str) -> bool:
    """Return whether every evidence quote is copied from the source question.

    空白/全角差异不算失配（LLM 常对中文标点做全半角转换）。
    """
    q = _grounding_norm(question)
    return bool(evidence) and all(_grounding_norm(item.quote) in q for item in evidence)


_VALID_FEATURES = set(get_args(ComplexFeature))
_VALID_EXCLUSIONS = set(get_args(ExclusionReason))
_VALID_CRITERIA = _VALID_FEATURES | _VALID_EXCLUSIONS


def clean_policy_fields(data: dict[str, Any]) -> dict[str, Any]:
    """模型偶发输出非法枚举：过滤掉不在受控枚举里的值，避免一次脏值就 fail-closed 丢候选。

    过滤后若路由因此不自洽（如 complex 无任何合法 feature），交给
    validate_route_consistency 正常拒绝；多余证据也会被一并丢弃。
    """
    cleaned = dict(data)
    cleaned["complex_features"] = [
        f for f in data.get("complex_features", []) if f in _VALID_FEATURES
    ]
    cleaned["exclusion_reasons"] = [
        e for e in data.get("exclusion_reasons", []) if e in _VALID_EXCLUSIONS
    ]
    evidence = data.get("evidence")
    if isinstance(evidence, list):
        cleaned["evidence"] = [
            item for item in evidence
            if isinstance(item, dict) and item.get("criterion") in _VALID_CRITERIA
        ]
    return cleaned
