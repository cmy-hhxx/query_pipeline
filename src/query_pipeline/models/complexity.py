from __future__ import annotations

from typing import Literal, TypeAlias

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


def evidence_is_grounded(evidence: list[PolicyEvidence], question: str) -> bool:
    """Return whether every evidence quote is copied from the source question."""
    return bool(evidence) and all(item.quote in question for item in evidence)
