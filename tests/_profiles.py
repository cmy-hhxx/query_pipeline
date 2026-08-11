from __future__ import annotations

from typing import Any


def complexity_label(
    is_complex: bool,
    *,
    route: str | None = None,
    reason: str = "判定",
    goal: str = "analyze financial subject",
    complex_features: list[str] | None = None,
    exclusion_reasons: list[str] | None = None,
    evidence_quote: str | None = None,
    confidence: str = "high",
    question_quality: str = "high",
    subject_type: str = "stock",
    operations: list[str] | None = None,
    data_dimensions: list[str] | None = None,
    temporal_shape: str = "current",
    output_shape: list[str] | None = None,
) -> dict[str, Any]:
    selected_route = route or ("complex" if is_complex else "normal")
    if selected_route == "complex":
        features = complex_features or ["multi_dimension_attribution"]
        exclusions: list[str] = []
        criteria = features
    else:
        features = []
        exclusions = exclusion_reasons or [
            "eval_template" if selected_route == "reject" else "simple_lookup"
        ]
        criteria = exclusions
    quote = evidence_quote or goal
    return {
        "route": selected_route,
        "complex_features": features,
        "exclusion_reasons": exclusions,
        "evidence": [{"criterion": criterion, "quote": quote} for criterion in criteria],
        "confidence": confidence,
        "question_quality": question_quality,
        "semantic_signature": {
            "goal": goal,
            "subject_type": subject_type,
            "operations": operations or (["analyze"] if selected_route == "complex" else ["lookup"]),
            "data_dimensions": data_dimensions or ["financial_data"],
            "temporal_shape": temporal_shape,
            "output_shape": output_shape or ["explanation"],
        },
        "reason": reason,
    }


def verify_label(
    is_complex: bool,
    *,
    route: str | None = None,
    reason: str = "复核",
    complex_features: list[str] | None = None,
    exclusion_reasons: list[str] | None = None,
    evidence_quote: str = "测试证据",
    confidence: str = "high",
) -> dict[str, Any]:
    selected_route = route or ("complex" if is_complex else "normal")
    if selected_route == "complex":
        features = complex_features or ["multi_dimension_attribution"]
        exclusions: list[str] = []
        criteria = features
    else:
        features = []
        exclusions = exclusion_reasons or [
            "eval_template" if selected_route == "reject" else "simple_lookup"
        ]
        criteria = exclusions
    quote = evidence_quote
    return {
        "route": selected_route,
        "complex_features": features,
        "exclusion_reasons": exclusions,
        "evidence": [{"criterion": criterion, "quote": quote} for criterion in criteria],
        "confidence": confidence,
        "reason": reason,
    }
