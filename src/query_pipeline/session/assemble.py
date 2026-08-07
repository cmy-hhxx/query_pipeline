from __future__ import annotations

from typing import Any

from query_pipeline.models.records import ENGLISH_CATEGORIES
from query_pipeline.models.session import Segment, prior_indices
from query_pipeline.session.candidates import extract_tool_names


def assemble_row(
    session: dict[str, Any],
    turns: list[dict[str, Any]],
    segment: Segment,
    idx: int,
    category_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build one complex-query output row (filter_out.jsonc schema) for a turn.

    context[] holds prior same-segment turns trimmed to {question, answer}; a
    segment-leading turn falls back to every earlier session turn, so only the
    session's very first turn yields an empty context. trace_id carries the
    original input turn's trace_id, and meta carries the judge's reason plus
    the original question timestamp (request_time).
    """
    turn = turns[idx]
    prior = [
        {"question": turns[k].get("question", ""), "answer": turns[k].get("answer", "")}
        for k in prior_indices(segment, idx)
    ]
    category = f"{category_id}-{ENGLISH_CATEGORIES[category_id]}"
    return {
        "capture_mode": "full_link",
        "user_cohort": "regular",
        "source_case_id": session.get("thread_id", ""),
        "answer_key": "",
        "trace_id": turn.get("trace_id", ""),
        "category": category,
        "input": {"text": turn.get("question", ""), "image": "", "file": ""},
        "session_round": idx - segment.start + 1,
        "context": prior,
        "chain": turn.get("chain", []),
        "tools": extract_tool_names(turn),
        "raw_answer": turn.get("answer", ""),
        "text_answer": turn.get("answer", ""),
        "multimodal": [],
        "model_version": "",
        "release_id": "",
        "agent_mode": "",
        "translation": "",
        "user_id": turn.get("user_id", ""),
        "difficulty_level": "hard",
        "first_token_time_ms": turn.get("first_token_ms"),
        "finish_answer_time_ms": turn.get("total_duration_ms"),
        "input_tokens": turn.get("input_tokens"),
        "output_tokens": turn.get("output_tokens"),
        "request_time_ms": None,
        "meta": {"reason": reason, "request_time": turn.get("request_time", "")},
    }
