from __future__ import annotations

from typing import Any

from query_pipeline.models.turn import Session, Turn


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _tool_names_fallback(raw: dict[str, Any]) -> str:
    for key in ("tool_names", "tool_names_text"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _request_time(raw: dict[str, Any]) -> str:
    for key in ("request_time", "question_at"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return _as_str(value)
    return ""


def adapt_turn(raw: dict[str, Any]) -> Turn:
    chain = raw.get("chain")
    return Turn(
        question=_as_str(raw.get("question")),
        answer=_as_str(raw.get("answer") or raw.get("answer_full")),
        trace_id=_as_str(raw.get("trace_id")),
        run_id=_as_str(raw.get("run_id")),
        chain=chain if isinstance(chain, list) else [],
        request_time=_request_time(raw),
        first_token_ms=raw.get("first_token_ms"),
        total_duration_ms=raw.get("total_duration_ms"),
        input_tokens=raw.get("input_tokens"),
        output_tokens=raw.get("output_tokens"),
        user_id=_as_str(raw.get("user_id")),
        status=raw.get("status", "completed"),
        outcome=raw.get("outcome", "success"),
        tool_names=_tool_names_fallback(raw),
        tool_count=raw.get("tool_count"),
    )


def adapt_session(record: dict[str, Any]) -> Session:
    context = record.get("context")
    if not isinstance(context, list):
        raise ValueError("session.context must be a list")
    turns = [adapt_turn(t) for t in context if isinstance(t, dict)]
    return Session(
        thread_id=_as_str(record.get("thread_id")),
        turns=turns,
        candidate_mode="all",
    )
