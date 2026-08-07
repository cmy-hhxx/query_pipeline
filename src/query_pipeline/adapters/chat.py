from __future__ import annotations

from typing import Any

from query_pipeline.models.turn import Session, Turn


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def adapt_chat(record: dict[str, Any]) -> Session:
    """Parse iwencai-style judge_data wrapper into a Session (last turn = case)."""
    jd = record.get("judge_data")
    if not isinstance(jd, dict):
        raise ValueError("record missing judge_data object")

    prior = jd.get("context")
    if not isinstance(prior, list):
        raise ValueError("judge_data.context must be a list")

    turns: list[Turn] = [
        Turn(question=_as_str(t.get("question")), answer=_as_str(t.get("answer")))
        for t in prior
        if isinstance(t, dict)
    ]

    meta = jd.get("meta") if isinstance(jd.get("meta"), dict) else {}
    raw_input = jd.get("input")
    if isinstance(raw_input, dict):
        question = _as_str(raw_input.get("text") or record.get("question"))
    elif isinstance(raw_input, str):
        question = raw_input or _as_str(record.get("question"))
    else:
        question = _as_str(record.get("question"))

    turns.append(
        Turn(
            question=question,
            answer=_as_str(jd.get("text_answer") or jd.get("raw_answer")),
            trace_id=_as_str(jd.get("trace_id") or record.get("trace_id")),
            chain=jd.get("chain") if isinstance(jd.get("chain"), list) else [],
            request_time=_as_str(meta.get("request_time")),
            first_token_ms=meta.get("first_token_time_cost"),
            total_duration_ms=meta.get("finish_answer_time_cost"),
            input_tokens=meta.get("input_tokens"),
            output_tokens=meta.get("output_tokens"),
            user_id=_as_str(meta.get("user_id")),
            status="completed",
            outcome="success",
        )
    )

    return Session(
        thread_id=_as_str(jd.get("case_id") or record.get("trace_id")),
        turns=turns,
        candidate_mode="last_only",
    )
