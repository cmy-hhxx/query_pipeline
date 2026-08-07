from __future__ import annotations

from typing import Any


def normalize_judge_data_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a judge_data line into the canonical session shape.

    judge_data lines are single-case: one question (judge_data.input.text)
    plus its pre-assembled prior context (judge_data.context). The normalized
    session appends the current turn after the prior context turns so the
    per-turn pipeline can consume it unchanged; thread_id carries the case_id.
    """
    jd = record.get("judge_data")
    if not isinstance(jd, dict):
        raise ValueError("record missing judge_data object")

    prior = jd.get("context")
    if not isinstance(prior, list):
        raise ValueError("judge_data.context must be a list")

    turns: list[dict[str, Any]] = [
        {"question": t.get("question", ""), "answer": t.get("answer", "")}
        for t in prior
        if isinstance(t, dict)
    ]

    meta = jd.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    raw_input = jd.get("input")
    if isinstance(raw_input, dict):
        question = raw_input.get("text") or record.get("question", "")
    elif isinstance(raw_input, str):
        question = raw_input or record.get("question", "")
    else:
        question = record.get("question", "")
    if not isinstance(question, str):
        question = str(question) if question is not None else ""

    current: dict[str, Any] = {
        "question": question,
        "answer": jd.get("text_answer") or jd.get("raw_answer") or "",
        "trace_id": jd.get("trace_id") or record.get("trace_id", ""),
        "chain": jd.get("chain") if isinstance(jd.get("chain"), list) else [],
        "request_time": meta.get("request_time", ""),
        "first_token_ms": meta.get("first_token_time_cost"),
        "total_duration_ms": meta.get("finish_answer_time_cost"),
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "user_id": meta.get("user_id", ""),
        "status": "completed",
        "outcome": "success",
    }
    turns.append(current)

    return {
        "thread_id": jd.get("case_id") or record.get("trace_id", ""),
        "context": turns,
    }
