from __future__ import annotations

import logging
from typing import Any

from query_pipeline.models.turn import Session, Turn

logger = logging.getLogger(__name__)


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

    bad_turns = [t for t in prior if not isinstance(t, dict)]
    if bad_turns:
        # fail-loud：静默过滤非 dict turn 会让整行在 chat 门槛（3/1/2）下无声落选，
        # 且无任何计数/日志。抛错 → preclean 按 adapt_failed 进 bad_lines（可审计）。
        raise ValueError(f"judge_data.context 含 {len(bad_turns)} 个非对象 turn")

    turns: list[Turn] = [
        Turn(question=_as_str(t.get("question")), answer=_as_str(t.get("answer")))
        for t in prior
    ]

    meta = jd.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    raw_input = jd.get("input")
    chain = jd.get("chain")
    if chain is not None and not isinstance(chain, list):
        # 畸形 chain 静默置 [] 会让 chain_tool_calls 回退 tool_count（chat 未设置 → 0），
        # 整行在 3/1/2 门槛下无声落选。记 warning 暴露问题，但保留行（chain 缺省合法）。
        logger.warning("judge_data.chain 非列表（置空）：%r", str(chain)[:80])
        chain = []
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
            answer_full=_as_str(jd.get("raw_answer") or jd.get("text_answer")),
            trace_id=_as_str(jd.get("trace_id") or record.get("trace_id")),
            chain=chain if isinstance(chain, list) else [],
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
