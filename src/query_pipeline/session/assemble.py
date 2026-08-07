from __future__ import annotations

from query_pipeline.models.output import ContextTurn, OutputInput, OutputMeta, OutputRow
from query_pipeline.models.records import ENGLISH_CATEGORIES
from query_pipeline.models.session import Segment, prior_indices
from query_pipeline.models.turn import Session
from query_pipeline.session.candidates import extract_tool_names


def assemble_row(
    session: Session,
    segment: Segment,
    idx: int,
    category_id: str,
    reason: str | None = None,
) -> dict:
    """Build one filter_out.jsonc row for a turn; returns a plain dict for stages."""
    turns = session.turns
    turn = turns[idx]
    prior = [
        ContextTurn(question=turns[k].question, answer=turns[k].answer)
        for k in prior_indices(segment, idx)
    ]
    row = OutputRow(
        source_case_id=session.thread_id,
        trace_id=turn.trace_id,
        category=f"{category_id}-{ENGLISH_CATEGORIES[category_id]}",
        input=OutputInput(text=turn.question),
        session_round=idx - segment.start + 1,
        context=prior,
        chain=turn.chain,
        tools=extract_tool_names(turn),
        raw_answer=turn.answer,
        text_answer=turn.answer,
        user_id=turn.user_id,
        first_token_time_ms=turn.first_token_ms,
        finish_answer_time_ms=turn.total_duration_ms,
        input_tokens=turn.input_tokens,
        output_tokens=turn.output_tokens,
        meta=OutputMeta(reason=reason, request_time=turn.request_time),
    )
    return row.model_dump(mode="python")
