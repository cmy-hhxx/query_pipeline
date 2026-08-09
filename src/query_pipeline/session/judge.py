from __future__ import annotations

from query_pipeline.models.session import Segment, prior_indices
from query_pipeline.models.turn import Turn


def segment_of(segments: list[Segment], idx: int) -> Segment:
    for seg in segments:
        if seg.start <= idx <= seg.end:
            return seg
    raise IndexError(f"index {idx} not covered by any segment")


def build_judge_payload(turns: list[Turn], segment: Segment, idx: int) -> dict[str, Any]:
    prior = [turns[k].question for k in prior_indices(segment, idx)]
    return {"prior_questions": prior, "current_question": turns[idx].question}


from typing import Any  # noqa: E402
