from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

from query_pipeline.models.records import CATEGORIES


@dataclass(frozen=True)
class Segment:
    """Contiguous span of turns in a session, as 0-based inclusive indices."""

    start: int
    end: int
    topic: str


def prior_indices(segment: Segment, idx: int) -> range:
    """Indices of the turns that form turn ``idx``'s context.

    Same-segment prior turns; if the turn starts a segment (no same-segment
    prior), fall back to every earlier session turn. Only the session's very
    first turn yields an empty range.
    """
    start = segment.start if idx > segment.start else 0
    return range(start, idx)


class Step2Result(BaseModel):
    is_complex: bool
    category_id: str | None = None
    reason: str | None = None

    @field_validator("category_id")
    @classmethod
    def validate_category_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in CATEGORIES:
            raise ValueError(f"invalid category_id: {value}")
        return value

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("reason must be non-empty when present")
        return text

    @model_validator(mode="after")
    def validate_complex_contract(self) -> Step2Result:
        if not self.is_complex:
            self.category_id = None
            return self
        if self.category_id is None:
            raise ValueError("category_id is required when is_complex is true")
        return self

    def to_cache_label(self) -> dict[str, Any]:
        return {
            "is_complex": self.is_complex,
            "category_id": self.category_id,
            "reason": self.reason,
        }


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating markdown fences."""
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            raise ValueError(f"cannot parse JSON response: {text[:200]}")
        data = json.loads(match.group(1))
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    return data


def parse_step2_response(raw: str) -> Step2Result:
    data = parse_json_object(raw)
    return Step2Result.model_validate(data)


def parse_step2_payload(data: dict[str, Any]) -> Step2Result:
    return Step2Result.model_validate(data)


class VerifyResult(BaseModel):
    """Standalone second-pass verdict: is the question complex on its own.

    Deliberately category-free — pass 1 already assigned the category; this
    stage only re-confirms complexity without session context.
    """

    is_complex: bool
    reason: str | None = None

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("reason must be non-empty when present")
        return text

    def to_cache_label(self) -> dict[str, Any]:
        return {"is_complex": self.is_complex, "reason": self.reason}


def parse_verify_response(raw: str) -> VerifyResult:
    return VerifyResult.model_validate(parse_json_object(raw))


def parse_verify_payload(data: dict[str, Any]) -> VerifyResult:
    return VerifyResult.model_validate(data)


def parse_segment_response(raw: str, *, num_turns: int) -> list[Segment]:
    """Parse the segmentation LLM output and merge recurring topics.

    The LLM returns contiguous, non-overlapping segments covering all turns.
    A topic that recurs non-adjacently (A, B, A) is merged into a single
    segment spanning from its first occurrence to its last (the rule chosen in
    the design interview): the whole span becomes one topic A.

    Small LLM boundary slips (off-by-one gaps/overlaps and coverage misses)
    are repaired into a valid covering instead of rejected; grossly malformed
    output still raises so the caller can fall back to whole-session.
    """
    data = parse_json_object(raw)
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("segments must be a list")
    if not raw_segments:
        raise ValueError("segments must not be empty")

    segments: list[Segment] = []
    for i, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise ValueError(f"segment {i} must be an object")
        start = item.get("start")
        end = item.get("end")
        topic = item.get("topic")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(topic, str):
            raise ValueError(f"segment {i} must have int start/end and str topic")
        if not (0 <= start <= end < num_turns):
            raise ValueError(f"segment {i} out of range: start={start} end={end} n={num_turns}")
        topic = topic.strip()
        if not topic:
            raise ValueError(f"segment {i} has empty topic")
        segments.append(Segment(start=start, end=end, topic=topic))

    segments = _repair_contiguous(segments, num_turns)
    return _merge_recurring_topics(segments)


def _repair_contiguous(segments: list[Segment], num_turns: int) -> list[Segment]:
    """Snap small LLM boundary slips into a valid contiguous [0, n-1] covering.

    Total index deviation (left gap + right gap + per-boundary gaps) must be
    <= 2; anything grosser raises so the caller can fall back to whole-session.
    """
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    deviation = ordered[0].start + (num_turns - 1 - ordered[-1].end)
    for a, b in zip(ordered, ordered[1:]):
        deviation += abs(b.start - (a.end + 1))
    if deviation > 2:
        raise ValueError("segments too far from a valid contiguous covering")

    repaired: list[Segment] = []
    for i, seg in enumerate(ordered):
        start = 0 if i == 0 else repaired[-1].end + 1
        end = num_turns - 1 if i == len(ordered) - 1 else seg.end
        repaired.append(Segment(start=start, end=max(end, start), topic=seg.topic))
    return repaired


def _merge_recurring_topics(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    seen: dict[str, int] = {}
    for seg in segments:
        if seg.topic in seen:
            idx = seen[seg.topic]
            merged[idx] = Segment(start=merged[idx].start, end=seg.end, topic=seg.topic)
            merged = merged[: idx + 1]
            seen = {m.topic: i for i, m in enumerate(merged)}
        else:
            seen[seg.topic] = len(merged)
            merged.append(seg)
    return merged
