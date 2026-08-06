from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from query_pipeline.config.models import LLMStageConfig
from query_pipeline.llm.cache import append_cache, make_cache_key
from query_pipeline.llm.client import LLMClient
from query_pipeline.models.session import Segment, parse_segment_response
from query_pipeline.prompts import resolve_prompt

logger = logging.getLogger(__name__)


def build_segment_payload(turns: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "questions": [{"idx": i, "question": turns[i].get("question", "")} for i in range(len(turns))]
    }


def _build_user_prompt(turns: list[dict[str, Any]]) -> str:
    return "请切分以下会话问句，只输出严格 JSON：\n" + json.dumps(
        build_segment_payload(turns), ensure_ascii=False, separators=(",", ":")
    )


def _segments_from_cache(label: dict[str, Any], num_turns: int) -> list[Segment]:
    items = label.get("segments") or []
    if not isinstance(items, list):
        raise ValueError("cached segments must be a list")
    segments: list[Segment] = []
    for i, item in enumerate(items):
        start = item.get("start")
        end = item.get("end")
        topic = item.get("topic")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(topic, str):
            raise ValueError(f"cached segment {i} malformed")
        if not (0 <= start <= end < num_turns):
            raise ValueError(f"cached segment {i} out of range")
        if segments and start != segments[-1].end + 1:
            raise ValueError(f"cached segments not contiguous at index {i}")
        segments.append(Segment(start=start, end=end, topic=topic))
    if not segments:
        raise ValueError("cached segments must not be empty")
    return segments


async def segment_session(
    *,
    client: LLMClient,
    turns: list[dict[str, Any]],
    llm_cfg: LLMStageConfig,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
) -> list[Segment]:
    """Split a session's turns into topic-contiguous segments via one LLM call.

    Falls back to a single whole-session segment on any LLM/parse/cache error.
    """
    num_turns = len(turns)
    if num_turns <= 1:
        return [Segment(0, num_turns - 1, "whole_session")]

    system_prompt = resolve_prompt("segment")
    user_prompt = _build_user_prompt(turns)
    cache_key = make_cache_key(user_prompt, step="segment", model=llm_cfg.model, prompt=system_prompt)

    try:
        if cache_key in cache:
            segments = _segments_from_cache(cache[cache_key], num_turns)
        else:
            raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            segments: list[Segment] = []
            attempts = 3
            for attempt in range(attempts):
                try:
                    segments = parse_segment_response(raw, num_turns=num_turns)
                    break
                except ValueError:
                    # LLM output is not strictly deterministic; a malformed response
                    # is worth clean retries before giving up on the whole session.
                    if attempt == attempts - 1:
                        raise
                    logger.warning(
                        "segmentation parse failed for %d-turn session, retrying (%d/%d)",
                        num_turns,
                        attempt + 1,
                        attempts - 1,
                    )
                    raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
            label = {"segments": [{"start": s.start, "end": s.end, "topic": s.topic} for s in segments]}
            cache[cache_key] = label
            append_cache(cache_path, cache_key, label, meta={"step": "segment", "model": llm_cfg.model})
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "segmentation failed for %d-turn session, falling back to whole_session: %s",
            num_turns,
            str(exc)[:200],
        )
        return [Segment(0, num_turns - 1, "whole_session")]

    return segments
