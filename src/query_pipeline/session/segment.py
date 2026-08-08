from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from query_pipeline.config.models import LLMConfig
from query_pipeline.llm.cache import make_cache_key, put_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.models.session import Segment, parse_segment_response
from query_pipeline.models.turn import Turn
from query_pipeline.prompts import resolve_prompt

logger = logging.getLogger(__name__)


def build_segment_payload(turns: list[Turn]) -> dict:
    return {"questions": [{"idx": i, "question": turns[i].question} for i in range(len(turns))]}


def _build_user_prompt(turns: list[Turn]) -> str:
    return "请切分以下会话问句，只输出严格 JSON：\n" + json.dumps(
        build_segment_payload(turns), ensure_ascii=False, separators=(",", ":")
    )


def _segments_from_cache(label: dict, num_turns: int) -> list[Segment]:
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
    # Fresh labels are repaired to full [0, num_turns-1] coverage; a locally-contiguous but
    # partial cache (e.g. [[0,2]] for 5 turns) would leave indices uncovered and later raise
    # IndexError in segment_of. Reject it so the caller re-calls the LLM instead.
    if segments[0].start != 0 or segments[-1].end != num_turns - 1:
        raise ValueError("cached segments do not cover the whole session")
    return segments


async def segment_session(
    *,
    client: LLMClient,
    turns: list[Turn],
    llm_cfg: LLMConfig,
    cache: dict[str, dict],
    cache_path: Path,
    cache_lock: asyncio.Lock | None = None,
) -> list[Segment]:
    """Split turns into topic segments; fall back to whole_session on failure."""
    num_turns = len(turns)
    if num_turns <= 1:
        return [Segment(0, num_turns - 1, "whole_session")]

    system_prompt = resolve_prompt("segment")
    user_prompt = _build_user_prompt(turns)
    cache_key = make_cache_key(user_prompt, step="segment", model=llm_cfg.model, prompt=system_prompt)
    lock = cache_lock or asyncio.Lock()

    try:
        if cache_key in cache:
            try:
                return _segments_from_cache(cache[cache_key], num_turns)
            except ValueError as exc:
                logger.warning(
                    "cached segmentation invalid for %d-turn session, re-calling LLM: %s",
                    num_turns,
                    str(exc)[:200],
                )
                cache.pop(cache_key, None)

        raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        segments: list[Segment] = []
        attempts = 3
        for attempt in range(attempts):
            try:
                segments = parse_segment_response(raw, num_turns=num_turns)
                break
            except ValueError:
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
        await put_cache(
            cache, cache_path, cache_key, label, meta={"step": "segment", "model": llm_cfg.model}, lock=lock
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "segmentation failed for %d-turn session, falling back to whole_session: %s",
            num_turns,
            str(exc)[:200],
        )
        return [Segment(0, num_turns - 1, "whole_session")]

    return segments
