"""Pluggable stage registry.

A stage is an async callable ``(ctx, client, cache, cache_lock) -> ctx``.
Adding a module to the pipeline = implement that contract, register it here,
and (optionally) list it in config ``stages``. The default stage list is the
built-in funnel order; unknown stage names fail loudly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.llm.client import LLMClient

Stage = Callable[
    [PipelineContext, LLMClient | None, dict[str, dict[str, Any]], asyncio.Lock],
    Awaitable[PipelineContext],
]

DEFAULT_STAGES: tuple[str, ...] = (
    "precheck",
    "preclean",
    "segment",
    "rule_gate",
    "judge",
    "verify",
    "answer_gate",
    "post",
)

REGISTRY: dict[str, Stage] = {}


def register(name: str) -> Callable[[Stage], Stage]:
    def decorator(stage: Stage) -> Stage:
        if name in REGISTRY:
            raise ValueError(f"duplicate stage registration: {name!r}")
        REGISTRY[name] = stage
        return stage

    return decorator


def get_stage(name: str) -> Stage:
    try:
        return REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"unknown stage {name!r}; registered stages: {available}") from None


def stage_names(config_stages: list[str] | None) -> list[str]:
    names = list(config_stages) if config_stages else list(DEFAULT_STAGES)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown stages in config: {unknown}")
    return names
