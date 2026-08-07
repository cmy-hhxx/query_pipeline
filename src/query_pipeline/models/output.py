from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OutputInput(BaseModel):
    text: str = ""
    image: str = ""
    file: str = ""


class OutputMeta(BaseModel):
    reason: str | None = None
    request_time: str = ""
    translation: str = ""


class ContextTurn(BaseModel):
    question: str = ""
    answer: str = ""


class OutputRow(BaseModel):
    """Annotated complex-query row — fields match data/example/filter_out.jsonc."""

    capture_mode: str = "full_link"
    user_cohort: str = "regular"
    source_case_id: str = ""
    answer_key: str = ""
    trace_id: str = ""
    category: str = ""
    input: OutputInput = Field(default_factory=OutputInput)
    session_round: int = 1
    context: list[ContextTurn] = Field(default_factory=list)
    chain: list[Any] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    raw_answer: str = ""
    text_answer: str = ""
    multimodal: list[Any] = Field(default_factory=list)
    model_version: str = ""
    release_id: str = ""
    agent_mode: str = ""
    translation: str = ""
    user_id: str = ""
    difficulty_level: str = "hard"
    first_token_time_ms: int | float | None = None
    finish_answer_time_ms: int | float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_time_ms: int | None = None
    meta: OutputMeta = Field(default_factory=OutputMeta)
