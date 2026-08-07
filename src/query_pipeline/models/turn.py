from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """Canonical turn after input adapters; stages never read dialect field names."""

    question: str = ""
    answer: str = ""
    trace_id: str = ""
    run_id: str = ""
    chain: list[Any] = Field(default_factory=list)
    request_time: str = ""
    first_token_ms: int | float | None = None
    total_duration_ms: int | float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    user_id: str = ""
    status: str | None = "completed"
    outcome: str | None = "success"
    tool_names: str = ""  # fallback when chain has no tools (from tool_names / tool_names_text)


class Session(BaseModel):
    thread_id: str = ""
    turns: list[Turn] = Field(default_factory=list)
    # chat = last turn only; session = all turns subject to step1/eligibility
    candidate_mode: Literal["all", "last_only"] = "all"
