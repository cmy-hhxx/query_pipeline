from __future__ import annotations

from typing import Any

from query_pipeline.adapters.chat import adapt_chat
from query_pipeline.adapters.session import adapt_session
from query_pipeline.models.turn import Session


def adapt_record(record: dict[str, Any], fmt: str) -> Session:
    if fmt == "session":
        return adapt_session(record)
    if fmt == "chat":
        return adapt_chat(record)
    raise ValueError(f"unknown input.format: {fmt!r}")
