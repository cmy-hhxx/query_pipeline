"""Dialect adapters: registered by name, matched by top-level marker keys.

Adding a new input dialect = register one adapter here (or import a module
that does): ``register_adapter("myfmt", ("top_key_a", "top_key_b"), fn)``.
Format sniffing and adapt_record both go through the registry, so a new
dialect is picked up everywhere with a single registration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from query_pipeline.adapters.chat import adapt_chat
from query_pipeline.adapters.session import adapt_session
from query_pipeline.models.turn import Session

SESSION = "session"
CHAT = "chat"


@dataclass(frozen=True)
class Adapter:
    name: str
    markers: tuple[str, ...]  # top-level keys that must all be present
    validate: Callable[[dict[str, Any]], bool]  # key presence + type check
    adapt: Callable[[dict[str, Any]], Session]

    def matches(self, record: dict[str, Any]) -> bool:
        if not all(key in record for key in self.markers):
            return False
        return self.validate(record)


ADAPTERS: dict[str, Adapter] = {}


def register_adapter(
    name: str,
    markers: tuple[str, ...],
    validate: Callable[[dict[str, Any]], bool],
    adapt: Callable[[dict[str, Any]], Session],
) -> Adapter:
    if name in ADAPTERS:
        raise ValueError(f"duplicate adapter registration: {name!r}")
    adapter = Adapter(name=name, markers=markers, validate=validate, adapt=adapt)
    ADAPTERS[name] = adapter
    return adapter


# 内置方言：session（thread_id+context）与 chat（judge_data 包装）。
register_adapter(
    SESSION,
    ("thread_id", "context"),
    lambda r: isinstance(r.get("thread_id"), str)
    and bool(r.get("thread_id"))
    and isinstance(r.get("context"), list),
    adapt_session,
)
register_adapter(
    CHAT,
    ("judge_data",),
    lambda r: isinstance(r.get("judge_data"), dict),
    adapt_chat,
)


def match_adapter(record: dict[str, Any]) -> str | None:
    """Return the name of the first adapter whose markers all match.

    A record carrying any marker key but matching no adapter is malformed
    (partial markers) and raises instead of being silently skipped.
    """
    marker_keys = {key for adapter in ADAPTERS.values() for key in adapter.markers}
    for adapter in ADAPTERS.values():
        if adapter.matches(record):
            return adapter.name
    if marker_keys.intersection(record):
        raise ValueError(
            f"record has partial or malformed format markers ({sorted(marker_keys)}), "
            "cannot classify; registered formats: " + ", ".join(sorted(ADAPTERS))
        )
    return None


def adapt_record(record: dict[str, Any], fmt: str) -> Session:
    try:
        adapter = ADAPTERS[fmt]
    except KeyError:
        raise ValueError(
            f"unknown input.format: {fmt!r}; registered adapters: {', '.join(sorted(ADAPTERS))}"
        ) from None
    return adapter.adapt(record)
