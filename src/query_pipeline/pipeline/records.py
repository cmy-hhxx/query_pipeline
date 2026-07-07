from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OUTPUT_FIELD = "query_pipeline_output"


@dataclass(frozen=True)
class TextSelection:
    found: bool
    value: Any = None
    missing_at: str | None = None


def resolve_dot_path(record: dict[str, Any], path: str) -> TextSelection:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return TextSelection(found=False, missing_at=part)
        current = current[part]
    return TextSelection(found=True, value=current)


def is_text_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def pipeline_output(record: dict[str, Any]) -> dict[str, Any]:
    output = record.get(OUTPUT_FIELD)
    return output if isinstance(output, dict) else {}


def set_pipeline_output(record: dict[str, Any], **fields: Any) -> dict[str, Any]:
    updated = dict(record)
    output = dict(pipeline_output(record))
    output.update(fields)
    updated[OUTPUT_FIELD] = output
    return updated


def update_rule_signals(record: dict[str, Any], **signals: Any) -> dict[str, Any]:
    output = pipeline_output(record)
    rule_signals = dict(output.get("rule_signals") or {})
    rule_signals.update(signals)
    return set_pipeline_output(record, rule_signals=rule_signals)


def output_value(record: dict[str, Any], key: str, default: Any = None) -> Any:
    return pipeline_output(record).get(key, default)


def normalized_text(record: dict[str, Any]) -> str:
    value = output_value(record, "normalized_text", "")
    return value if isinstance(value, str) else ""


def source_line_number(record: dict[str, Any]) -> int | None:
    value = record.get("_line_number", record.get("line_number"))
    return value if isinstance(value, int) else None
