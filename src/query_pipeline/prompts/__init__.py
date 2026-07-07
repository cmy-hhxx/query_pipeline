from __future__ import annotations

from query_pipeline.prompts.unified_label import UNIFIED_LABEL

PROMPTS: dict[str, str] = {
    "unified_label": UNIFIED_LABEL,
}


def resolve_prompt(prompt_id: str) -> str:
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPTS))
        raise ValueError(f"unknown prompt_id {prompt_id!r}; available prompts: {available}") from exc
