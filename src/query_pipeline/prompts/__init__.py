from __future__ import annotations

from query_pipeline.prompts.unified_label import UNIFIED_LABEL_V1

PROMPTS: dict[str, str] = {
    "unified_label_v1": UNIFIED_LABEL_V1,
}


def resolve_prompt(prompt_id: str) -> str:
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPTS))
        raise ValueError(f"unknown prompt_id {prompt_id!r}; available prompts: {available}") from exc
