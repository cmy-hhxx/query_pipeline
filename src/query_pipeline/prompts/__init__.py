from __future__ import annotations

from query_pipeline.prompts.complex_judge import COMPLEX_JUDGE
from query_pipeline.prompts.segment import SEGMENT

PROMPTS: dict[str, str] = {
    "segment": SEGMENT,
    "complex_judge": COMPLEX_JUDGE,
}


def resolve_prompt(prompt_id: str) -> str:
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPTS))
        raise ValueError(f"unknown prompt_id {prompt_id!r}; available prompts: {available}") from exc
