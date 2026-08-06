from __future__ import annotations

from query_pipeline.prompts.complex_judge import COMPLEX_JUDGE
from query_pipeline.prompts.segment import SEGMENT
from query_pipeline.prompts.translate import TRANSLATE
from query_pipeline.prompts.verify import VERIFY_COMPLEX

PROMPTS: dict[str, str] = {
    "segment": SEGMENT,
    "complex_judge": COMPLEX_JUDGE,
    "verify_complex": VERIFY_COMPLEX,
    "translate": TRANSLATE,
}


def resolve_prompt(prompt_id: str) -> str:
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPTS))
        raise ValueError(f"unknown prompt_id {prompt_id!r}; available prompts: {available}") from exc
