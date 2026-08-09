from __future__ import annotations

from query_pipeline.prompts.assemble import (
    build_complex_classify_prompt,
    build_normal_classify_prompt,
    build_verify_prompt,
)
from query_pipeline.prompts.complex_judge import COMPLEX_JUDGE
from query_pipeline.prompts.segment import SEGMENT
from query_pipeline.prompts.translate import TRANSLATE
from query_pipeline.prompts.verify import VERIFY_COMPLEX, VERIFY_RECHECK

PROMPTS: dict[str, str] = {
    "segment": SEGMENT,
    "complex_judge": COMPLEX_JUDGE,
    "classify_complex": build_complex_classify_prompt(),
    "classify_normal": build_normal_classify_prompt(),
    "verify_complex": build_verify_prompt(VERIFY_COMPLEX),
    "verify_recheck": build_verify_prompt(VERIFY_RECHECK),
    "translate": TRANSLATE,
}


def resolve_prompt(prompt_id: str) -> str:
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPTS))
        raise ValueError(f"unknown prompt_id {prompt_id!r}; available prompts: {available}") from exc
