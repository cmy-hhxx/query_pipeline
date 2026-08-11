from __future__ import annotations

# PROMPTS 惰性构建：模板 fail-loud 校验（parse_* 对格式错误 raise）不能发生在
# 模块导入期——cli.py import api → runner → steps → prompts 全链，模板一个小笔误
# 会让 `query-pipeline --help` / `suggest`（运行时根本不碰 PROMPTS）全部崩溃。
# 首次 resolve_prompt 时才读 templates 并校验，错误指向实际使用提示词的命令。

PROMPTS: dict[str, str] = {}
_BUILT = False


def _build_prompts() -> dict[str, str]:
    from query_pipeline.prompts.assemble import (
        build_complex_classify_prompt,
        build_normal_classify_prompt,
        build_verify_prompt,
        load_complex_quality_policy,
    )
    from query_pipeline.prompts.complexity_gate import COMPLEXITY_GATE
    from query_pipeline.prompts.dedup import DEDUP_PAIR
    from query_pipeline.prompts.value_gate import VALUE_GATE
    from query_pipeline.prompts.segment import SEGMENT
    from query_pipeline.prompts.template_family import TEMPLATE_FAMILY
    from query_pipeline.prompts.translate import TRANSLATE
    from query_pipeline.prompts.verify import VERIFY_COMPLEX, VERIFY_RECHECK

    policy = load_complex_quality_policy()
    return {
        "segment": SEGMENT,
        "classify_complex": build_complex_classify_prompt(),
        "classify_normal": build_normal_classify_prompt(),
        "value_gate": VALUE_GATE + "\n\n---\n\n" + policy,
        "complexity_gate": COMPLEXITY_GATE + "\n\n---\n\n" + policy,
        "dedup_pair": DEDUP_PAIR,
        "template_family": TEMPLATE_FAMILY + "\n\n---\n\n" + policy,
        "verify_complex": build_verify_prompt(VERIFY_COMPLEX),
        "verify_recheck": build_verify_prompt(VERIFY_RECHECK),
        "translate": TRANSLATE,
    }


def _ensure_prompts() -> None:
    global _BUILT
    if _BUILT:
        return
    # setdefault：允许测试 patch.dict 预置的条目（如模拟 prompt 内容变化）优先，
    # 其余键用真实构建结果补齐；首次构建后置位，后续 resolve 不再触碰模板。
    for key, value in _build_prompts().items():
        PROMPTS.setdefault(key, value)
    _BUILT = True


def resolve_prompt(prompt_id: str) -> str:
    _ensure_prompts()
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPTS))
        raise ValueError(f"unknown prompt_id {prompt_id!r}; available prompts: {available}") from exc
