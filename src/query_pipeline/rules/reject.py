from __future__ import annotations

import re

from query_pipeline.rules.normalize import normalize_question

LOW_VALUE_COMMON = {
    "好的", "好", "要", "需要", "可以", "是", "否", "嗯", "恩", "对", "行", "继续",
    "帮我", "看看", "分析一下", "转人工", "客服", "1", "2", "3", "A", "B", "a", "b",
    "ok", "OK", "PC", "PC端",
}

PAGE_OR_PROMPT_CONTEXT_RE = re.compile(
    r"(图片是用户所处网页截图|Pageinfo|page_info_prompt|当前用户query|供你总结|不要透露你根据截图|"
    r"忽略图右侧对话框|<\|上下文历史\|>|<\|用户问题\|>)",
    re.I,
)

SUBMISSION_TEMPLATE_FRAGMENT_RE = re.compile(
    r"^您好,请按以下Step1,Step2,Step3步骤,完成[“\"]?提交"
)

CONTEXT_ONLY_FOLLOWUP_RE = re.compile(
    r"^(继续|接着|再来|那|这个|那个|这些|上述|上面|刚才|前面|上一[个条轮]|前述|这[个些])"
    r"([\s,，。!?！？]*)$"
)

CONTEXT_MARKER_RE = re.compile(
    r"(上面|上述|刚才|前面|前述|继续|接着|上一[个条轮]|这个|那个|这些|该股|该公司|这个基金|这只)"
)


def zh_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def generic_reject_reason(question: str, *, min_length: int = 6) -> str:
    text = normalize_question(question)
    if not text:
        return "blank"
    if PAGE_OR_PROMPT_CONTEXT_RE.search(text):
        return "embedded_page_or_prompt_context"
    if SUBMISSION_TEMPLATE_FRAGMENT_RE.search(text):
        return "submission_template_fragment"
    if len(text) > 2000:
        return "too_long_gt2000"
    if text in LOW_VALUE_COMMON:
        return "low_value_common"
    if CONTEXT_ONLY_FOLLOWUP_RE.fullmatch(text):
        return "context_only_followup"
    if re.fullmatch(r"[A-Za-z]?\d+(?:\.\d+)?(?:\.[A-Za-z]+)?", text):
        return "number_or_code_only"
    if re.fullmatch(r"[A-Za-z0-9_.\-]{1,15}", text) and zh_count(text) == 0:
        return "ascii_token_only"
    if len(text) < min_length:
        return "too_short"
    if zh_count(text) <= 2 and len(text) <= 12:
        return "mostly_symbol_short"
    return ""
