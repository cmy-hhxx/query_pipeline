from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from query_pipeline.models.output import OutputRow
from query_pipeline.steps.answer_gate_stage import (
    MIN_ANSWER_LEN,
    _REFUSAL_PATTERNS,
    truncation_reason,
)
from query_pipeline.taxonomy import COMPLEX_PREFIX, load_taxonomy
from query_pipeline.post.dedup import dedup_rows
from query_pipeline.config.models import DedupConfig

# Rule thresholds — code constants, no per-rule config (project convention).
# MIN_ANSWER_LEN / 截断判定与 answer_gate 共用单一实现（第四轮 #7：两侧阈值与
# 条件曾各自复制，必然漂移；_DANGLING_END 也曾本地重定义 shadowing 死导入）。
QUESTION_MIN_LEN = 5
QUESTION_MAX_LEN = 2000
EMPTY_FIELD_RATIO = 0.10
CATEGORY_SKEW_MAX_SHARE = 0.50
NEAR_CONSTANT_SHARE = 0.90  # soft tier: top value covers >=90% of records
NEAR_DUP_THRESHOLD = 0.85

# Content fields monitored for constant/near-constant values at the dataset level.
_CONTENT_FIELDS = ("text_answer", "raw_answer", "input.text", "category", "tools", "meta.reason")

# Key fields whose empty rate is tracked at the dataset level.
# translation 不入空值率统计：中文问句按规范 translation=null 属正常（见 _check_meta 逐条校验）
_KEY_FIELDS = ("trace_id", "source_case_id", "category", "input.text", "text_answer", "meta.reason")


@dataclass(frozen=True)
class RuleCheck:
    name: str
    check: Callable[[dict[str, Any]], tuple[bool, str]]  # -> (ok, detail)


@dataclass(frozen=True)
class DatasetRule:
    name: str
    check: Callable[[list[dict[str, Any]]], tuple[bool, str, list[str]]]  # -> (ok, detail, evidence)


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if "一" <= ch <= "鿿") / len(text)


def _has_cjk(text: str) -> bool:
    # 与 post.translate.needs_translation 口径一致：CJK 占比 >= 30% 视为中文，
    # 否则视为需要翻译（避免日文/韩文/混合语言行误报）。
    return _cjk_ratio(text) >= 0.3


def _field_value(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == []


# ---------------------------------------------------------------------------
# Per-record rules
# ---------------------------------------------------------------------------

def _check_structure(row: dict[str, Any]) -> tuple[bool, str]:
    try:
        OutputRow.model_validate(row)
        return True, "ok"
    except Exception as exc:  # pydantic.ValidationError and friends
        return False, str(exc)[:200]


def _check_question(row: dict[str, Any]) -> tuple[bool, str]:
    inp = row.get("input")
    text = inp.get("text") if isinstance(inp, dict) else None
    if not isinstance(text, str) or not text.strip():
        return False, "input.text 缺失或为空"
    length = len(text.strip())
    if length < QUESTION_MIN_LEN:
        return False, f"input.text 过短（{length}<{QUESTION_MIN_LEN}）"
    if length > QUESTION_MAX_LEN:
        return False, f"input.text 过长（{length}>{QUESTION_MAX_LEN}）"
    if "�" in text:
        return False, "input.text 含乱码字符"
    return True, "ok"


def _check_category(row: dict[str, Any]) -> tuple[bool, str]:
    category = row.get("category")
    if not isinstance(category, str) or not category:
        return False, "category 缺失或非字符串"
    if category == "other":
        # 普通分类的兜底标签（classify_normal 允许 other，仅 normal 行可用）
        return (True, "ok") if row.get("difficulty_level") == "normal" else (False, "other 仅允许出现在 normal 行")
    cat = next((c for c in load_taxonomy().all() if c.path == category), None)
    if cat is None:
        return False, f"未知 category：{category!r}"
    return True, "ok"


def _check_chain(row: dict[str, Any]) -> tuple[bool, str]:
    if row.get("capture_mode") == "end2end":
        # end2end 输入本身无 chain（工具调用由 tool_count 兜底过 rule_gate），
        # 只校验 tools 非空，不再要求 chain。
        tools = row.get("tools")
        if not isinstance(tools, list) or not tools:
            return False, "end2end 行 tools 缺失或为空"
        return True, "ok"
    chain = row.get("chain")
    if not isinstance(chain, list) or not chain:
        return False, "chain 缺失、非列表或为空"
    for i, hop in enumerate(chain):
        if not isinstance(hop, dict):
            return False, f"chain[{i}] 非对象"
        if not isinstance(hop.get("plan"), str):
            return False, f"chain[{i}].plan 非字符串"
        tools = hop.get("tools")
        if not isinstance(tools, list):
            return False, f"chain[{i}].tools 非列表"
        for j, tool in enumerate(tools):
            if not isinstance(tool, dict):
                return False, f"chain[{i}].tools[{j}] 非对象"
            if not isinstance(tool.get("name"), str) or not tool["name"]:
                return False, f"chain[{i}].tools[{j}].name 缺失或为空"
            if not isinstance(tool.get("input"), dict):
                return False, f"chain[{i}].tools[{j}].input 非对象"
            if not isinstance(tool.get("output"), str):
                return False, f"chain[{i}].tools[{j}].output 非字符串"
    return True, "ok"


def _check_answer(row: dict[str, Any]) -> tuple[bool, str]:
    text = row.get("text_answer")
    if not isinstance(text, str) or not text.strip():
        return False, "text_answer 缺失或为空"
    length = len(text.strip())
    if length < MIN_ANSWER_LEN:
        return False, f"text_answer 过短（{length}<{MIN_ANSWER_LEN}）"
    return True, "ok"


def _check_refusal(row: dict[str, Any]) -> tuple[bool, str]:
    text = (row.get("text_answer") or "").strip()
    if not text:
        return False, "text_answer 为空"
    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(text):
            return False, f"回答疑似拒绝（命中 {pattern.pattern[:40]}）：{text[:60]}"
    return True, "ok"


def _check_event_type(row: dict[str, Any]) -> tuple[bool, str]:
    meta = row.get("meta")
    event = meta.get("last_event_type") if isinstance(meta, dict) else None
    if event is None:
        return True, "ok"  # chat rows carry no event field
    if event != "runFinished":
        return False, f"last_event_type={event}（回答不完整/非正常回答）"
    return True, "ok"


def _check_truncation(row: dict[str, Any]) -> tuple[bool, str]:
    """与 answer_gate 共用同一判定（truncation_reason）：无问句不判截断。

    空回答由 answer 规则负责，这里不重复报（避免两条规则对同一缺陷各报一次）。
    """
    inp = row.get("input")
    question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
    reason = truncation_reason(row.get("text_answer"), question=question)
    if reason is None:
        return True, "ok"
    text = str(row.get("text_answer") or "").strip()
    return False, f"回答以未完结标点结尾，疑似截断：…{text[-40:]}"


def _check_timing(row: dict[str, Any]) -> tuple[bool, str]:
    first = row.get("first_token_time_ms")
    finish = row.get("finish_answer_time_ms")
    if first is not None and finish is not None:
        try:
            if float(first) > float(finish):
                return False, f"first_token_time_ms({first}) > finish_answer_time_ms({finish})"
        except (TypeError, ValueError):
            return False, f"时间字段非数值：first={first!r} finish={finish!r}"
    for field in ("input_tokens", "output_tokens"):
        value = row.get(field)
        if value is not None and not isinstance(value, (int, float)):
            return False, f"{field} 非数值：{value!r}"
        if isinstance(value, (int, float)) and value < 0:
            return False, f"{field} 为负：{value!r}"
    return True, "ok"


def _check_meta(row: dict[str, Any]) -> tuple[bool, str]:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return False, "meta 缺失或非对象"
    reason = meta.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False, "meta.reason 缺失或为空"
    inp = row.get("input")
    question = inp.get("text") if isinstance(inp, dict) else ""
    translation = row.get("translation")
    if _has_cjk(str(question)):
        # 中文问句不需要翻译：translation 应为 null（旧语义是回填原文，已废弃）
        if translation is not None and str(translation).strip():
            return False, "中文问句不应有 translation（应为 null）"
    else:
        # 非中文问句需要翻译；翻译失败是故意 fail-open（translate 阶段落
        # meta.translate_failed 标记，filter_out.jsonc 明示"翻译失败 → null"），
        # 与"从未翻译"（null 且无失败记录）区分开——后者仍判 fail。
        if isinstance(translation, str) and translation.strip():
            return True, "ok"
        if meta.get("translate_failed"):
            return True, "ok"
        return False, "非中文问句缺少 translation 翻译"
    return True, "ok"


PER_RECORD_RULES: list[RuleCheck] = [
    RuleCheck("structure", _check_structure),
    RuleCheck("question", _check_question),
    RuleCheck("category", _check_category),
    RuleCheck("chain", _check_chain),
    RuleCheck("answer", _check_answer),
    RuleCheck("truncation", _check_truncation),
    RuleCheck("refusal", _check_refusal),
    RuleCheck("event_type", _check_event_type),
    RuleCheck("timing", _check_timing),
    RuleCheck("meta", _check_meta),
]


def check_record(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"rule": rule.name, "ok": ok, "detail": detail}
        for rule in PER_RECORD_RULES
        for ok, detail in (rule.check(row),)
    ]


# ---------------------------------------------------------------------------
# Dataset-level rules
# ---------------------------------------------------------------------------

def _dataset_constant_field(records: list[dict[str, Any]]) -> tuple[bool, str, list[str]]:
    if not records:
        return True, "无记录", []
    n = len(records)
    issues: list[str] = []
    for field in _CONTENT_FIELDS:
        counts: dict[Any, int] = {}
        for row in records:
            value = _field_value(row, field)
            counts[value] = counts.get(value, 0) + 1
        if len(counts) == 1:
            issues.append(f"字段 {field} 在所有 {n} 条记录中恒值")
        else:
            top_value, top_count = max(counts.items(), key=lambda kv: kv[1])
            if top_count / n >= NEAR_CONSTANT_SHARE:
                issues.append(f"字段 {field} 取值高度集中（单一值占 {top_count / n:.0%}）")
    if issues:
        return False, "；".join(issues), issues
    return True, "ok", []


def _dataset_near_duplicate(records: list[dict[str, Any]]) -> tuple[bool, str, list[str]]:
    if len(records) < 2:
        return True, "记录太少，跳过", []
    cfg = DedupConfig(enabled=True, threshold=NEAR_DUP_THRESHOLD)
    kept, dropped = dedup_rows(records, cfg)
    if not dropped:
        return True, f"未发现相似度≥{NEAR_DUP_THRESHOLD} 的近重复问句", []
    evidence = [
        f"{d['trace_id']} 与 {d['dedup_of_trace_id']} 相似度 {d['similarity']:.3f}：{d['text']}"
        for d in dropped[:20]
    ]
    return False, f"发现 {len(dropped)} 条近重复问句（实体槽化 token-Jaccard≥{NEAR_DUP_THRESHOLD}）", evidence


def _dataset_length_outlier(records: list[dict[str, Any]]) -> tuple[bool, str, list[str]]:
    if not records:
        return True, "无记录", []
    issues: list[str] = []
    for field, lower, upper in (
        ("input.text", QUESTION_MIN_LEN, QUESTION_MAX_LEN),
        ("text_answer", MIN_ANSWER_LEN, None),
    ):
        lengths = [
            len(str(v)) if isinstance(v, str) else 0 for v in (_field_value(r, field) for r in records)
        ]
        too_short = sum(1 for ln in lengths if ln < lower)
        too_long = sum(1 for ln in lengths if upper is not None and ln > upper)
        if too_short or too_long:
            issues.append(f"{field}：过短 {too_short} 条、过长 {too_long} 条")
    if issues:
        return False, "；".join(issues), issues
    return True, "长度分布正常", []


def _dataset_category_skew(records: list[dict[str, Any]]) -> tuple[bool, str, list[str]]:
    if not records:
        return True, "无记录", []
    counts: dict[str, int] = {}
    for row in records:
        category = str(row.get("category") or "")
        counts[category] = counts.get(category, 0) + 1
    n = len(records)
    top_count = max(counts.values())
    top_share = top_count / n
    evidence = [f"{cat}: {cnt}（{cnt / n:.0%}）" for cat, cnt in sorted(counts.items(), key=lambda kv: -kv[1])]
    # complex id 从 taxonomy path 提取（complex-topic/09-… → "09"）；normal id
    # （01-16）与 complex id（01-09）会碰撞，绝不能计入 complex 覆盖集合。
    complex_present: set[str] = set()
    for cat in counts:
        if cat.startswith(COMPLEX_PREFIX):
            complex_present.add(cat[len(COMPLEX_PREFIX) :].split("-", 1)[0])
    zero_ids = [cid for cid in load_taxonomy().complex if cid not in complex_present]
    detail = f"最高类别占 {top_share:.0%}"
    if zero_ids:
        detail += f"；零记录类别：{', '.join(zero_ids)}"
    if top_share > CATEGORY_SKEW_MAX_SHARE:
        return False, f"类别分布偏斜：{detail}", evidence
    return True, detail, evidence


def _dataset_empty_rate(records: list[dict[str, Any]]) -> tuple[bool, str, list[str]]:
    if not records:
        return True, "无记录", []
    n = len(records)
    issues: list[str] = []
    for field in _KEY_FIELDS:
        empty = sum(1 for r in records if _is_empty(_field_value(r, field)))
        ratio = empty / n
        if ratio > EMPTY_FIELD_RATIO:
            issues.append(f"{field} 空值率 {ratio:.0%}")
    if issues:
        return False, "；".join(issues), issues
    return True, "ok", []


def _dataset_unknown_fields(records: list[dict[str, Any]]) -> tuple[bool, str, list[str]]:
    known = set(OutputRow.model_fields.keys())
    # Ignore underscore-prefixed keys: reader bookkeeping (e.g. _line_number).
    unknown = sorted(
        {k for row in records for k in row.keys() if k not in known and not k.startswith("_")}
    )
    if not unknown:
        return True, "无未知顶层字段", []
    return True, f"发现未知顶层字段（信息，不标失败）：{', '.join(unknown)}", unknown


DATASET_RULES: list[DatasetRule] = [
    DatasetRule("constant_field", _dataset_constant_field),
    DatasetRule("near_duplicate", _dataset_near_duplicate),
    DatasetRule("length_outlier", _dataset_length_outlier),
    DatasetRule("category_skew", _dataset_category_skew),
    DatasetRule("empty_field_rate", _dataset_empty_rate),
    DatasetRule("unknown_fields", _dataset_unknown_fields),
]


def run_dataset_rules(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"rule": rule.name, "ok": ok, "detail": detail, "evidence": evidence}
        for rule in DATASET_RULES
        for ok, detail, evidence in (rule.check(records),)
    ]
