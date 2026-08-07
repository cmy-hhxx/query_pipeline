from __future__ import annotations

import re
import unicodedata
from typing import Any

# 实体槽位占位符:把 ticker/数字/日期抽象成统一槽,使"同模板不同实体"的查询骨架一致。
# 占位符用尖括号包裹并与周围以空格隔离,避免与真实词冲突,tokenize 时原样保留。
SLOT_PLACEHOLDERS = frozenset({"<ticker>", "<num>", "<date>"})

# 按序应用:先 ticker 后货币(让 $AAPL 归 ticker、$100 归 num);
# 先日期后裸数字(让 2026-08-03 归 date,而不是拆成 <num>-<num>-<num>)。
_SLOT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\$[a-z][a-z0-9.\-]*", re.IGNORECASE), "<ticker>"),
    (re.compile(r"(?:\$|us\$|hk\$|rmb|人民币|美元|港币|元)\s*\d[\d,]*(?:\.\d+)?", re.IGNORECASE), "<num>"),
    (re.compile(r"\d[\d,]*(?:\.\d+)?\s*%", re.IGNORECASE), "<num>"),
    (re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,4}年\d{1,2}月(?:\d{1,2}日)?", re.IGNORECASE), "<date>"),
    # 裸数字只匹配独立 token,不碰嵌在单词里的数字(如 T0、Q1、v2、4hr),否则会把这些标识符错误坍缩。
    (re.compile(r"(?<![0-9a-z一-鿿])\d[\d,]*(?:\.\d+)?(?![0-9a-z一-鿿])", re.IGNORECASE), "<num>"),
]
_WORD_SPLIT = re.compile(r"\W+")
_CJK_CHAR = re.compile(r"([一-鿿])")


def normalize_question(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value).replace("　", " "))
    return re.sub(r"\s+", " ", text).strip()


def slot_entities(text: str) -> str:
    """把 ticker/数字/百分比/日期替换为槽位占位符,并折叠空白。"""
    text = normalize_question(text)
    for pattern, replacement in _SLOT_PATTERNS:
        text = pattern.sub(lambda _m, repl=replacement: f" {repl} ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_question(text: str, *, entity_slot: bool = True) -> set[str]:
    """槽化后的问题 token 集合,用于精确 Jaccard 去重。

    槽位占位符原样保留;CJK 拆成单字(无依赖的中文切分);其余按非词边界切分。
    无空格语言(如泰语)会整段成一个 token —— 字面近似去重对它们不生效,可接受。
    """
    text = (slot_entities(text) if entity_slot else normalize_question(text)).lower()
    tokens: set[str] = set()
    for word in text.split():
        if entity_slot and word in SLOT_PLACEHOLDERS:
            tokens.add(word)
            continue
        for sub in _WORD_SPLIT.split(word):
            if not sub:
                continue
            for part in _CJK_CHAR.split(sub):
                if part:
                    tokens.add(part)
    return tokens


def exact_token_jaccard(left: set[str], right: set[str]) -> float:
    """精确 token-set Jaccard;任一为空集返回 0.0(空文本永不与任何查询合并)。"""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


def question_length_without_punctuation(question: str) -> int:
    count = 0
    for ch in question:
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "S", "Z")):
            continue
        count += 1
    return count
