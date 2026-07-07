from __future__ import annotations

import re
import unicodedata
from typing import Any

from datasketch import MinHash, MinHashLSH

from query_pipeline.rules.normalize import normalize_question

STOCK_CODE_RE = re.compile(r"\b[036]\d{5}\b")
DATE_RE = re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?")
COMPANY_SUFFIX_RE = re.compile(r"(股份有限公司|有限公司|集团|股份)$")
PUNCTUATION_RE = re.compile(
    r"[\s`~!@#$%^&*()_+\-=\[\]{};:'\"\\|,.<>/?，。！？、；：“”‘’（）【】《》…—-]+"
)


def theme_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = STOCK_CODE_RE.sub("<code>", text)
    text = DATE_RE.sub("<date>", text)
    text = COMPANY_SUFFIX_RE.sub("<co>", text)
    text = PUNCTUATION_RE.sub("", text)
    text = re.sub(r"\d+", "<num>", text)
    return text


def char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def build_minhash(text: str, *, num_perm: int, normalization: str | None) -> MinHash:
    normalized = theme_normalize(text) if normalization == "theme" else normalize_question(text).lower()
    shingles = char_ngrams(normalized, 3)
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh


def minhash_dedup(
    records: list[dict[str, Any]],
    *,
    text_field: str,
    num_perm: int,
    threshold: float,
    normalization: str | None,
    method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    minhashes: dict[str, MinHash] = {}

    for index, record in enumerate(records):
        question = str(record.get(text_field, ""))
        record_id = record.get("id") or f"row_{index}"
        mh = build_minhash(question, num_perm=num_perm, normalization=normalization)
        minhashes[record_id] = mh
        candidates = lsh.query(mh)
        if candidates:
            duplicate_of = candidates[0]
            removed.append(
                {
                    **record,
                    "reject_reason": "duplicate_minhash",
                    "duplicate_of_id": duplicate_of,
                }
            )
            continue
        lsh.insert(record_id, mh)
        kept.append(record)

    report = {
        "method": method if normalization != "theme" else "minhash_theme_normalized_char_3gram",
        "num_perm": num_perm,
        "threshold": threshold,
        "source_rows": len(records),
        "dedup_rows": len(kept),
        "removed_rows": len(removed),
    }
    return kept, removed, report
