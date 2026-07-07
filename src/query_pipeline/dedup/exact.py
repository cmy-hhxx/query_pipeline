from __future__ import annotations

import hashlib
import re

from query_pipeline.rules.normalize import normalize_question

PUNCTUATION_RE = re.compile(
    r"[\s`~!@#$%^&*()_+\-=\[\]{};:'\"\\|,.<>/?，。！？、；：“”‘’（）【】《》…—-]+"
)


def dedup_key(question: str) -> str:
    return PUNCTUATION_RE.sub("", normalize_question(question).lower())


def sha1_12(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
