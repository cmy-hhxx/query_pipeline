from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_question(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value).replace("\u3000", " "))
    return re.sub(r"\s+", " ", text).strip()


def question_length_without_punctuation(question: str) -> int:
    count = 0
    for ch in question:
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "S", "Z")):
            continue
        count += 1
    return count
