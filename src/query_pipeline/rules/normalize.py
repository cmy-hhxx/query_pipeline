from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_question(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value).replace("\u3000", " "))
    return re.sub(r"\s+", " ", text).strip()
