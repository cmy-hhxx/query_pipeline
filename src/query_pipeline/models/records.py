from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, field_validator, model_validator


CATEGORIES: dict[str, str] = {
    "01": "复杂取数计算类",
    "02": "预测类",
    "03": "分析研究类",
    "04": "机会挖掘类",
    "05": "资产配置类",
    "06": "账户诊断优化类",
    "07": "策略触发/设置类",
    "08": "目标跟踪执行类",
    "09": "动作类",
}


class UnifiedLabelResult(BaseModel):
    is_complex: bool
    category_id: str | None = None
    category_name: str | None = None
    is_multi_turn: bool
    difficulty_score: float
    difficulty_reason: str
    reason: str

    @field_validator("category_id")
    @classmethod
    def validate_category_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in CATEGORIES:
            raise ValueError(f"invalid category_id: {value}")
        return value

    @field_validator("difficulty_score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        score = round(float(value), 1)
        if not 0.0 <= score <= 5.0:
            raise ValueError("difficulty_score must be in [0, 5]")
        return score

    @field_validator("difficulty_reason", "reason")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("reason fields must be non-empty")
        return text

    @model_validator(mode="after")
    def validate_category_name(self) -> UnifiedLabelResult:
        if self.category_id is None:
            self.category_name = None
            return self
        expected = CATEGORIES[self.category_id]
        if self.category_name not in (None, expected):
            raise ValueError(f"category_name must match category_id {self.category_id}")
        self.category_name = expected
        return self

    def to_output(self) -> dict[str, Any]:
        return {
            "is_complex": self.is_complex,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "is_multi_turn": self.is_multi_turn,
            "difficulty_score": self.difficulty_score,
            "difficulty_reason": self.difficulty_reason,
            "reason": self.reason,
        }


def parse_unified_label_response(raw: str) -> UnifiedLabelResult:
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            raise ValueError(f"cannot parse unified label response: {text[:200]}")
        data = json.loads(match.group(1))
    return UnifiedLabelResult.model_validate(data)
