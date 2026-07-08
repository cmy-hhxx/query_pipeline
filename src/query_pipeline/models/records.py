from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


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

CORE_LABEL_FIELDS = {
    "is_complex",
    "category_id",
    "is_multi_turn",
    "difficulty_score",
    "difficulty_reason",
    "category_reason",
}


class CoreLabelResult(BaseModel):
    is_complex: bool
    category_id: str | None = None
    is_multi_turn: bool
    difficulty_score: float | None = None
    difficulty_reason: str | None = None
    category_reason: str
    extra_fields: tuple[str, ...] = Field(default_factory=tuple, exclude=True)

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
    def validate_score(cls, value: float | None) -> float | None:
        if value is None:
            return None
        score = round(float(value), 1)
        if not 0.0 <= score <= 5.0:
            raise ValueError("difficulty_score must be in [0, 5]")
        return score

    @field_validator("difficulty_reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("difficulty_reason must be non-empty when present")
        return text

    @field_validator("category_reason")
    @classmethod
    def validate_category_reason(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("category_reason must be non-empty")
        return text

    @model_validator(mode="after")
    def validate_complex_contract(self) -> CoreLabelResult:
        if not self.is_complex:
            self.category_id = None
            self.difficulty_score = None
            self.difficulty_reason = None
            return self
        if self.category_id is None:
            raise ValueError("category_id is required when is_complex is true")
        if self.difficulty_score is None:
            raise ValueError("difficulty_score is required when is_complex is true")
        if self.difficulty_reason is None:
            raise ValueError("difficulty_reason is required when is_complex is true")
        return self

    @property
    def category_name(self) -> str | None:
        if self.category_id is None:
            return None
        return CATEGORIES[self.category_id]

    def to_output(self) -> dict[str, Any]:
        return {
            "is_complex": self.is_complex,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "is_multi_turn": self.is_multi_turn,
            "difficulty_score": self.difficulty_score,
            "difficulty_reason": self.difficulty_reason,
            "category_reason": self.category_reason,
        }

    def to_cache_label(self) -> dict[str, Any]:
        return {
            "is_complex": self.is_complex,
            "category_id": self.category_id,
            "is_multi_turn": self.is_multi_turn,
            "difficulty_score": self.difficulty_score,
            "difficulty_reason": self.difficulty_reason,
            "category_reason": self.category_reason,
        }


def parse_core_label_response(raw: str) -> CoreLabelResult:
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            raise ValueError(f"cannot parse core label response: {text[:200]}")
        data = json.loads(match.group(1))
    if not isinstance(data, dict):
        raise ValueError("core label response must be a JSON object")
    return parse_core_label_payload(data)


def parse_core_label_payload(data: dict[str, Any]) -> CoreLabelResult:
    extra_fields = tuple(sorted(key for key in data if key not in CORE_LABEL_FIELDS))
    payload = {key: data[key] for key in CORE_LABEL_FIELDS if key in data}
    payload["extra_fields"] = extra_fields
    return CoreLabelResult.model_validate(payload)
