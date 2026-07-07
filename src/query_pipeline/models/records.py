from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


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


class ClassifyResult(BaseModel):
    is_complex: bool
    category_id: str | None = None
    category_name: str | None = None
    reason: str

    @field_validator("category_id")
    @classmethod
    def validate_category_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in CATEGORIES:
            raise ValueError(f"invalid category_id: {value}")
        return value

    def to_record_fields(self) -> dict[str, Any]:
        return {
            "is_complex": self.is_complex,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "judge_reason": self.reason,
        }


class DifficultyResult(BaseModel):
    is_multi_turn: bool = Field(alias="是否多轮")
    difficulty_score: float = Field(alias="难度评分")
    difficulty_reason: str = Field(alias="难度评分的理由")

    model_config = {"populate_by_name": True}

    @field_validator("difficulty_score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        score = round(float(value), 1)
        if not 0.0 <= score <= 5.0:
            raise ValueError("difficulty_score must be in [0, 5]")
        return score

    @field_validator("difficulty_reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("difficulty_reason must be non-empty")
        return value.strip()

    def to_record_fields(self) -> dict[str, Any]:
        return {
            "is_multi_turn": self.is_multi_turn,
            "difficulty_score": self.difficulty_score,
            "difficulty_reason": self.difficulty_reason,
        }


def parse_classify_response(raw: str) -> ClassifyResult:
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            raise ValueError(f"cannot parse classify response: {text[:200]}")
        data = json.loads(match.group(1))
    return ClassifyResult.model_validate(
        {
            "is_complex": data.get("is_complex"),
            "category_id": data.get("category_id"),
            "category_name": data.get("category_name"),
            "reason": data.get("reason", ""),
        }
    )


def parse_difficulty_response(raw: str) -> DifficultyResult:
    data = json.loads(raw)
    return DifficultyResult.model_validate(data)
