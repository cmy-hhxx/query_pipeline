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

INTENT_LABELS = {
    "数据取数计算",
    "指标型标的筛选",
    "事件概念型标的筛选",
    "公开信息查询解读",
    "时效性投教百科问答",
    "策略与事件回测",
    "标的趋势预测",
    "投资标的推荐",
    "标的四维深度分析",
    "宏观与市场分析",
    "资产配置与仓位管理",
    "交易点位规划",
    "闲聊情感陪伴",
    "非时效通用知识问答",
    "文生图创作",
    "文学内容创作",
    "通用工具类任务",
    "智能客服服务",
    "多媒体内容检索",
    "系统指令执行",
    "金融数据可视化",
    "文件生成导出",
    "客户端操作执行",
    "事件预测",
}

DEMAND_LABELS = {
    "数学",
    "用户 KYC 理解",
    "逻辑推理、预测能力",
}

DOMAIN_LABELS = {
    "",
    "A股股票",
    "基金",
    "港股",
    "美股",
    "新三板",
    "指数",
    "可转债",
    "期货",
    "基金公司",
    "基金经理",
    "宏观",
    "债券",
    "全量债券",
    "银行理财",
    "市场环境",
    "同花顺保险",
}

LEVEL_LABELS = {"低", "中", "高"}


class UnifiedLabelResult(BaseModel):
    is_complex: bool
    category_id: str | None = None
    category_name: str | None = None
    is_multi_turn: bool
    difficulty_score: float
    difficulty_reason: str
    reason: str
    intent_labels: list[str] | None = None
    demand_labels: list[str] | None = None
    domain_label: str | None = None
    query_quality: str | None = None
    query_difficulty: str | None = None

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

    @field_validator("intent_labels")
    @classmethod
    def validate_intent_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        invalid = sorted(label for label in value if label not in INTENT_LABELS)
        if invalid:
            raise ValueError(f"invalid intent_labels: {invalid}")
        return value

    @field_validator("demand_labels")
    @classmethod
    def validate_demand_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        invalid = sorted(label for label in value if label not in DEMAND_LABELS)
        if invalid:
            raise ValueError(f"invalid demand_labels: {invalid}")
        return value

    @field_validator("domain_label")
    @classmethod
    def validate_domain_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        label = value.strip()
        if label not in DOMAIN_LABELS:
            raise ValueError(f"invalid domain_label: {label}")
        return label

    @field_validator("query_quality", "query_difficulty")
    @classmethod
    def validate_level_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        label = value.strip()
        if label not in LEVEL_LABELS:
            raise ValueError(f"invalid level label: {label}")
        return label

    @model_validator(mode="after")
    def validate_category_name(self) -> UnifiedLabelResult:
        if self.category_id is None:
            self.category_name = None
        else:
            expected = CATEGORIES[self.category_id]
            if self.category_name not in (None, expected):
                raise ValueError(f"category_name must match category_id {self.category_id}")
            self.category_name = expected
        if self.query_difficulty is not None:
            expected_difficulty = _difficulty_level(self.difficulty_score)
            if self.query_difficulty != expected_difficulty:
                raise ValueError(
                    f"query_difficulty must be {expected_difficulty!r} for score {self.difficulty_score}"
                )
        return self

    def to_output(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "is_complex": self.is_complex,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "is_multi_turn": self.is_multi_turn,
            "difficulty_score": self.difficulty_score,
            "difficulty_reason": self.difficulty_reason,
            "reason": self.reason,
        }
        if self.intent_labels is not None:
            output["intent_labels"] = self.intent_labels
        if self.demand_labels is not None:
            output["demand_labels"] = self.demand_labels
        if self.domain_label is not None:
            output["domain_label"] = self.domain_label
        if self.query_quality is not None:
            output["query_quality"] = self.query_quality
        if self.query_difficulty is not None:
            output["query_difficulty"] = self.query_difficulty
        return output


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


def _difficulty_level(score: float) -> str:
    if score <= 1.9:
        return "低"
    if score <= 3.4:
        return "中"
    return "高"
