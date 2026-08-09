from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OutputInput(BaseModel):
    text: str = ""
    image: str = ""  # 预留：图片（chat 源 judge_data.input 携带，当前管线不透传）
    file: str = ""   # 预留：附件（同上）


class OutputMeta(BaseModel):
    """扩展元信息（逃生字段）。

    extra="allow"：额外键会被 pydantic 原样保留并随 model_dump 输出，
    新增需求优先加进 meta，避免改动顶层结构（见 templates/filter_out.jsonc）。
    """

    model_config = ConfigDict(extra="allow")

    reason: str | None = None
    request_time: str = ""
    run_id: str = ""
    last_event_type: str | None = None


class ContextTurn(BaseModel):
    question: str = ""
    answer: str = ""


class OutputRow(BaseModel):
    """Annotated complex-query row — 字段与 templates/filter_out.jsonc 逐项对齐。

    capture_mode / user_cohort / answer_key / multimodal / model_version /
    release_id / agent_mode 是下游契约里的预留字段：当前管线取默认值或
    不填，由后续消费方填充；capture_mode 由输入是否带 chain 推导
    （带 → "full_link"，不带 → "end2end"）。删除或改语义前必须与用户确认。
    """

    capture_mode: str = "full_link"
    user_cohort: str = "regular"
    source_case_id: str = ""
    answer_key: str = ""
    trace_id: str = ""
    category: str = ""
    input: OutputInput = Field(default_factory=OutputInput)
    context: list[ContextTurn] = Field(default_factory=list)
    chain: list[Any] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    raw_answer: str = ""
    text_answer: str = ""
    multimodal: list[Any] = Field(default_factory=list)
    model_version: str = ""
    release_id: str = ""
    agent_mode: str = ""
    translation: str | None = None  # 问句译文（post_stage translate 填充）；原文已是中文或翻译失败 → null
    user_id: str = ""
    difficulty_level: str = "hard"
    first_token_time_ms: int | float | None = None
    finish_answer_time_ms: int | float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_time_ms: int | None = None
    meta: OutputMeta = Field(default_factory=OutputMeta)
