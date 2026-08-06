from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ConfigModel(BaseModel):
    model_config = {"extra": "forbid"}


class InputConfig(ConfigModel):
    path: Path


class OutputConfig(ConfigModel):
    dir: Path = Path("outputs")
    complex_queries: str = "complex_queries.jsonl"
    summary: str = "summary.json"


class SegmentationConfig(ConfigModel):
    enabled: bool = True


class Step1Config(ConfigModel):
    enabled: bool = True
    reject_rules: bool = True
    min_chain_tool_calls: int = 7
    min_chain_steps: int = 1
    min_unique_tools: int = 2


class Step2Config(ConfigModel):
    enabled: bool = True
    prompt_id: str = "complex_judge"

    @field_validator("prompt_id")
    @classmethod
    def validate_prompt_id(cls, value: str) -> str:
        prompt_id = value.strip()
        if not prompt_id:
            raise ValueError("prompt_id must be non-empty")
        from query_pipeline.prompts import resolve_prompt

        resolve_prompt(prompt_id)
        return prompt_id


class SessionStageConfig(ConfigModel):
    enabled: bool = True
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    step1: Step1Config = Field(default_factory=Step1Config)
    step2: Step2Config = Field(default_factory=Step2Config)


class LLMStageConfig(ConfigModel):
    enabled: bool = True
    base_url_env: str = "OPENAI_BASE_URL"
    model: str = "gpt-5.4-mini"
    api_key_env: str = "OPENAI_API_KEY"
    concurrency: int = 64
    max_retries: int = 5
    timeout_seconds: float = 90.0
    response_format: str = "json_object"
    cache: Path = Path("work/llm_cache.jsonl")


class PipelineConfig(ConfigModel):
    name: str = "question_pipeline"
    input: InputConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    work_dir: Path = Path("work")
    session_stage: SessionStageConfig = Field(default_factory=SessionStageConfig)
    llm_stage: LLMStageConfig
