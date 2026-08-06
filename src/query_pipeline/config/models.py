from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ConfigModel(BaseModel):
    model_config = {"extra": "forbid"}


class InputConfig(ConfigModel):
    path: Path
    text_path: str = "question"

    @field_validator("text_path")
    @classmethod
    def validate_text_path(cls, value: str) -> str:
        parts = value.split(".")
        if not value or any(part == "" for part in parts):
            raise ValueError("text_path must be a non-empty dot path")
        return value


class OutputConfig(ConfigModel):
    dir: Path = Path("outputs")
    accepted: str = "accepted.jsonl"
    non_complex: str = "non_complex.jsonl"
    rejected: str = "rejected.jsonl"
    skipped: str = "skipped.jsonl"
    summary: str = "summary.json"


class CleanRulesConfig(ConfigModel):
    enabled: bool = True
    min_length: int = 6
    finance_semantic: bool = True


class ExactDedupConfig(ConfigModel):
    enabled: bool = True


class MinHashConfig(ConfigModel):
    enabled: bool = True
    threshold: float = 0.85


class ComplexityGateConfig(ConfigModel):
    enabled: bool = True
    min_score: int = 3
    min_text_length: int = 18


class RulesStageConfig(ConfigModel):
    enabled: bool = True
    clean: CleanRulesConfig = Field(default_factory=CleanRulesConfig)
    exact_dedup: ExactDedupConfig = Field(default_factory=ExactDedupConfig)
    minhash: MinHashConfig = Field(default_factory=MinHashConfig)
    complexity_gate: ComplexityGateConfig = Field(default_factory=ComplexityGateConfig)


class LLMStageConfig(ConfigModel):
    enabled: bool = True
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    api_key_env: str = "DEEPSEEK_API_KEY"
    concurrency: int = 64
    max_retries: int = 5
    timeout_seconds: float = 90.0
    response_format: str = "json_object"
    cache: Path = Path("work/llm_cache.jsonl")
    prompt_id: str = "core_label"

    @field_validator("prompt_id")
    @classmethod
    def validate_prompt_id(cls, value: str) -> str:
        prompt_id = value.strip()
        if not prompt_id:
            raise ValueError("prompt_id must be non-empty")
        from query_pipeline.prompts import resolve_prompt

        resolve_prompt(prompt_id)
        return prompt_id


class PipelineConfig(ConfigModel):
    name: str = "question_pipeline"
    input: InputConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    work_dir: Path = Path("work")
    rules_stage: RulesStageConfig = Field(default_factory=RulesStageConfig)
    llm_stage: LLMStageConfig
