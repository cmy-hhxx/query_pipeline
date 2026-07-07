from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    path: Path
    text_field: str = "question"


class OutputConfig(BaseModel):
    dir: Path = Path("outputs")
    labeled: str = "labeled.jsonl"
    rejected: str = "rejected.jsonl"
    skipped: str = "skipped_low_score.jsonl"
    summary: str = "run_summary.json"


class RulesConfig(BaseModel):
    min_length: int = 6
    finance_semantic: bool = True
    cleaning_version: str = "finance_query_rules_v1"


class MinHashConfig(BaseModel):
    enabled: bool = True
    method: str = "minhash_char_3gram"
    num_perm: int = 128
    threshold: float = 0.85
    normalization: Literal["none", "theme"] | None = None


class DedupConfig(BaseModel):
    exact: bool = True
    minhash: MinHashConfig = Field(default_factory=MinHashConfig)


class ClassifyLLMConfig(BaseModel):
    enabled: bool = True
    prompt: Path = Path("configs/prompts/classify_complex.md")
    min_complexity_score: int = 3
    min_question_length: int = 18


class DifficultyLLMConfig(BaseModel):
    enabled: bool = True
    prompt: Path = Path("configs/prompts/label_difficulty.md")


class LLMConfig(BaseModel):
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    api_key_env: str = "DEEPSEEK_API_KEY"
    concurrency: int = 64
    max_retries: int = 5
    timeout_seconds: float = 90.0
    response_format: str = "json_object"
    cache: Path = Path("work/llm_cache.jsonl")
    classify: ClassifyLLMConfig = Field(default_factory=ClassifyLLMConfig)
    difficulty: DifficultyLLMConfig = Field(default_factory=DifficultyLLMConfig)


class PipelineConfig(BaseModel):
    name: str = "question_pipeline"
    pipeline_version: str = "v1"
    input: InputConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    work_dir: Path = Path("work")
    rules: RulesConfig = Field(default_factory=RulesConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    steps: list[str] = Field(
        default_factory=lambda: [
            "clean",
            "dedup_exact",
            "dedup_minhash",
            "llm_classify",
            "llm_difficulty",
        ]
    )
