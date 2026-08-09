from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ConfigModel(BaseModel):
    model_config = {"extra": "forbid"}


class InputConfig(ConfigModel):
    path: Path
    format: str = "auto"  # "auto" | "session" | "chat"

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        fmt = value.strip().lower()
        if fmt not in {"auto", "session", "chat"}:
            raise ValueError(f"invalid input.format: {value!r} (expected 'auto', 'session' or 'chat')")
        return fmt


class OutputConfig(ConfigModel):
    dir: Path = Path("outputs")
    complex_queries: str = "complex_queries.jsonl"
    summary: str = "summary.json"


class SegmentationConfig(ConfigModel):
    enabled: bool = True


class RuleGateConfig(ConfigModel):
    enabled: bool = True
    reject_rules: bool = True
    min_chain_tool_calls: int = 7
    min_chain_steps: int = 1
    min_unique_tools: int = 2


class JudgeConfig(ConfigModel):
    enabled: bool = True
    value_prompt: str = "value_gate"
    complexity_prompt: str = "complexity_gate"
    classify_complex_prompt: str = "classify_complex"
    classify_normal_prompt: str = "classify_normal"

    @field_validator("value_prompt", "complexity_prompt", "classify_complex_prompt", "classify_normal_prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        prompt_id = value.strip()
        if not prompt_id:
            raise ValueError("prompt id must be non-empty")
        from query_pipeline.prompts import resolve_prompt

        resolve_prompt(prompt_id)
        return prompt_id


class LLMConfig(ConfigModel):
    enabled: bool = True
    base_url_env: str = "OPENAI_BASE_URL"
    model: str = "gpt-5.4-mini"
    api_key_env: str = "OPENAI_API_KEY"
    concurrency: int = 64
    max_retries: int = 5
    timeout_seconds: float = 90.0
    response_format: str = "json_object"
    cache: Path | None = None  # None -> <work_dir>/logs/llm_cache.jsonl


class VerifyConfig(ConfigModel):
    enabled: bool = True
    prompt_id: str = "verify_complex"
    max_rounds_hard: int = 5
    max_rounds_normal: int = 2

    @field_validator("prompt_id")
    @classmethod
    def validate_prompt_id(cls, value: str) -> str:
        prompt_id = value.strip()
        if not prompt_id:
            raise ValueError("prompt_id must be non-empty")
        from query_pipeline.prompts import resolve_prompt

        resolve_prompt(prompt_id)
        return prompt_id

    @field_validator("max_rounds_hard", "max_rounds_normal")
    @classmethod
    def validate_max_rounds(cls, value: int) -> int:
        if value < 1:
            raise ValueError("verify rounds must be >= 1")
        return value


class DedupConfig(ConfigModel):
    enabled: bool = True
    threshold: float = 0.80
    entity_slot: bool = True

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return value


class TranslateConfig(ConfigModel):
    enabled: bool = True


class PostConfig(ConfigModel):
    enabled: bool = False
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)


class CheckpointConfig(ConfigModel):
    enabled: bool = True
    dir: Path | None = None  # None -> <work_dir>/logs/checkpoints


class DebugConfig(ConfigModel):
    dump_intermediates: bool = True


class PipelineConfig(ConfigModel):
    name: str = "question_pipeline"

    @property
    def cache_path(self) -> Path:
        """LLM 缓存实际路径（默认 <work_dir>/logs/llm_cache.jsonl）。"""
        return self.llm.cache or (self.work_dir or Path("logs")) / "logs" / "llm_cache.jsonl"

    @property
    def checkpoint_dir(self) -> Path:
        """阶段 checkpoint 实际目录（默认 <work_dir>/logs/checkpoints）。"""
        return self.checkpoint.dir or (self.work_dir or Path("logs")) / "logs" / "checkpoints"

    input: InputConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    work_dir: Path | None = None  # None -> output.dir（产物、日志、缓存同目录）
    stages: list[str] | None = None  # None -> pipeline default stage order
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    rule_gate: RuleGateConfig = Field(default_factory=RuleGateConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    llm: LLMConfig
    post: PostConfig = Field(default_factory=PostConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
