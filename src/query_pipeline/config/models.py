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
    cleaned_queries: str = "cleaned_queries.jsonl"  # 汇总（hard + normal 两类）
    complex_queries: str = "complex_queries.jsonl"  # 仅复杂问句
    normal_queries: str = "normal_queries.jsonl"    # 仅普通问句
    summary: str = "summary.json"


class SegmentationConfig(ConfigModel):
    enabled: bool = True


class PrecheckConfig(ConfigModel):
    """数据预检：run 在 LLM 阶段之前快速扫描输入，严重问题即中止（fail fast）。"""
    enabled: bool = True
    # 合格 turn 的 chain 覆盖率低于该比例 → critical（session/chat 均适用）
    min_chain_coverage: float = 0.5
    # 坏行占比超过该比例 → critical（以下仅 warning）
    max_bad_line_ratio: float = 0.01

    @field_validator("min_chain_coverage", "max_bad_line_ratio")
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("precheck ratio must be in [0, 1]")
        return value


class RuleGateConfig(ConfigModel):
    enabled: bool = True
    reject_rules: bool = True
    # None = 未显式设置：rule_gate 阶段按嗅探到的实际输入格式补齐默认
    # （session 7/1/2，chat 3/1/2——chat 工具调用分布平坦，>=7 次仅覆盖 ~1%）。
    min_chain_tool_calls: int | None = None
    min_chain_steps: int = 1
    min_unique_tools: int | None = None


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
    cache: Path | None = None  # None -> <work_dir>/runtime/cache/llm_cache.jsonl


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
    dir: Path | None = None  # None -> <work_dir>/runtime/checkpoints


class DebugConfig(ConfigModel):
    dump_intermediates: bool = True


class LoggingConfig(ConfigModel):
    dir: Path | None = None  # None -> <output.dir>/logs
    batch_id: str | None = None
    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in {"INFO", "DEBUG"}:
            raise ValueError("logging.level must be 'INFO' or 'DEBUG'")
        return level

    @field_validator("batch_id")
    @classmethod
    def validate_batch_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from query_pipeline.logging_setup import validate_batch_id

        return validate_batch_id(value)


class PipelineConfig(ConfigModel):
    name: str = "question_pipeline"

    @property
    def cache_path(self) -> Path:
        """LLM cache path (default: <work_dir>/runtime/cache/llm_cache.jsonl)."""
        return self.llm.cache or (self.work_dir or self.output.dir) / "runtime" / "cache" / "llm_cache.jsonl"

    @property
    def checkpoint_dir(self) -> Path:
        """Stage checkpoint path (default: <work_dir>/runtime/checkpoints)."""
        return self.checkpoint.dir or (self.work_dir or self.output.dir) / "runtime" / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.logging.dir or self.output.dir / "logs"

    input: InputConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    work_dir: Path | None = None  # None -> output.dir（产物、日志、缓存同目录）
    stages: list[str] | None = None  # None -> pipeline default stage order
    precheck: PrecheckConfig = Field(default_factory=PrecheckConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    rule_gate: RuleGateConfig = Field(default_factory=RuleGateConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    llm: LLMConfig
    post: PostConfig = Field(default_factory=PostConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
