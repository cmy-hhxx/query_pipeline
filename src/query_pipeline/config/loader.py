from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

from query_pipeline.config.models import PipelineConfig


def load_pipeline_config(path: str | Path = "config.yaml") -> PipelineConfig:
    config_path = Path(path).resolve()
    load_dotenv(config_path.parent / ".env", override=False)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = PipelineConfig.model_validate(data)
    return _resolve_paths(cfg, config_path.parent)


def _resolve_paths(cfg: PipelineConfig, base: Path) -> PipelineConfig:
    if not cfg.input.path.is_absolute():
        cfg.input.path = (base / cfg.input.path).resolve()
    if not cfg.work_dir.is_absolute():
        cfg.work_dir = (base / cfg.work_dir).resolve()
    if not cfg.output.dir.is_absolute():
        cfg.output.dir = (base / cfg.output.dir).resolve()
    if not cfg.llm_stage.cache.is_absolute():
        cfg.llm_stage.cache = (base / cfg.llm_stage.cache).resolve()
    return cfg
