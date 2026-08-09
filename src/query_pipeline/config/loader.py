from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

from query_pipeline.config.models import PipelineConfig


def load_pipeline_config(path: str | Path = "config.yaml") -> PipelineConfig:
    config_path = Path(path).resolve()
    root = _project_root(config_path)
    load_dotenv(root / ".env", override=False)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = PipelineConfig.model_validate(data)
    return _resolve_paths(cfg, root)


def _project_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parent.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return config_path.parent


def _resolve_paths(cfg: PipelineConfig, base: Path) -> PipelineConfig:
    if not cfg.input.path.is_absolute():
        cfg.input.path = (base / cfg.input.path).resolve()
    if not cfg.output.dir.is_absolute():
        cfg.output.dir = (base / cfg.output.dir).resolve()
    # 默认化：work_dir 并入输出目录；缓存/断点统一在 logs/ 下。
    if cfg.work_dir is None:
        cfg.work_dir = cfg.output.dir
    elif not cfg.work_dir.is_absolute():
        cfg.work_dir = (base / cfg.work_dir).resolve()
    if cfg.checkpoint.dir is None:
        cfg.checkpoint.dir = cfg.work_dir / "logs" / "checkpoints"
    elif not cfg.checkpoint.dir.is_absolute():
        cfg.checkpoint.dir = (base / cfg.checkpoint.dir).resolve()
    if cfg.llm.cache is None:
        cfg.llm.cache = cfg.work_dir / "logs" / "llm_cache.jsonl"
    elif not cfg.llm.cache.is_absolute():
        cfg.llm.cache = (base / cfg.llm.cache).resolve()
    return cfg
