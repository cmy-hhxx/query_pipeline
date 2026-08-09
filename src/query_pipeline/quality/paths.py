from __future__ import annotations

from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Find the repo root (contains pyproject.toml or .git), walking up from start."""
    path = (start or Path.cwd()).resolve()
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return path


def source_path(dataset: str, date: str, root: Path | None = None) -> Path:
    # 管线新默认输出名（含 hard + normal 两类样本）；date 用于 QC 产物目录。
    return (root or project_root()) / "outputs" / dataset / "cleaned_queries.jsonl"


def qc_dir(dataset: str, date: str, root: Path | None = None) -> Path:
    # QC 产物与管线产物同在一个数据集目录（outputs/<dataset>/qc/），date 仅用于产物内标注。
    return (root or project_root()) / "outputs" / dataset / "qc"


def llm_cache_path(dataset: str, date: str, root: Path | None = None) -> Path:
    # 复用管线 LLM 缓存（outputs/<dataset>/logs/llm_cache.jsonl）
    return (root or project_root()) / "outputs" / dataset / "logs" / "llm_cache.jsonl"
