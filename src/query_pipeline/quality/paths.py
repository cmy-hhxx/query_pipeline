from __future__ import annotations

from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Find the repo root (contains pyproject.toml or .git), walking up from start."""
    path = (start or Path.cwd()).resolve()
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return path


def source_path(dataset: str, root: Path | None = None) -> Path:
    # 管线新默认输出名（含 hard + normal 两类样本）；date 与产物路径无关。
    return (root or project_root()) / "outputs" / dataset / "cleaned_queries.jsonl"


def qc_dir(dataset: str, date: str, root: Path | None = None) -> Path:
    # QC 产物按日期分目录（outputs/<dataset>/qc/<date>/）：同一数据集不同日期的
    # QC 运行互不覆盖（旧实现 date 是死参数，0806/0807 两次运行互相覆盖产物，
    # 且 overview 内 date 字段与文件名对不上）。
    return (root or project_root()) / "outputs" / dataset / "qc" / date


def llm_cache_path(dataset: str, root: Path | None = None) -> Path:
    # 复用管线 LLM 缓存（outputs/<dataset>/logs/llm_cache.jsonl）
    return (root or project_root()) / "outputs" / dataset / "logs" / "llm_cache.jsonl"
