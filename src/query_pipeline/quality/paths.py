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
    return (root or project_root()) / "outputs" / dataset / f"complex_queries_{date}.jsonl"


def qc_dir(dataset: str, date: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "work" / dataset / date / "qc"


def llm_cache_path(dataset: str, date: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "work" / dataset / date / "llm_cache.jsonl"
