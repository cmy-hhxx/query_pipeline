"""Data-driven taxonomy — single source of truth is templates/categories.md.

The markdown file defines both taxonomies in one table:
- complex 9:  `01    complex-topic/01-data-metrics-calculation    复杂取数计算`
- normal 16:  `01-event-and-concept-stock-selection    事件与概念选股`

The slug column doubles as the output `category` value: complex rows carry the
`complex-topic/` prefix (mirrors fin_bench directory names, a downstream join
key — 07's slug is a historical artifact and must never be "fixed"), normal
rows stay `{id}-{slug}`. Ids collide between the two sets (01-09 vs 01-16);
difficulty + prefix disambiguate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

COMPLEX_PREFIX = "complex-topic/"
HARD = "hard"
NORMAL = "normal"


@dataclass(frozen=True)
class Category:
    """One taxonomy entry."""

    id: str          # "01"
    slug: str        # "data-metrics-calculation"
    name: str        # CN name, e.g. "复杂取数计算"
    difficulty: str  # "hard" | "normal"

    @property
    def path(self) -> str:
        """Output `category` field value."""
        if self.difficulty == HARD:
            return f"{COMPLEX_PREFIX}{self.id}-{self.slug}"
        return f"{self.id}-{self.slug}"

    @property
    def label(self) -> str:
        return self.path


@dataclass(frozen=True)
class Taxonomy:
    complex: dict[str, Category]
    normal: dict[str, Category]

    def all(self) -> tuple[Category, ...]:
        return tuple(self.complex.values()) + tuple(self.normal.values())

    def get(self, difficulty: str, category_id: str) -> Category:
        table = self.complex if difficulty == HARD else self.normal
        try:
            return table[category_id]
        except KeyError:
            raise KeyError(
                f"unknown category_id {category_id!r} for difficulty {difficulty!r}"
            ) from None


def _parse_line(line: str) -> Category:
    parts = line.split()
    if len(parts) == 3 and parts[1].startswith(COMPLEX_PREFIX):
        cid, path, name = parts
        slug = path[len(COMPLEX_PREFIX) + len(cid) + 1 :]
        return Category(id=cid, slug=slug, name=name, difficulty=HARD)
    if len(parts) == 2 and "-" in parts[0]:
        path, name = parts
        cid, _, slug = path.partition("-")
        return Category(id=cid, slug=slug, name=name, difficulty=NORMAL)
    raise ValueError(f"unparseable category line: {line!r}")


def parse_categories(text: str) -> Taxonomy:
    complex_cats: dict[str, Category] = {}
    normal_cats: dict[str, Category] = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cat = _parse_line(line)
        table = complex_cats if cat.difficulty == HARD else normal_cats
        if cat.id in table:
            raise ValueError(f"duplicate category id {cat.id!r} at line {line_no}")
        table[cat.id] = cat
    if not complex_cats or not normal_cats:
        raise ValueError("categories.md must define both complex and normal categories")
    return Taxonomy(complex=complex_cats, normal=normal_cats)


def templates_dir() -> Path:
    """Locate the templates/ directory.

    Search order:
    1. QUERY_PIPELINE_TEMPLATES env override;
    2. <repo-or-site-packages>/templates — source tree layout and wheel
       artifacts (hatch `artifacts = ["../templates"]`) both land here;
    3. package-data fallback via importlib.resources (templates shipped
       inside the wheel package).
    """
    env = os.environ.get("QUERY_PIPELINE_TEMPLATES")
    if env:
        return Path(env)
    # templates 是包内数据（src/query_pipeline/templates/），随 wheel 自动发布，
    # 源码树与安装后的定位一致。
    return Path(__file__).resolve().parent / "templates"


_TAXONOMY: Taxonomy | None = None


def load_taxonomy() -> Taxonomy:
    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY = parse_categories(
            (templates_dir() / "categories.md").read_text(encoding="utf-8")
        )
    return _TAXONOMY
