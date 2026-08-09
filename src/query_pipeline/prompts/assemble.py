"""Runtime prompt assembly from templates/*.md.

Classification and verify prompts are assembled at runtime so taxonomy
changes (categories, few-shot examples, bad cases) are data edits, not code
edits. Parsers are pure functions over markdown text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from query_pipeline.taxonomy import templates_dir, load_taxonomy

_COMPLEX_HEADER = re.compile(r"^###\s+(\d{2})\s+")
_EXAMPLE_LIST = re.compile(r"^(?:所有例子|典型例子)：\s*$")
_ITEM_NUM = re.compile(r"^\d+\.\s*(.+)$")
_ITEM_BULLET = re.compile(r"^[-•]\s*(.+)$")

_NORMAL_HEADER = re.compile(r"^##\s+(\d{2})-(.+?)\s*\|\s*(.+)$")
_SECTION = re.compile(r"^(定义|适用场景|排除场景|边界规则|易混类别)：\s*(.*)$")

_BAD_CASE = re.compile(r"^(?P<case>.+?)(?:\s*--\s*.+)?$")
_ANNOTATION = re.compile(r"^(?:以及|看下来|\d+、)")


@dataclass(frozen=True)
class ComplexCategorySpec:
    id: str
    definition: str
    examples: tuple[str, ...]


@dataclass(frozen=True)
class NormalCategorySpec:
    id: str
    slug: str
    name: str
    sections: dict[str, str]  # ordered: 定义/适用场景/排除场景/边界规则/易混类别


def _read(name: str) -> str:
    return (templates_dir() / name).read_text(encoding="utf-8")


def parse_complex_few_shot(text: str) -> dict[str, ComplexCategorySpec]:
    specs: dict[str, ComplexCategorySpec] = {}
    current: str | None = None
    definition: list[str] = []
    examples: list[str] = []
    in_examples = False
    for raw in text.splitlines():
        line = raw.strip()
        match = _COMPLEX_HEADER.match(line)
        if match:
            if current is not None:
                specs[current] = ComplexCategorySpec(
                    current, " ".join(definition).strip(), tuple(examples)
                )
            current = match.group(1)
            definition, examples, in_examples = [], [], False
            continue
        if current is None:
            continue
        if _EXAMPLE_LIST.match(line):
            in_examples = True
            continue
        if in_examples:
            num = _ITEM_NUM.match(line)
            bullet = _ITEM_BULLET.match(line)
            if num:
                examples.append(num.group(1).strip())
                continue
            if bullet:
                examples.append(bullet.group(1).strip())
                continue
            if line:  # 示例段里的非列表行：结束示例收集
                in_examples = False
                definition.append(line)
            continue
        if line:
            definition.append(line)
    if current is not None:
        specs[current] = ComplexCategorySpec(current, " ".join(definition).strip(), tuple(examples))
    return specs


def parse_normal_few_shot(text: str) -> dict[str, NormalCategorySpec]:
    specs: dict[str, NormalCategorySpec] = {}
    current: str | None = None
    current_slug = ""
    current_name = ""
    section: str | None = None
    sections: dict[str, str] = {}
    buffer: list[str] = []

    def flush() -> None:
        if section:
            sections[section] = "\n".join(buffer).strip()

    def save() -> None:
        flush()
        specs[current] = NormalCategorySpec(current, current_slug, current_name, dict(sections))

    for raw in text.splitlines():
        line = raw.strip()
        match = _NORMAL_HEADER.match(line)
        if match:
            if current is not None:
                save()
            current = match.group(1)
            current_slug = match.group(2).strip()
            current_name = match.group(3).strip()
            section, sections, buffer = None, {}, []
            continue
        if current is None:
            continue
        sec = _SECTION.match(line)
        if sec:
            flush()
            section = sec.group(1)
            buffer = [sec.group(2)] if sec.group(2) else []
            continue
        buffer.append(line)
    if current is not None:
        save()
    return specs


def parse_bad_cases(text: str) -> tuple[str, ...]:
    """Extract negative examples from bad_cases_for_complex.md.

    ``<example> -- <reason>`` lines and bare example lines both count;
    annotation notes (以及/看下来/N、 prefixed) are skipped.
    """
    cases: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _ANNOTATION.match(line):
            continue
        match = _BAD_CASE.match(line)
        if match:
            case = match.group("case").strip()
            if case:
                cases.append(case)
    return tuple(cases)


def build_complex_classify_prompt(*, max_examples: int = 1000) -> str:
    tax = load_taxonomy()
    specs = parse_complex_few_shot(_read("complex_few_shot.md"))
    blocks: list[str] = []
    for cid, cat in tax.complex.items():
        spec = specs.get(cid)
        lines = [f"{cid} {cat.name}（{cat.path}）"]
        if spec is not None and spec.definition:
            lines.append(f"特征：{spec.definition}")
        if spec is not None and spec.examples:
            lines.append("示例：")
            lines.extend(f"- {ex}" for ex in spec.examples[:max_examples])
        blocks.append("\n".join(lines))
    taxonomy_block = "\n\n".join(blocks)
    return f"""你是一个金融问句分类器。给定一个**已判定为复杂金融问句**的问句（附前文问题列表，仅用于消解指代），从下列 9 类中选择**唯一**类别。

要求：
- 只按问句意图归类，不按关键词机械匹配。
- 若多个类别都沾边，选择最决定答案正确性的那一类。
- 只输出严格 JSON，不要 Markdown：{{"category_id": "01", "reason": "中文短句，说明归类理由"}}
- category_id 必须是下列 9 类之一。

{taxonomy_block}
""".strip()


def build_normal_classify_prompt() -> str:
    tax = load_taxonomy()
    specs = parse_normal_few_shot(_read("normal_few_shot.md"))
    blocks: list[str] = []
    for cid, cat in tax.normal.items():
        spec = specs.get(cid)
        lines = [f"{cid} {cat.name}（{cat.path}）"]
        if spec is not None:
            for key, value in spec.sections.items():
                if value:
                    lines.append(f"{key}：\n{value}")
        blocks.append("\n".join(lines))
    taxonomy_block = "\n\n".join(blocks)
    return f"""你是一个金融问句分类器。给定一个**有价值但非复杂**的金融问句（附前文问题列表，仅用于消解指代），从下列 16 类中选择**唯一**类别。

要求：
- 只按问句意图归类，不按关键词机械匹配。
- 若多个类别都沾边，选择最决定答案正确性的那一类。
- 只输出严格 JSON，不要 Markdown：{{"category_id": "01", "reason": "中文短句，说明归类理由"}}
- category_id 必须是下列 16 类之一。

{taxonomy_block}
""".strip()


def build_verify_prompt(base_prompt: str) -> str:
    """Append confirmed negative examples to a verify prompt."""
    cases = parse_bad_cases(_read("bad_cases_for_complex.md"))
    if not cases:
        return base_prompt
    block = "\n".join(f"- {case}" for case in cases)
    return (
        base_prompt
        + "\n\n以下问句已被确认为不复杂（负例，判定时参照，不要与之冲突）：\n"
        + block
    )
