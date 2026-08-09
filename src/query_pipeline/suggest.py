"""rule_gate 门槛推荐：纯规则扫描，不调 LLM。

对输入数据穷举常用门槛组合，统计每种组合过滤后的候选数（即进入 LLM
判定的问句量），按候选数从低到高返回推荐组合，让用户按目标数据量选参数。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from query_pipeline.adapters import adapt_record
from query_pipeline.io.jsonl import read_jsonl_with_bad_lines
from query_pipeline.io.sniff import preclean_records, sniff_format
from query_pipeline.rules.reject import generic_reject_reason
from query_pipeline.session.candidates import (
    chain_steps,
    chain_tool_calls,
    is_eligible,
    unique_tools,
)

logger = logging.getLogger(__name__)

# 扫描网格：工具调用数 / 工具种数 / reject 规则开关（min_chain_steps 固定 1）。
TOOL_CALLS_GRID = (0, 1, 2, 3, 4, 5, 7, 10, 15)
UNIQUE_TOOLS_GRID = (1, 2, 3, 5)
REJECT_GRID = (True, False)


@dataclass(frozen=True)
class GateSuggestion:
    min_chain_tool_calls: int
    min_unique_tools: int
    reject_rules: bool
    candidates: int
    total_turns: int
    ratio: float  # candidates / total_turns
    is_default: bool = False  # 与当前格式的默认门槛一致

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_chain_tool_calls": self.min_chain_tool_calls,
            "min_unique_tools": self.min_unique_tools,
            "reject_rules": self.reject_rules,
            "candidates": self.candidates,
            "total_turns": self.total_turns,
            "ratio": round(self.ratio, 4),
            "is_default": self.is_default,
        }


@dataclass
class _TurnFeatures:
    eligible: bool
    reject: str | None
    calls: int
    steps: int
    tools: int


def _load_sessions(input_path: Path, fmt: str) -> tuple[list[Any], int]:
    """读入 → 预清洗 → adapt；返回 (sessions, 候选基数)。

    候选基数 = session 格式的 turn 总数；chat 格式的记录数（每记录只有
    末轮是候选位）。占比 = 候选数 / 候选基数。
    """
    raw_records, bad_count = read_jsonl_with_bad_lines(input_path, Path("/dev/null"))
    records, _dup, _empty = preclean_records(raw_records, fmt)
    sessions: list[Any] = []
    for record in records:
        try:
            sessions.append(adapt_record(record, fmt))
        except ValueError as exc:
            logger.warning("adapt skipped a record: %s", str(exc)[:120])
    base = sum(
        1 if s.candidate_mode == "last_only" else len(s.turns) for s in sessions
    )
    return sessions, base


def suggest_gates(
    input_path: str | Path,
    *,
    format: str = "auto",
    top: int = 10,
) -> list[dict[str, Any]]:
    """Scan rule_gate thresholds; return ``top`` combos sampled evenly across
    the candidate-count spectrum, ordered from strictest to loosest.

    Every combo in the grid is evaluated (rules only, no LLM), duplicates with
    the same candidate count collapse, then ``top`` combos are picked at even
    intervals across the sorted list so both the aggressive and the loose end
    are visible.
    """
    src = Path(input_path).resolve()
    fmt = format if format != "auto" else sniff_format(src)
    sessions, total_turns = _load_sessions(src, fmt)

    # 预计算每个 turn 的特征（与门槛无关的部分只算一次）
    features: dict[str, list[_TurnFeatures]] = {}
    for session in sessions:
        feats: list[_TurnFeatures] = []
        for idx, turn in enumerate(session.turns):
            feats.append(
                _TurnFeatures(
                    eligible=is_eligible(turn),
                    reject=generic_reject_reason(turn.question),
                    calls=chain_tool_calls(turn),
                    steps=chain_steps(turn),
                    tools=unique_tools(turn),
                )
            )
        features[session.thread_id] = feats

    def count(calls_min: int, tools_min: int, reject_on: bool) -> int:
        n = 0
        for session in sessions:
            feats = features[session.thread_id]
            if session.candidate_mode == "last_only":
                # chat：只有末轮是候选，同样吃全部门槛
                idx = len(feats) - 1
                if idx < 0:
                    continue
                f = feats[idx]
                if (
                    f.eligible
                    and (not reject_on or not f.reject)
                    and f.calls >= calls_min
                    and f.steps >= 1
                    and f.tools >= tools_min
                ):
                    n += 1
                continue
            for f in feats:
                if (
                    f.eligible
                    and (not reject_on or not f.reject)
                    and f.calls >= calls_min
                    and f.steps >= 1
                    and f.tools >= tools_min
                ):
                    n += 1
        return n

    default_calls, default_tools = (7, 2) if fmt == "session" else (3, 2)
    seen: set[int] = set()
    suggestions: list[GateSuggestion] = []
    for calls_min, tools_min, reject_on in product(
        TOOL_CALLS_GRID, UNIQUE_TOOLS_GRID, REJECT_GRID
    ):
        candidates = count(calls_min, tools_min, reject_on)
        is_default = (
            calls_min == default_calls
            and tools_min == default_tools
            and reject_on
        )
        if candidates in seen and not is_default:
            continue  # 相同数据量的组合只保留一个（默认组合优先保留）
        seen.add(candidates)
        suggestions.append(
            GateSuggestion(
                min_chain_tool_calls=calls_min,
                min_unique_tools=tools_min,
                reject_rules=reject_on,
                candidates=candidates,
                total_turns=total_turns,
                ratio=candidates / total_turns if total_turns else 0.0,
                is_default=is_default,
            )
        )

    suggestions.sort(key=lambda s: s.candidates)

    if len(suggestions) <= top:
        return [s.as_dict() for s in suggestions]
    # 全谱等距采样：从最严到最松均匀取 top 个，两端都可见。
    step = (len(suggestions) - 1) / (top - 1)
    picked = [suggestions[round(i * step)] for i in range(top)]
    # 默认组合若不在采样点，插入并保持升序
    default = next((s for s in picked if s.is_default), None)
    if default is None:
        default = next(s for s in suggestions if s.is_default)
        picked.append(default)
        picked.sort(key=lambda s: s.candidates)
    return [s.as_dict() for s in picked]


def render_suggestions(
    input_path: str | Path,
    suggestions: list[dict[str, Any]],
    *,
    fmt: str,
) -> str:
    """Human-readable table for the CLI."""
    total = suggestions[0]["total_turns"]
    lines = [
        f"=== rule_gate 参数推荐：{Path(input_path).name}（{fmt}）===",
        f"候选基数 {total}，按过滤后候选数从低到高展示 {len(suggestions)} 个组合（全谱等距采样）",
        "",
        f"  {'#':<4}{'候选数':<9}{'占比':<8}{'min_tool_calls':<16}{'min_unique_tools':<18}reject_rules",
    ]
    for i, s in enumerate(suggestions, start=1):
        mark = "  ※默认" if s.get("is_default") else ""
        lines.append(
            f"  {i:<4}{s['candidates']:<9}{s['ratio'] * 100:>6.1f}%  "
            f"{s['min_chain_tool_calls']:<16}{s['min_unique_tools']:<18}{str(s['reject_rules']).lower()}{mark}"
        )
    lines.append("")
    lines.append("用法示例（选一个组合跑管线）：")
    lines.append(
        f"  uv run query-pipeline run {input_path} "
        "--min-tool-calls 5 --min-unique-tools 3"
    )
    return "\n".join(lines)
