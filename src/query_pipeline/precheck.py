"""数据预检：跑 LLM 阶段之前快速暴露输入文件的结构性问题。

纯规则、单遍流式扫描，不调 LLM。severity=critical 的问题让 run 在
preclean 之前中止（fail fast，避免浪费 LLM 资源）；warning 只报告不拦截。

典型场景：session/chat 输入整体缺 chain（上游导出不完整）时，rule_gate
只能回退 tool_count/tool_names 兜底，LLM 判定质量无保证——预检提前拦下。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from query_pipeline.adapters import CHAT, SESSION
from query_pipeline.io.sniff import sniff_format


@dataclass(frozen=True)
class PrecheckIssue:
    severity: str  # "critical" | "warning"
    code: str
    message: str
    count: int = 0


@dataclass
class PrecheckReport:
    path: Path
    format: str
    lines: int = 0
    bad_lines: int = 0
    records: int = 0                 # 可解析记录数（去重前）
    duplicate_records: int = 0
    empty_context_records: int = 0   # session：无 context；chat：缺 judge_data
    turns: int = 0                   # session：turn 总数；chat：记录数（每记录一个候选位）
    eligible_turns: int = 0
    turns_with_chain: int = 0
    issues: list[PrecheckIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "critical" for i in self.issues)

    @property
    def chain_coverage(self) -> float:
        return self.turns_with_chain / self.eligible_turns if self.eligible_turns else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "lines": self.lines,
            "bad_lines": self.bad_lines,
            "records": self.records,
            "duplicate_records": self.duplicate_records,
            "empty_context_records": self.empty_context_records,
            "turns": self.turns,
            "eligible_turns": self.eligible_turns,
            "turns_with_chain": self.turns_with_chain,
            "chain_coverage": round(self.chain_coverage, 4),
            "ok": self.ok,
            "issues": [
                {"severity": i.severity, "code": i.code, "message": i.message, "count": i.count}
                for i in self.issues
            ],
        }


def _session_turn_eligible(turn: dict[str, Any]) -> bool:
    """与 session/candidates.is_eligible 同口径：status/outcome 正常 + 有问有答。"""
    if turn.get("status") not in (None, "completed"):
        return False
    if turn.get("outcome") not in (None, "success"):
        return False
    if not str(turn.get("question") or "").strip():
        return False
    answer = turn.get("answer") or turn.get("answer_full")
    return bool(str(answer or "").strip())


def _chat_record_eligible(record: dict[str, Any]) -> bool:
    """与 adapters/chat.py + candidates.is_eligible 保持同一口径。"""
    jd = record.get("judge_data")
    if not isinstance(jd, dict):
        return False
    context = jd.get("context")
    if not isinstance(context, list) or any(not isinstance(turn, dict) for turn in context):
        return False
    raw = jd.get("input")
    if isinstance(raw, dict):
        text = raw.get("text")
    elif isinstance(raw, str):
        text = raw
    else:
        text = None
    if not str(text or record.get("question") or "").strip():
        return False
    answer = jd.get("text_answer") or jd.get("raw_answer")
    return bool(str(answer or "").strip())


def _chat_has_chain(record: dict[str, Any]) -> bool:
    jd = record.get("judge_data")
    if not isinstance(jd, dict):
        return False
    chain = jd.get("chain")
    return isinstance(chain, list) and len(chain) > 0


def precheck(
    path: str | Path,
    *,
    format: str = "auto",
    min_chain_coverage: float = 0.5,
    max_bad_line_ratio: float = 0.01,
) -> PrecheckReport:
    """单遍流式扫描输入文件，返回预检报告（纯规则，不调 LLM）。

    格式无法识别或混合时报 ValueError（与 sniff_format 一致）。
    """
    for name, value in (
        ("min_chain_coverage", min_chain_coverage),
        ("max_bad_line_ratio", max_bad_line_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    src = Path(path).resolve()
    fmt = format if format != "auto" else sniff_format(src)

    report = PrecheckReport(path=src, format=fmt)
    seen_ids: set[str] = set()

    with src.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            report.lines += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                report.bad_lines += 1
                continue
            if not isinstance(record, dict):
                report.bad_lines += 1
                continue
            report.records += 1

            if fmt == SESSION:
                tid = str(record.get("thread_id") or "")
                if tid:
                    if tid in seen_ids:
                        report.duplicate_records += 1
                    seen_ids.add(tid)
                context = record.get("context")
                if not isinstance(context, list) or not context:
                    report.empty_context_records += 1
                    continue
                for turn in context:
                    if not isinstance(turn, dict):
                        continue
                    report.turns += 1
                    if _session_turn_eligible(turn):
                        report.eligible_turns += 1
                        chain = turn.get("chain")
                        if isinstance(chain, list) and chain:
                            report.turns_with_chain += 1
            else:  # chat
                if not isinstance(record.get("judge_data"), dict):
                    report.empty_context_records += 1
                    continue
                jd = record["judge_data"]
                tid = str(jd.get("case_id") or record.get("trace_id") or "")
                if tid:
                    if tid in seen_ids:
                        report.duplicate_records += 1
                    seen_ids.add(tid)
                report.turns += 1
                if _chat_record_eligible(record):
                    report.eligible_turns += 1
                    if _chat_has_chain(record):
                        report.turns_with_chain += 1

    _finalize(report, min_chain_coverage=min_chain_coverage, max_bad_line_ratio=max_bad_line_ratio)
    return report


def _finalize(
    report: PrecheckReport,
    *,
    min_chain_coverage: float,
    max_bad_line_ratio: float,
) -> None:
    issues = report.issues

    if report.bad_lines:
        ratio = report.bad_lines / report.lines if report.lines else 1.0
        if ratio > max_bad_line_ratio:
            issues.append(
                PrecheckIssue(
                    "critical",
                    "bad_line_ratio_exceeded",
                    f"坏行 {report.bad_lines}/{report.lines}（{ratio:.2%}）超过阈值 {max_bad_line_ratio:.0%}，输入文件可能损坏",
                    report.bad_lines,
                )
            )
        else:
            issues.append(
                PrecheckIssue(
                    "warning",
                    "bad_lines",
                    f"{report.bad_lines} 个坏行将被跳过（preclean 会写入 bad_lines.jsonl）",
                    report.bad_lines,
                )
            )

    if report.duplicate_records:
        id_label = "thread_id" if report.format == SESSION else "case_id/trace_id"
        issues.append(
            PrecheckIssue(
                "warning",
                "duplicate_records",
                f"{report.duplicate_records} 个重复 {id_label}（preclean 将去重）",
                report.duplicate_records,
            )
        )

    if report.empty_context_records:
        label = "无 context" if report.format == SESSION else "缺 judge_data"
        issues.append(
            PrecheckIssue(
                "warning",
                "empty_context_records",
                f"{report.empty_context_records} 个记录{label}（preclean 将过滤）",
                report.empty_context_records,
            )
        )

    if report.eligible_turns == 0:
        issues.append(
            PrecheckIssue(
                "critical",
                "no_eligible_turns",
                "0 个合格 turn（需 question+answer 且 status/outcome 正常）——输入结构可能不对（如整批 run 失败或字段缺失）",
            )
        )
        return

    coverage = report.chain_coverage
    if coverage < min_chain_coverage:
        issues.append(
            PrecheckIssue(
                "critical",
                "missing_chain",
                f"chain 覆盖率 {coverage:.1%}（{report.turns_with_chain}/{report.eligible_turns}）低于阈值 "
                f"{min_chain_coverage:.0%}——缺 chain 时 rule_gate 只能回退 tool_count 兜底，LLM 判定质量无保证；"
                f"如确属 end2end 输入请显式放宽（--allow-no-chain）",
            )
        )


def render(report: PrecheckReport) -> str:
    lines = [
        f"precheck {report.path} [{report.format}]",
        f"  lines={report.lines:,}  bad={report.bad_lines:,}",
        f"  records={report.records:,}  duplicates={report.duplicate_records:,}  empty_context={report.empty_context_records:,}",
        f"  eligible turns={report.eligible_turns:,}/{report.turns:,}  with_chain={report.turns_with_chain:,}  coverage={report.chain_coverage:.1%}",
    ]
    for issue in report.issues:
        lines.append(f"  [{issue.severity.upper():8}] {issue.code}: {issue.message}")
    if report.ok:
        lines.append("verdict: OK — 数据可用，可继续运行")
    else:
        critical = sum(1 for i in report.issues if i.severity == "critical")
        lines.append(f"verdict: FAIL — {critical} 个严重问题，运行将中止")
    return "\n".join(lines)
