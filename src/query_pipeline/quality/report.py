from __future__ import annotations

from typing import Any

_STATUS_LABELS = {"pass": "正常", "fail": "FAIL（规则）", "needs_review": "待复核（LLM）"}


def _row_count(status_counts: dict[str, int], total: int, status: str) -> str:
    count = status_counts.get(status, 0)
    rate = f"（{count / total:.1%}）" if total else ""
    return f"{count}{rate}"


def render_markdown(overview: dict[str, Any], results: list[dict[str, Any]]) -> str:
    total = overview["total"]
    status_counts = overview["status_counts"]
    sample = overview["sample"]

    lines: list[str] = [
        "# 质检报告",
        "",
        f"- 数据源：`{overview['source']}`",
        f"- 记录数：{total}（跳过坏行 {overview['skipped_bad_lines']}）",
        f"- 生成时间：{overview['generated_at']}",
        "",
        "## 状态概览",
        "",
        "| 状态 | 数量 |",
        "| --- | --- |",
    ]
    for status in ("pass", "fail", "needs_review"):
        lines.append(f"| {_STATUS_LABELS.get(status, status)} | {_row_count(status_counts, total, status)} |")

    if sample["count"]:
        lines += [
            "",
            "## LLM 抽检",
            "",
            f"- 抽样：{sample['count']} 条（比例 {sample['ratio']}，seed {sample['seed']}）",
            f"- 问句质量：high {sample['question_quality_high']} / low {sample['question_quality_low']}",
            f"- 标签归属：正确 {sample['label_ok']} / 不符 {sample['label_not_ok']}",
            f"- judge 错误：{sample['judge_errors']}",
        ]

    lines += [
        "",
        "## 逐条规则命中",
        "",
        "| 规则 | 通过 | 失败 |",
        "| --- | --- | --- |",
    ]
    for name, hits in overview["rule_hits"].items():
        lines.append(f"| {name} | {hits['pass']} | {hits['fail']} |")

    lines += [
        "",
        "## 数据集级规则",
        "",
        "| 规则 | 结论 | 说明 |",
        "| --- | --- | --- |",
    ]
    for rule in overview["dataset_rules"]:
        lines.append(f"| {rule['rule']} | {'通过' if rule['ok'] else '关注'} | {rule['detail']} |")

    flagged = overview["flagged"]
    lines += [
        "",
        "## 待关注记录",
        "",
        f"共 {len(flagged)} 条（显示前 {len(flagged)} 条）：",
        "",
        "| trace_id | 状态 | 类别 | 问句 | 原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in flagged:
        question = item["question"].replace("|", "\\|") or "-"
        reason = item["reason"].replace("|", "\\|") or "-"
        status = _STATUS_LABELS.get(item["status"], item["status"])
        lines.append(
            f"| {item['trace_id']} | {status} | {item['category']} | {question} | {reason} |"
        )
    lines.append("")
    return "\n".join(lines)


def print_terminal(overview: dict[str, Any]) -> None:
    total = overview["total"]
    status_counts = overview["status_counts"]
    sample = overview["sample"]
    print(f"QC 完成：{overview['source']}（{total} 条，跳过坏行 {overview['skipped_bad_lines']}）")
    print(
        "状态："
        f"正常 {_row_count(status_counts, total, 'pass')}，"
        f"FAIL {_row_count(status_counts, total, 'fail')}，"
        f"待复核 {_row_count(status_counts, total, 'needs_review')}"
    )
    if sample["count"]:
        print(
            f"LLM 抽检 {sample['count']} 条："
            f"低质 {sample['question_quality_low']}，"
            f"标签不符 {sample['label_not_ok']}，"
            f"judge 错误 {sample['judge_errors']}"
        )
    for rule in overview["dataset_rules"]:
        mark = "✓" if rule["ok"] else "!"
        print(f"数据集级 [{mark}] {rule['rule']}：{rule['detail']}")
