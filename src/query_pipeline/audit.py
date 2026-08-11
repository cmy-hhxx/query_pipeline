"""Strict audit of complex-queries output.

Runs an independent strict LLM review over every row of a complex_queries
jsonl and reports rows judged non-complex plus the overall ratio. Serves as
the precision gate for the pipeline: any new simple-question type the
LLM let through shows up here and can be fed back into the verify prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from query_pipeline.config.models import LLMConfig
from query_pipeline.io.jsonl import read_jsonl_with_bad_lines
from query_pipeline.llm.client import LLMClient
from query_pipeline.models.session import VerifyResult
from query_pipeline.prompts.assemble import load_complex_quality_policy

logger = logging.getLogger(__name__)

AUDIT_PROMPT = (
    """你是独立的 complex 输出精度审计员。只依据输入 question，使用附带的统一政策输出
三路结构。不得读取答案、chain、已有标签或前文。自然、非模板化且包含至少 3 个彼此
独立实质条件的量化筛选可以是 complex；共享短语本身不是模板证据。边界不清时 normal，
只有 eval_template 或 embedded_prompt 可 reject。

只输出严格 JSON：
{"route":"complex|normal|reject","complex_features":["受控枚举"],"exclusion_reasons":["受控枚举"],"evidence":[{"criterion":"受控枚举","quote":"question 原文"}],"confidence":"low|medium|high","reason":"一句话理由"}
""".strip()
    + "\n\n---\n\n"
    + load_complex_quality_policy()
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    # 与管线同一套读取语义（read_jsonl_with_bad_lines）：坏 JSON / 非对象行落盘
    # bad_lines 并跳过，绝不让一行坏 JSON 崩溃整个 audit 命令（旧实现直接
    # json.loads，坏行抛异常炸掉 audit，与管线 bad_lines 容忍哲学不一致）。
    bad_path = path.with_name(path.name + ".bad_lines.jsonl")
    rows, skipped = read_jsonl_with_bad_lines(path, bad_path)
    if skipped:
        logger.warning("audit: skipped %d bad line(s) → %s", skipped, bad_path)
    return rows


async def audit_rows(
    rows: list[Any],
    *,
    model: str = "gpt-5.4-mini",
    concurrency: int = 64,
) -> list[dict[str, Any]]:
    client = LLMClient(LLMConfig(model=model, concurrency=concurrency))
    # 限流由 LLMClient 进程级 semaphore 承担，此处不再叠加第二层。
    results: list[dict[str, Any]] = []

    async def check(row: Any) -> dict[str, Any]:
        if not isinstance(row, dict):
            # 防御兜底：_load_rows 已保证 dict，但 audit_rows 也可被直接调用；
            # 非 dict 行按"无法判定"处理，不得让 AttributeError 穿透 gather 炸掉 audit。
            return {
                "trace_id": "",
                "category": "",
                "question": "",
                "is_complex": True,
                "reason": "audit_error: 非对象行",
                "audit_errors": 3,
            }
        inp = row.get("input")
        question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
        # 3 次独立判定取多数：LLM 对边界问句的判定有波动，多数裁决更稳。
        # 调用失败 / 字段缺失 / 非布尔：一律记 audit_error 票（"无法判定"），
        # 不得判复杂（掩盖故障）也不得判非复杂（拉低通过率）。
        verdicts: list[tuple[bool, str]] = []  # (admissible_hard, reason_or_error_detail)
        errors = 0
        for _ in range(3):
            try:
                raw = await client.complete(
                    system_prompt=AUDIT_PROMPT,
                    user_prompt=json.dumps({"question": question}, ensure_ascii=False),
                )
                data = VerifyResult.model_validate(json.loads(raw))
                if not data.evidence_is_grounded_for(question):
                    raise ValueError("audit evidence quote must be copied from question")
                verdicts.append((data.route == "complex", data.reason or data.route))
            except Exception as exc:  # noqa: BLE001 无法判定：单独计数，由错误率闸门拦截
                errors += 1
                verdicts.append((True, f"audit_error: {str(exc)[:80]}"))
        is_complex = sum(1 for v, _ in verdicts if v) >= 2
        reasons = [r for v, r in verdicts if not v]
        reason = "; ".join(dict.fromkeys(reasons))[:120] or "多数判定为复杂"
        return {
            "trace_id": row.get("trace_id", ""),
            "category": row.get("category", ""),
            "question": question,
            "is_complex": is_complex,
            "reason": reason,
            "audit_errors": errors,
        }

    try:
        results = await asyncio.gather(*(check(r) for r in rows))
    finally:
        await client.close()
    return results


def _error_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r.get("audit_errors", 0) > 0]


def conclusion(results: list[dict[str, Any]], *, max_ratio: float, max_error_ratio: float) -> tuple[bool, float, float]:
    """单一结论源：render 与 cli 退出码共用同一判定，避免两处独立实现漂移。

    错误率（审计自身 LLM 判定失败的行占比）与非复杂率分用独立阈值；若共用争议率
    容错会系统性掩盖审计失效（3 次判定全挂也可能被误当作可接受争议）。
    错误率默认 0——任何一行无法判定，审计就无法为整批输出背书，必须 FAIL。
    """
    total = len(results)
    if not total:
        return True, 0.0, 0.0
    ratio = sum(1 for r in results if not r["is_complex"]) / total
    error_ratio = len(_error_rows(results)) / total
    passed = ratio <= max_ratio and error_ratio <= max_error_ratio
    return passed, ratio, error_ratio


def render(results: list[dict[str, Any]], *, max_ratio: float, max_error_ratio: float = 0.0) -> str:
    passed, ratio, error_ratio = conclusion(results, max_ratio=max_ratio, max_error_ratio=max_error_ratio)
    total = len(results)
    non_complex = [r for r in results if not r["is_complex"]]
    errors = _error_rows(results)
    lines = [
        f"=== complex 输出审计：共 {total} 行，非复杂 {len(non_complex)} 行 "
        f"({ratio * 100:.1f}%，阈值 {max_ratio * 100:.0f}%)；无法判定 {len(errors)} 行 "
        f"({error_ratio * 100:.1f}%，阈值 {max_error_ratio * 100:.0f}%) ===",
    ]
    for r in non_complex[:20]:
        lines.append(f"  [非复杂] {r['reason'][:50]} | {r['question'][:70]}")
    for r in errors[:10]:
        lines.append(f"  [无法判定] {r['reason'][:50]} | {r['question'][:70]}")
    if len(non_complex) > 20:
        lines.append(f"  ... 其余 {len(non_complex) - 20} 条见完整结果")
    lines.append(
        f"结论: {'PASS' if passed else 'FAIL'}（非复杂率 {ratio * 100:.1f}% "
        f"{'<=' if ratio <= max_ratio else '>'} {max_ratio * 100:.0f}%；"
        f"错误率 {error_ratio * 100:.1f}% "
        f"{'<=' if error_ratio <= max_error_ratio else '>'} {max_error_ratio * 100:.0f}%）"
    )
    return "\n".join(lines)
