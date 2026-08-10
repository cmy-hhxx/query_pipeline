from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from query_pipeline.config.models import LLMConfig
from query_pipeline.logging_setup import beijing_converter
from query_pipeline.io.jsonl import read_jsonl_with_bad_lines, write_jsonl
from query_pipeline.llm.cache import load_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.quality import aggregate, judge as judge_mod, report, rules
from query_pipeline.quality.paths import llm_cache_path, project_root, qc_dir, source_path

logger = logging.getLogger(__name__)


# 与管线一致：整个日志系统统一北京时间（含第三方库）。
setattr(logging.Formatter, "converter", staticmethod(beijing_converter))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="query-pipeline-qc",
        description="清洗后问句输出的质检模块：全量规则 + LLM 抽检（问句质量 / 标签归属）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="运行质检并写入 QC 产物目录")
    run.add_argument("--dataset", required=True, help="数据集名，如 aime")
    run.add_argument("--date", required=True, help="日期，如 0807")
    run.add_argument("--ratio", type=float, default=0.05, help="LLM 抽样比例（默认 0.05）")
    run.add_argument("--seed", type=int, default=42, help="抽样随机种子（默认 42）")
    run.add_argument("--no-llm", action="store_true", help="跳过 LLM 抽检，只跑规则")
    run.add_argument("--model", default="gpt-5.4-mini", help="judge 模型（默认复用管线模型）")
    run.add_argument("--concurrency", type=int, default=64, help="LLM 并发数")
    run.add_argument("--input", type=Path, help="输出 jsonl 路径覆盖（默认 outputs/<dataset>/cleaned_queries.jsonl）")
    run.add_argument("--qc-dir", type=Path, help="QC 产物目录覆盖（默认 outputs/<dataset>/qc/<date>）")
    run.add_argument("--cache", type=Path, help="LLM 缓存文件覆盖（默认 outputs/<dataset>/logs/llm_cache.jsonl）")
    run.add_argument("-v", "--verbose", action="store_true")
    return parser


def _run_llm_phase(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    cache_path: Path,
) -> tuple[list[int], dict[str, dict[str, Any]]]:
    """Sample + judge; returns (sample_indices, verdicts_by_trace_id)."""

    async def _judge() -> tuple[list[int], list[dict[str, Any]]]:
        client = LLMClient(LLMConfig(model=args.model, concurrency=args.concurrency))
        cache = load_cache(cache_path)
        cache_lock = asyncio.Lock()
        try:
            return await judge_mod.run_llm_judge(
                records,
                client,
                cache,
                cache_lock,
                cache_path,
                ratio=args.ratio,
                seed=args.seed,
            )
        finally:
            await client.close()

    sample_indices, verdicts = asyncio.run(_judge())
    by_trace: dict[str, dict[str, Any]] = {v["trace_id"]: v for v in verdicts}
    return sample_indices, by_trace


def run_quality(args: argparse.Namespace) -> int:
    root = project_root()
    load_dotenv(root / ".env", override=False)

    source = (args.input or source_path(args.dataset, root)).resolve()
    if not source.exists():
        print(f"错误：找不到输出文件 {source}")
        return 1
    out_dir = (args.qc_dir or qc_dir(args.dataset, args.date, root)).resolve()
    cache_path = (args.cache or llm_cache_path(args.dataset, root)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records, skipped = read_jsonl_with_bad_lines(source, out_dir / "bad_lines.jsonl")
    logger.info("读取 %d 条记录，跳过 %d 条坏行", len(records), skipped)

    # 1. rules (deterministic, all records)
    per_record: dict[str, list[dict[str, Any]]] = {
        aggregate.record_key(row): rules.check_record(row) for row in records
    }
    dataset_rules = rules.run_dataset_rules(records)

    # 2. LLM sampling (skip with --no-llm)
    judge_results: dict[str, dict[str, Any]] = {}
    sample_indices: list[int] = []
    if not args.no_llm:
        try:
            sample_indices, judge_results = _run_llm_phase(records, args, cache_path)
        except RuntimeError as exc:
            print(f"错误：无法初始化 LLM client（{exc}）。可用 --no-llm 只跑规则。")
            return 1

    # 3. aggregate + persist
    sample_set = {aggregate.record_key(records[i]) for i in sample_indices}
    results = aggregate.build_results(records, per_record, sample_set, judge_results)
    overview = aggregate.build_overview(
        records,
        results,
        dataset_rules,
        dataset=args.dataset,
        date=args.date,
        source=str(source),
        ratio=args.ratio,
        seed=args.seed,
        bad_lines=skipped,
    )

    write_jsonl(out_dir / "results.jsonl", results)
    (out_dir / "overview.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(report.render_markdown(overview, results), encoding="utf-8")

    if sample_indices:
        sampled_rows: list[dict[str, Any]] = []
        for i in sample_indices:
            row = records[i]
            key = aggregate.record_key(row)
            inp = row.get("input")
            question = inp.get("text") if isinstance(inp, dict) else ""
            sampled_rows.append(
                {
                    "trace_id": str(row.get("trace_id") or "") or key,
                    "source_case_id": row.get("source_case_id", ""),
                    "category": row.get("category", ""),
                    "question": str(question)[:80],
                    "judge": judge_results.get(key),
                }
            )
        write_jsonl(out_dir / "sampled.jsonl", sampled_rows)

    report.print_terminal(overview)
    print(f"QC 产物目录：{out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "run":
        return run_quality(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
