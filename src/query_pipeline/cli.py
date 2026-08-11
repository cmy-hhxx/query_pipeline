from __future__ import annotations

import argparse
import json
from pathlib import Path

from query_pipeline import api
from query_pipeline.api import _find_env_file
from query_pipeline.logging_setup import logging_session


def _add_log_args(parser: argparse.ArgumentParser, *, verbose: bool = True) -> None:
    parser.add_argument("--log-dir", default=None, help="Ordinary/business log root")
    parser.add_argument("--batch-id", default=None, help="Stable batch id (default: generated)")
    if verbose:
        parser.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="query-pipeline", description="Question cleaning and annotation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the pipeline on an input jsonl")
    run_parser.add_argument("input", help="Input jsonl (session or chat; format auto-detected)")
    run_parser.add_argument("-o", "--output-dir", default=None, help="Output directory (default: outputs/<input parent>)")
    run_parser.add_argument("--format", default="auto", choices=("auto", "session", "chat"))
    run_parser.add_argument("--model", default="gpt-5.4-mini", help="LLM model id")
    run_parser.add_argument("--concurrency", type=int, default=256, help="Parallel LLM calls")
    run_parser.add_argument("--stages", default=None, help="Comma-separated stage names (default: built-in order)")
    run_parser.add_argument("--no-post", action="store_true", help="Skip dedup/translate")
    run_parser.add_argument("--work-dir", default=None, help="Scratch dir override (default: the output dir)")
    run_parser.add_argument("--no-llm", action="store_true", help="Rules only (empty output expected)")
    run_parser.add_argument("--skip-precheck", action="store_true", help="跳过数据预检（默认预检发现严重问题即中止，避免浪费 LLM 资源）")
    run_parser.add_argument("--allow-no-chain", action="store_true", help="允许输入整体缺 chain（chain 覆盖率检查降为 0）")
    run_parser.add_argument("--min-tool-calls", type=int, default=None, help="rule_gate 工具调用数门槛（默认 session 7 / chat 3）")
    run_parser.add_argument("--min-unique-tools", type=int, default=None, help="rule_gate 工具种数门槛（默认 session 2 / chat 2）")
    run_parser.add_argument("--no-reject-rules", action="store_true", help="关闭 reject 规则")
    run_parser.add_argument("--verify-rounds", type=int, default=None, help="hard 复杂问句严格复核轮数（默认 1）")
    run_parser.add_argument("--dedup-mode", choices=("semantic", "lexical"), default="semantic", help="去重模式（默认 semantic）")
    run_parser.add_argument("--semantic-dedup-threshold", type=float, default=None, help="语义签名候选阈值（默认 0.60）")
    run_parser.add_argument("--max-dedup-candidates", type=int, default=20, help="每行最多语义去重候选数（默认 20）")
    run_parser.add_argument("--dedup-threshold", type=float, default=None, help="lexical 回退 Jaccard 阈值（默认 0.80）")
    run_parser.add_argument("--api-key", default=None, help="OPENAI_API_KEY 覆盖（默认读 .env）")
    run_parser.add_argument("--base-url", default=None, help="OPENAI_BASE_URL 覆盖（默认读 .env）")
    _add_log_args(run_parser)

    suggest_parser = sub.add_parser(
        "suggest", help="扫描 rule_gate 门槛，按过滤后候选数推荐参数组合（不调 LLM）"
    )
    suggest_parser.add_argument("input", help="输入 jsonl（格式自动识别）")
    suggest_parser.add_argument("--format", default="auto", choices=("auto", "session", "chat"))
    suggest_parser.add_argument("--top", type=int, default=10, help="返回组合数（默认 10）")
    _add_log_args(suggest_parser)

    precheck_parser = sub.add_parser(
        "precheck", help="数据预检：扫描输入，提前暴露坏行/缺 chain/零合格 turn 等问题（纯规则，不调 LLM）"
    )
    precheck_parser.add_argument("input", help="输入 jsonl（session 或 chat；格式自动识别）")
    precheck_parser.add_argument("--format", default="auto", choices=("auto", "session", "chat"))
    precheck_parser.add_argument("--min-chain-coverage", type=float, default=0.5, help="chain 覆盖率阈值（默认 0.5，低于即失败）")
    precheck_parser.add_argument("--max-bad-line-ratio", type=float, default=0.01, help="坏行占比阈值（默认 0.01，超过即失败）")

    audit_parser = sub.add_parser(
        "audit", help="对 complex_queries 输出做严格复核（独立 LLM），报告非复杂占比"
    )
    audit_parser.add_argument("input", help="complex_queries.jsonl 路径")
    audit_parser.add_argument("--max-ratio", type=float, default=0.02, help="非复杂争议率阈值（默认 2%%，超出则退出码 1）")
    audit_parser.add_argument("--max-error-ratio", type=float, default=0.0, help="无法判定行占比阈值（默认 0：任何一行判定失败即 FAIL；独立于非复杂率）")
    audit_parser.add_argument("--model", default="gpt-5.4-mini")
    audit_parser.add_argument("--concurrency", type=int, default=64)
    _add_log_args(audit_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        stages = args.stages.split(",") if args.stages else None
        summary = api.run(
            args.input,
            output_dir=args.output_dir,
            format=args.format,
            model=args.model,
            concurrency=args.concurrency,
            stages=stages,
            post_enabled=not args.no_post,
            dedup_mode=args.dedup_mode,
            semantic_dedup_threshold=args.semantic_dedup_threshold,
            max_dedup_candidates=args.max_dedup_candidates,
            work_dir=args.work_dir,
            log_dir=args.log_dir,
            batch_id=args.batch_id,
            llm_enabled=not args.no_llm,
            precheck_enabled=not args.skip_precheck,
            precheck_min_chain_coverage=0.0 if args.allow_no_chain else None,
            min_tool_calls=args.min_tool_calls,
            min_unique_tools=args.min_unique_tools,
            reject_rules=not args.no_reject_rules,
            verify_rounds_hard=args.verify_rounds,
            api_key=args.api_key,
            base_url=args.base_url,
            verbose=args.verbose,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["success"] else 1
    if args.command == "precheck":
        from query_pipeline.precheck import precheck, render

        try:
            report = precheck(
                args.input,
                format=args.format,
                min_chain_coverage=args.min_chain_coverage,
                max_bad_line_ratio=args.max_bad_line_ratio,
            )
        except ValueError as exc:
            print(f"precheck 失败: {exc}")
            return 1
        print(render(report))
        return 0 if report.ok else 1
    if args.command == "audit":
        import asyncio
        from pathlib import Path as _Path

        from dotenv import load_dotenv

        input_path = Path(args.input).expanduser().resolve()
        log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else input_path.parent / "logs"
        with logging_session(
            log_dir, command="audit", batch_id=args.batch_id, verbose=args.verbose
        ):
            load_dotenv(_find_env_file(), override=False)
            from query_pipeline.audit import _load_rows, audit_rows, conclusion, render

            rows = _load_rows(input_path)
            if not rows:
                print("输入为空")
                return 1
            results = asyncio.run(
                audit_rows(rows, model=args.model, concurrency=args.concurrency)
            )
            print(render(results, max_ratio=args.max_ratio, max_error_ratio=args.max_error_ratio))
            # 退出码与 render 的 PASS/FAIL 共用同一结论（conclusion 单源，不重复计算）。
            passed, _ratio, _error_ratio = conclusion(
                results, max_ratio=args.max_ratio, max_error_ratio=args.max_error_ratio
            )
            return 0 if passed else 1
    if args.command == "suggest":
        log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else Path.cwd() / "logs"
        with logging_session(
            log_dir, command="suggest", batch_id=args.batch_id, verbose=args.verbose
        ):
            from query_pipeline.suggest import render_suggestions, suggest_gates

            fmt = args.format if args.format != "auto" else "auto"
            suggestions = suggest_gates(args.input, format=args.format, top=args.top)
            if not suggestions:
                print("无可推荐组合（输入无有效行）")
                return 1
            print(render_suggestions(args.input, suggestions, fmt=fmt))
            return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
