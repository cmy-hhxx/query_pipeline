from __future__ import annotations

import argparse
import json
import logging
import sys

from query_pipeline import api


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
    run_parser.add_argument("--min-tool-calls", type=int, default=None, help="rule_gate 工具调用数门槛（默认 session 7 / chat 3）")
    run_parser.add_argument("--min-unique-tools", type=int, default=None, help="rule_gate 工具种数门槛（默认 session 2 / chat 2）")
    run_parser.add_argument("--no-reject-rules", action="store_true", help="关闭 reject 规则")
    run_parser.add_argument("--verify-rounds", type=int, default=None, help="复杂问句 verify 轮数（默认 5）")
    run_parser.add_argument("--dedup-threshold", type=float, default=None, help="去重 Jaccard 阈值（默认 0.80）")
    run_parser.add_argument("--api-key", default=None, help="OPENAI_API_KEY 覆盖（默认读 .env）")
    run_parser.add_argument("--base-url", default=None, help="OPENAI_BASE_URL 覆盖（默认读 .env）")
    run_parser.add_argument("-v", "--verbose", action="store_true")

    suggest_parser = sub.add_parser(
        "suggest", help="扫描 rule_gate 门槛，按过滤后候选数推荐参数组合（不调 LLM）"
    )
    suggest_parser.add_argument("input", help="输入 jsonl（格式自动识别）")
    suggest_parser.add_argument("--format", default="auto", choices=("auto", "session", "chat"))
    suggest_parser.add_argument("--top", type=int, default=10, help="返回组合数（默认 10）")
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
            work_dir=args.work_dir,
            llm_enabled=not args.no_llm,
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
    if args.command == "suggest":
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
