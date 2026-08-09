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
    run_parser.add_argument("-v", "--verbose", action="store_true")
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
            verbose=args.verbose,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["success"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
