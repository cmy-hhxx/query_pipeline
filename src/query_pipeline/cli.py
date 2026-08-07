from __future__ import annotations

import argparse
import logging

from query_pipeline.config.loader import load_pipeline_config
from query_pipeline.pipeline.runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run question cleaning and classification pipeline")
    parser.add_argument("-c", "--config", default="configs/aime/config.yaml", help="Pipeline YAML config path")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_pipeline_config(args.config)
    if args.dry_run:
        print(f"OK: pipeline={cfg.name}, flow=session_stage->llm_stage")
        return 0

    summary = run_pipeline(cfg)
    print(summary.model_dump_json(indent=2, ensure_ascii=False))
    return 0 if summary.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
