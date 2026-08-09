"""Public pipeline API — minimal arguments, sane defaults, no config file needed.

Usage::

    from query_pipeline import run

    summary = run("data/aime/0806.jsonl", output_dir="outputs/aime")
    # summary: dict with per-stage stats, output file paths, success flag
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from query_pipeline.config.models import (
    CheckpointConfig,
    DebugConfig,
    DedupConfig,
    InputConfig,
    JudgeConfig,
    LLMConfig,
    OutputConfig,
    PipelineConfig,
    PostConfig,
    RuleGateConfig,
    SegmentationConfig,
    TranslateConfig,
    VerifyConfig,
)
from query_pipeline.pipeline.runner import run_pipeline


def _find_env_file() -> Path:
    for parent in (Path.cwd(), *Path.cwd().parents):
        if (parent / ".env").exists():
            return parent / ".env"
    return Path.cwd() / ".env"


def _setup_logging(log_file: Path, *, verbose: bool) -> None:
    from query_pipeline.logging_setup import setup_logging

    setup_logging(log_file, verbose=verbose)


def run(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    format: str = "auto",
    model: str = "gpt-5.4-mini",
    concurrency: int = 256,
    stages: list[str] | None = None,
    post_enabled: bool = True,
    dedup_threshold: float | None = None,
    verify_rounds_hard: int | None = None,
    verify_rounds_normal: int | None = None,
    work_dir: str | Path | None = None,
    llm_enabled: bool = True,
    min_tool_calls: int | None = None,
    min_unique_tools: int | None = None,
    reject_rules: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
    verbose: bool = False,
) -> dict:
    """Run the cleaning/annotation pipeline with defaults for everything else.

    Args:
        input_path: jsonl input (session or chat dialect; format auto-detected).
        output_dir: deliverable directory (default: ``outputs/<input parent name>``).
        format: "auto" | "session" | "chat".
        model: LLM model id.
        concurrency: parallel LLM calls.
        stages: custom stage list (default: the built-in funnel order).
        post_enabled: run dedup + translate after verify.
        dedup_threshold / verify_rounds_hard / verify_rounds_normal: knobs
            with pipeline defaults when omitted.
        work_dir: scratch dir for caches/checkpoints (default: ``work/<input stem>``).
        llm_enabled: False runs rules only (empty output is expected).
        verbose: debug logging.

    Returns:
        summary dict: per-stage counts, output files, success flag.
    """
    load_dotenv(_find_env_file(), override=False)
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)

    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"input file not found: {src}")

    if output_dir is None:
        output_dir = Path("outputs") / src.parent.name
    # 日志、输出 jsonl、中间产物、缓存/checkpoint 全部放在同一目录，便于审计。
    if work_dir is None:
        work_dir = output_dir

    out = Path(output_dir)
    work = Path(work_dir)
    _setup_logging(out / "run.log", verbose=verbose)

    # 门槛默认按格式区分（session 7/1/2，chat 3/1/2——chat 工具调用分布平坦，
    # >=7 次仅覆盖 ~1%）；显式传入时覆盖。拿不准时用 suggest 看推荐组合。
    if min_tool_calls is None and min_unique_tools is None:
        rule_gate = (
            RuleGateConfig(min_chain_tool_calls=3, min_unique_tools=2)
            if format == "chat"
            else RuleGateConfig()
        )
    else:
        rule_gate = RuleGateConfig(
            min_chain_tool_calls=min_tool_calls if min_tool_calls is not None else 0,
            min_unique_tools=min_unique_tools if min_unique_tools is not None else 1,
            reject_rules=reject_rules,
        )

    config = PipelineConfig(
        name=f"pipeline:{src.stem}",
        input=InputConfig(path=src, format=format),
        output=OutputConfig(dir=out),
        work_dir=work,
        stages=stages,
        segmentation=SegmentationConfig(enabled=True),
        # chat 工具调用分布平坦（>=7 次仅覆盖 ~1%），粗筛门槛按格式区分：
        # session 7/1/2，chat 3/1/2（与 rule_gate 设计决策一致）。
        rule_gate=rule_gate,
        judge=JudgeConfig(),
        verify=VerifyConfig(
            max_rounds_hard=verify_rounds_hard if verify_rounds_hard is not None else 5,
            max_rounds_normal=verify_rounds_normal if verify_rounds_normal is not None else 2,
        ),
        llm=LLMConfig(
            enabled=llm_enabled,
            model=model,
            concurrency=concurrency,
            cache=work / "logs" / "llm_cache.jsonl",
        ),
        post=PostConfig(
            enabled=post_enabled,
            dedup=DedupConfig(
                threshold=dedup_threshold if dedup_threshold is not None else 0.80,
                entity_slot=True,
            ),
            translate=TranslateConfig(enabled=True),
        ),
        checkpoint=CheckpointConfig(enabled=True, dir=work / "logs" / "checkpoints"),
        debug=DebugConfig(dump_intermediates=True),
    )
    summary = run_pipeline(config)
    import json

    return json.loads(summary.model_dump_json(indent=2, ensure_ascii=False))
