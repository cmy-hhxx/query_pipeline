"""Public pipeline API — minimal arguments, sane defaults, no config file needed.

Usage::

    from query_pipeline import run

    summary = run("data/aime/0806.jsonl", output_dir="outputs/aime")
    # summary: dict with per-stage stats, output file paths, success flag
"""

from __future__ import annotations

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
    if verbose:
        import logging

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    load_dotenv(_find_env_file(), override=False)

    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"input file not found: {src}")

    if output_dir is None:
        output_dir = Path("outputs") / src.parent.name
    if work_dir is None:
        work_dir = Path("work") / src.stem

    out = Path(output_dir)
    work = Path(work_dir)

    config = PipelineConfig(
        name=f"pipeline:{src.stem}",
        input=InputConfig(path=src, format=format),
        output=OutputConfig(dir=out, complex_queries="cleaned_queries.jsonl", summary="summary.json"),
        work_dir=work,
        stages=stages,
        segmentation=SegmentationConfig(enabled=True),
        rule_gate=RuleGateConfig(),
        judge=JudgeConfig(),
        verify=VerifyConfig(
            max_rounds_hard=verify_rounds_hard if verify_rounds_hard is not None else 5,
            max_rounds_normal=verify_rounds_normal if verify_rounds_normal is not None else 2,
        ),
        llm=LLMConfig(
            enabled=llm_enabled,
            model=model,
            concurrency=concurrency,
            cache=work / "llm_cache.jsonl",
        ),
        post=PostConfig(
            enabled=post_enabled,
            dedup=DedupConfig(
                threshold=dedup_threshold if dedup_threshold is not None else 0.80,
                entity_slot=True,
            ),
            translate=TranslateConfig(enabled=True),
        ),
        checkpoint=CheckpointConfig(enabled=True, dir=work / "checkpoints"),
        debug=DebugConfig(dump_intermediates=True),
    )
    summary = run_pipeline(config)
    import json

    return json.loads(summary.model_dump_json(indent=2, ensure_ascii=False))
