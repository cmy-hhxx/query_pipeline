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
    LoggingConfig,
    OutputConfig,
    PipelineConfig,
    PostConfig,
    PrecheckConfig,
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
    log_dir: str | Path | None = None,
    batch_id: str | None = None,
    llm_enabled: bool = True,
    precheck_enabled: bool = True,
    precheck_min_chain_coverage: float | None = None,
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
        work_dir: scratch dir for caches/checkpoints (default: the output dir).
        log_dir: ordinary/business log root (default: ``<output_dir>/logs``).
        batch_id: optional stable batch identity; omit to generate one.
        llm_enabled: False runs rules only (empty output is expected).
        precheck_enabled: run the data precheck stage before any LLM work
            (critical issues abort the run; disable only when you know the
            input is fine).
        precheck_min_chain_coverage: minimum chain coverage on eligible turns
            (default 0.5); pass 0.0 to allow chain-less (end2end) input.
        verbose: debug logging.

    Returns:
        summary dict: per-stage counts, output files, success flag.
    """
    load_dotenv(_find_env_file(), override=False)
    if api_key:
        # 显式传参必须覆盖 env：setdefault 在 env 已有 key 时会静默丢弃用户显式
        # 传入的 key（cli.py 帮助文本明示 --api-key 是"OPENAI_API_KEY 覆盖"）。
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    src = Path(input_path).expanduser().resolve()

    if output_dir is None:
        output_dir = Path("outputs") / src.parent.name
    if work_dir is None:
        work_dir = output_dir

    out = Path(output_dir).expanduser()
    work = Path(work_dir).expanduser()
    logs = Path(log_dir).expanduser() if log_dir is not None else out / "logs"

    # 门槛默认按格式区分（session 7/1/2，chat 3/1/2——chat 工具调用分布平坦，
    # >=7 次仅覆盖 ~1%）：未显式传入的旋钮保持 None，由 rule_gate 阶段按嗅探到的
    # 实际格式补齐（format="auto" 也不会用错门槛）；reject_rules 始终透传，
    # --no-reject-rules 不再被默认分支吞掉。拿不准时用 suggest 看推荐组合。
    rule_gate = RuleGateConfig(
        min_chain_tool_calls=min_tool_calls,
        min_unique_tools=min_unique_tools,
        reject_rules=reject_rules,
    )

    config = PipelineConfig(
        name=f"pipeline:{src.stem}",
        input=InputConfig(path=src, format=format),
        output=OutputConfig(dir=out),
        work_dir=work,
        stages=stages,
        precheck=PrecheckConfig(
            enabled=precheck_enabled,
            min_chain_coverage=0.5 if precheck_min_chain_coverage is None else precheck_min_chain_coverage,
        ),
        segmentation=SegmentationConfig(enabled=True),
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
            cache=work / "runtime" / "cache" / "llm_cache.jsonl",
        ),
        post=PostConfig(
            enabled=post_enabled,
            dedup=DedupConfig(
                threshold=dedup_threshold if dedup_threshold is not None else 0.80,
                entity_slot=True,
            ),
            translate=TranslateConfig(enabled=True),
        ),
        checkpoint=CheckpointConfig(enabled=True, dir=work / "runtime" / "checkpoints"),
        debug=DebugConfig(dump_intermediates=True),
        logging=LoggingConfig(
            dir=logs,
            batch_id=batch_id,
            level="DEBUG" if verbose else "INFO",
        ),
    )
    summary = run_pipeline(config)
    import json

    return json.loads(summary.model_dump_json(indent=2, ensure_ascii=False))
