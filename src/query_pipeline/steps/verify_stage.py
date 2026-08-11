from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

from query_pipeline.io.checkpoint import content_key, stage_checkpoint
from query_pipeline.io.jsonl import write_jsonl
from query_pipeline.llm.cache import make_cache_key, put_cache
from query_pipeline.llm.client import LLMClient
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.session import parse_verify_payload, parse_verify_response
from query_pipeline.pipeline.context import PipelineContext
from query_pipeline.post.dedup import review_template_families, semantic_dedup_rows
from query_pipeline.prompts import resolve_prompt
from query_pipeline.session.funnel import PARSE_MAX_ATTEMPTS, _call, parse_classify_response
from query_pipeline.taxonomy import load_taxonomy


def _prior_questions(row: dict[str, Any]) -> list[str]:
    context = row.get("context") or []
    return [str(t.get("question") or "") for t in context if isinstance(t, dict)]


def _verify_content_key(row: dict[str, Any], prior: list[str], difficulty: str) -> str:
    """Checkpoint key covers the verdict and possible normal reclassification.

    Verify itself only sees the current question. Prior questions remain in the
    key because a downgrade invokes the normal classifier, where context may be
    used for category selection. Difficulty controls whether review runs.
    """
    inp = row.get("input")
    question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
    return content_key(
        str(row.get("source_case_id", "")),
        str(row.get("trace_id", "")),
        question,
        difficulty,
        "\n".join(prior),
    )


async def run_verify_stage(
    ctx: PipelineContext,
    client: LLMClient | None,
    cache: dict[str, dict[str, Any]],
    cache_lock: asyncio.Lock,
) -> PipelineContext:
    """Precision-first hard critic.

    Only initial hard rows are reviewed. Semantic uncertainty becomes normal;
    confirmed templates are rejected. Infrastructure failures are counted as
    fatal verification errors and are never converted into business labels.
    """
    cfg = ctx.config
    # These diagnostics describe this run only. Remove stale files even when
    # intermediate dumping is enabled and the current run produces zero rows.
    for name in ("verified.jsonl", "deduped.jsonl", "complex_policy_rejected.jsonl"):
        ctx.path(name).unlink(missing_ok=True)
    if not cfg.verify.enabled or client is None or not ctx.rows:
        # llm_failed 不跳过 verify：个别候选 judge 失败只丢弃自身，其余行仍可复核
        # （fail-open，回到 pre-91cfeb2 语义）。
        return ctx

    checkpoint = stage_checkpoint(cfg, "verify")
    counts = {"kept": 0, "to_normal": 0, "rejected": 0, "uncertain": 0, "failed": 0}
    debug: list[dict[str, Any]] = []
    force_normal: set[int] = set()
    policy_rejected: list[dict[str, Any]] = []

    if cfg.post.enabled and cfg.post.dedup.enabled and cfg.post.dedup.mode == "semantic":
        # Corpus review runs before per-row verification only when semantic
        # post-processing is enabled. Shared phrases are candidates, not direct
        # deletion evidence.
        (
            family_rows,
            family_dropped,
            family_stats,
            force_normal,
            semantic_protected,
        ) = await review_template_families(
            ctx.rows,
            cfg.post.dedup,
            client=client,
            llm_cfg=cfg.llm,
            cache=cache,
            cache_path=cfg.cache_path,
            cache_lock=cache_lock,
        )
        deduped_rows, semantic_dropped, semantic_stats = await semantic_dedup_rows(
            family_rows,
            cfg.post.dedup,
            client=client,
            llm_cfg=cfg.llm,
            cache=cache,
            cache_path=cfg.cache_path,
            cache_lock=cache_lock,
            protected_row_ids=semantic_protected,
        )
        ctx.rows = deduped_rows
        counts["failed"] += family_stats["template_family_failed"] + semantic_stats[
            "semantic_dedup_failed"
        ]
        ctx.stats.update(family_stats)
        ctx.stats.update(semantic_stats)
        ctx.stats["template_family_rejected"] = family_stats["template_family_rejected"]
        ctx.stats["duplicate_removed"] = (
            family_stats["template_family_duplicates"] + len(semantic_dropped)
        )
        all_corpus_dropped = family_dropped + semantic_dropped
        ctx.stats["dedup_removed"] = len(all_corpus_dropped)
        ctx.stats["verify_corpus_review_done"] = True
        policy_rejected = [
            item for item in family_dropped if item.get("decision") == "eval_template_family"
        ]
        if all_corpus_dropped and cfg.debug.dump_intermediates:
            ctx.work_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(ctx.path("deduped.jsonl"), all_corpus_dropped)

    async def classify_normal(row: dict[str, Any]) -> tuple[str, str | None]:
        inp = row.get("input")
        question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
        payload = {"prior_questions": _prior_questions(row), "current_question": question}
        parsed = await _call(
            client,
            cfg.llm,
            cache,
            cfg.cache_path,
            cache_lock,
            step="classify_normal_after_verify",
            prompt_id=cfg.judge.classify_normal_prompt,
            payload=payload,
            parse=parse_classify_response,
        )
        if parsed.category_id not in load_taxonomy().normal:
            raise ValueError(
                f"classify_normal returned invalid normal category {parsed.category_id!r}"
            )
        category = load_taxonomy().get("normal", parsed.category_id).path
        return category, parsed.reason

    async def worker(row: dict[str, Any]) -> dict[str, Any]:
        inp = row.get("input")
        question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
        difficulty = row.get("difficulty_level", "hard")
        # 精度闸门只降级 hard；normal 不做反向升级。
        if difficulty != "hard":
            return {"action": "pass_normal", "reason": None, "error": None, "rounds": []}
        max_rounds = cfg.verify.max_rounds_hard
        if id(row) in force_normal:
            try:
                category, classify_reason = await classify_normal(row)
            except Exception as exc:  # noqa: BLE001 infrastructure failure, not a business label
                return {
                    "action": "error",
                    "reason": "模板族边界不清，normal 分类失败",
                    "error": str(exc)[:200],
                    "rounds": [],
                    "uncertain": False,
                }
            return {
                "action": "downgrade",
                "category": category,
                "reason": classify_reason or "模板族边界不清",
                "downgrade_reason": "模板族判定置信度低",
                "error": None,
                "rounds": [],
                "uncertain": True,
            }
        if max_rounds == 0:
            return {"action": "keep", "reason": None, "error": None, "rounds": []}
        prior = _prior_questions(row)
        key = _verify_content_key(row, prior, difficulty)
        record = checkpoint.get(key)
        if record is not None:
            action = record.get("action")
            if action in {"keep", "downgrade", "reject"}:
                return {
                    "action": action,
                    "category": record.get("category"),
                    "reason": record.get("reason"),
                    "downgrade_reason": record.get("downgrade_reason"),
                    "error": record.get("error"),
                    "rounds": record.get("rounds", []),
                    "uncertain": bool(record.get("uncertain", False)),
                }
        user_prompt = "请复核以下问句，只输出严格 JSON：\n" + json.dumps(
            {"question": question},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        rounds: list[dict[str, Any]] = []
        error: str | None = None
        reason: str | None = None
        semantic_uncertain = False
        keep_hard = False
        reject = False
        for round_no in range(1, max_rounds + 1):
            # round 1 是独立最简解法批判；显式配置多轮时，后续轮次使用
            # 同一严格口径的独立复核，并要求全票通过。
            if round_no == 1:
                prompt_id = cfg.verify.prompt_id
                system_prompt = resolve_prompt(prompt_id)
            else:
                prompt_id = "verify_recheck"
                system_prompt = resolve_prompt(prompt_id).format(round_no=round_no)
            cache_key = make_cache_key(
                user_prompt, step=f"verify:{prompt_id}", model=cfg.llm.model, prompt=system_prompt
            )
            try:
                parsed: Any | None = None
                if cache_key in cache:
                    try:
                        parsed = parse_verify_payload(cache[cache_key])
                        if not parsed.evidence_is_grounded_for(question):
                            raise ValueError(
                                "verify evidence quote must be copied from question"
                            )
                    except (ValueError, RuntimeError) as exc:
                        # 坏缓存 label：驱逐并重调，避免每次运行重复丢弃（与 segment 自愈一致）
                        logger.warning(
                            "cached verify label invalid, re-calling LLM: %s", str(exc)[:120]
                        )
                        cache.pop(cache_key, None)
                if parsed is None:
                    # 模型输出概率性：解析/校验失败重调（与 funnel._call 同一口径）。
                    for attempt in range(1, PARSE_MAX_ATTEMPTS + 1):
                        raw = await client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
                        try:
                            parsed = parse_verify_response(raw)
                            if not parsed.evidence_is_grounded_for(question):
                                raise ValueError("verify evidence quote must be copied from question")
                            break
                        except (ValueError, RuntimeError) as exc:
                            if attempt == PARSE_MAX_ATTEMPTS:
                                raise
                            logger.warning(
                                "verify label invalid (attempt %d/%d), re-calling LLM: %s",
                                attempt, PARSE_MAX_ATTEMPTS, str(exc)[:120],
                            )
                    # 循环内要么成功赋值要么 raise，不可能走到这里仍为 None
                    assert parsed is not None
                    await put_cache(
                        cache,
                        cfg.cache_path,
                        cache_key,
                        parsed.to_cache_label(),
                        meta={
                            "step": "verify",
                            "prompt_id": prompt_id,
                            "round": round_no,
                            "model": cfg.llm.model,
                            "question": question[:120],
                        },
                        lock=cache_lock,
                    )
            except Exception as exc:  # noqa: BLE001 任意批判调用故障都不能保留 hard
                error = str(exc)[:200]
                break
            reason = parsed.reason
            rounds.append(
                {
                    "round": round_no,
                    "prompt_id": prompt_id,
                    "route": parsed.route,
                    "complex_features": parsed.complex_features,
                    "exclusion_reasons": parsed.exclusion_reasons,
                    "evidence": [item.model_dump() for item in parsed.evidence],
                    "confidence": parsed.confidence,
                    "reason": parsed.reason,
                }
            )
            if parsed.route == "reject":
                reject = True
                break
            if not parsed.admits_complex_for(question):
                semantic_uncertain = parsed.confidence == "low"
                break
            keep_hard = round_no == max_rounds

        if error is not None:
            return {
                "action": "error",
                "reason": f"complex verification failed: {error}",
                "error": error,
                "rounds": rounds,
                "uncertain": False,
            }

        if keep_hard and error is None:
            await checkpoint.mark(key, action="keep", reason=reason, rounds=rounds)
            return {
                "action": "keep", "reason": reason, "error": None,
                "rounds": rounds, "uncertain": False,
            }

        if reject:
            await checkpoint.mark(key, action="reject", reason=reason, rounds=rounds)
            return {
                "action": "reject",
                "reason": reason or "严重模板或嵌入提示词",
                "error": None,
                "rounds": rounds,
                "uncertain": False,
            }

        # 语义边界不清不能留在 complex，但仍有金融任务时进入 normal。
        downgrade_reason = reason or "未通过 complex 复核"
        try:
            category, classify_reason = await classify_normal(row)
        except Exception as exc:  # noqa: BLE001 与 funnel 同样按单行失败兜底
            return {
                "action": "error",
                "reason": downgrade_reason,
                "error": str(exc)[:200],
                "rounds": rounds,
                "uncertain": False,
            }
        final_reason = classify_reason or downgrade_reason
        if error is None:
            await checkpoint.mark(
                key,
                action="downgrade",
                category=category,
                reason=final_reason,
                downgrade_reason=downgrade_reason,
                rounds=rounds,
                uncertain=semantic_uncertain,
            )
        return {
            "action": "downgrade",
            "category": category,
            "reason": final_reason,
            "downgrade_reason": downgrade_reason,
            "error": None,
            "rounds": rounds,
            "uncertain": semantic_uncertain,
        }

    results = await run_concurrent(ctx.rows, worker, description="LLM verify")

    kept: list[dict[str, Any]] = []
    for row, result in zip(ctx.rows, results):
        if result is None:
            result = {
                "action": "error",
                "reason": "complex verification failed unexpectedly",
                "error": "worker returned no result",
                "rounds": [],
                "uncertain": False,
            }
        action, reason, error = result["action"], result["reason"], result["error"]
        inp = row.get("input")
        question = str(inp.get("text") or "") if isinstance(inp, dict) else ""
        debug.append(
            {
                "source_case_id": row.get("source_case_id", ""),
                "trace_id": row.get("trace_id", ""),
                "difficulty": row.get("difficulty_level", ""),
                "category": row.get("category", ""),
                "question": question[:200],
                "route": "complex" if action == "keep" else (
                    "normal" if action in {"downgrade", "pass_normal"}
                    else "reject" if action == "reject" else "error"
                ),
                "action": action,
                "reason": reason,
                "error": error,
                "rounds": result.get("rounds", []),
            }
        )
        if error is not None or action == "error":
            counts["failed"] += 1
        elif action == "pass_normal":
            kept.append(row)
        elif action == "reject":
            counts["rejected"] += 1
            policy_rejected.append(
                {
                    "source_case_id": row.get("source_case_id", ""),
                    "trace_id": row.get("trace_id", ""),
                    "text": question[:200],
                    "method": "single_question_verify",
                    "decision": "reject",
                    "reason": reason,
                    "rounds": result.get("rounds", []),
                }
            )
        elif action == "keep":
            counts["kept"] += 1
            meta = row.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["verify_reason"] = reason
                profile = meta.get("complexity_profile")
                if isinstance(profile, dict):
                    profile["verified_hard"] = True
                    profile["verify_reason"] = reason
                    rounds = result.get("rounds") or []
                    if rounds:
                        critic = rounds[-1]
                        profile["route"] = "complex"
                        profile["complex_features"] = critic.get("complex_features", [])
                        profile["exclusion_reasons"] = critic.get("exclusion_reasons", [])
                        profile["evidence"] = critic.get("evidence", [])
                        profile["confidence"] = critic.get("confidence", profile.get("confidence"))
            kept.append(row)
        else:  # downgrade
            downgraded = copy.deepcopy(row)
            downgraded["difficulty_level"] = "normal"
            downgraded["category"] = result["category"]
            meta = downgraded.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["reason"] = reason
                meta["verify_reason"] = result.get("downgrade_reason") or reason
                meta["normal_classification_reason"] = reason
                profile = meta.get("complexity_profile")
                if isinstance(profile, dict):
                    profile["route"] = "normal"
                    profile["verified_hard"] = False
                    profile["verify_reason"] = result.get("downgrade_reason") or reason
                    profile["normal_classification_reason"] = reason
                    rounds = result.get("rounds") or []
                    if rounds:
                        critic = rounds[-1]
                        profile["complex_features"] = critic.get("complex_features", [])
                        profile["exclusion_reasons"] = critic.get("exclusion_reasons", [])
                        profile["evidence"] = critic.get("evidence", [])
                        profile["confidence"] = critic.get(
                            "confidence", profile.get("confidence")
                        )
            counts["to_normal"] += 1
            if result.get("uncertain"):
                counts["uncertain"] += 1
            kept.append(downgraded)

    ctx.rows = kept
    logger.info(
        "[verify] complex=%d to_normal=%d rejected=%d uncertain=%d failed=%d",
        counts["kept"], counts["to_normal"], counts["rejected"],
        counts["uncertain"], counts["failed"],
    )
    ctx.stats["verify_complex_kept"] = counts["kept"]
    ctx.stats["verify_to_normal"] = counts["to_normal"]
    ctx.stats["verify_rejected_template"] = counts["rejected"]
    ctx.stats["verify_failed"] = counts["failed"]
    ctx.stats["verify_uncertain"] = counts["uncertain"]
    ctx.stats["final_complex_rows"] = sum(r.get("difficulty_level") == "hard" for r in kept)
    ctx.stats["final_normal_rows"] = sum(r.get("difficulty_level") == "normal" for r in kept)
    final_complex_features: Counter[str] = Counter()
    final_complex_categories: Counter[str] = Counter()
    final_normal_categories: Counter[str] = Counter()
    for final_row in kept:
        meta = final_row.get("meta")
        profile = meta.get("complexity_profile") if isinstance(meta, dict) else None
        if isinstance(profile, dict) and isinstance(profile.get("complex_features"), list):
            final_complex_features.update(str(item) for item in profile["complex_features"])
        category = str(final_row.get("category") or "")
        if final_row.get("difficulty_level") == "hard":
            final_complex_categories[category] += 1
        else:
            final_normal_categories[category] += 1
    ctx.stats["complex_feature_counts_final"] = dict(sorted(final_complex_features.items()))
    ctx.stats["category_counts_final"] = dict(sorted(final_complex_categories.items()))
    ctx.stats["category_counts_normal_final"] = dict(sorted(final_normal_categories.items()))
    if debug and cfg.debug.dump_intermediates:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(ctx.path("verified.jsonl"), debug)
    if policy_rejected and cfg.debug.dump_intermediates:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(ctx.path("complex_policy_rejected.jsonl"), policy_rejected)
    return ctx
