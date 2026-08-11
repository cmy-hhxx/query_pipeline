from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from query_pipeline.config.models import DedupConfig, LLMConfig
from query_pipeline.llm.cache import make_cache_key, put_cache
from query_pipeline.llm.runner import run_concurrent
from query_pipeline.models.session import parse_json_object
from query_pipeline.prompts import resolve_prompt
from query_pipeline.rules.normalize import (
    SLOT_PLACEHOLDERS,
    exact_token_jaccard,
    normalize_question,
    question_length_without_punctuation,
    slot_counts,
    tokenize_question,
)

# 模板合并门槛:非槽 token 集合完全一致(措辞骨架相同)且实体槽计数一致 → 同模板。
# 实体槽数量不同的问句是不同的分析请求(2 只 vs 3 只股票的比较),绝不合并;
# 非槽 token 过少(如 <4 个词)时骨架太弱,不启用,避免短句误并。
_MIN_TEMPLATE_TOKENS = 4
_ANCHOR_K = 5
_SEMANTIC_ANCHOR_K = 8
_PAIR_BATCH_SIZE = 20
_FAMILY_BATCH_SIZE = 10
_FAMILY_SAMPLE_SIZE = 12

# 模板表达去重参数（默认与 DedupConfig 联动）：英文按词 n-gram、中文按字符 n-gram。
# 只取最短长度一个档位：共享更长的表达必然共享其最短长度子串，单档即可捕获
# 全部命中，且 n-gram 数量比多档少一个数量级（100k 行压力测试下仍可控）。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class DedupPairVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    label: str
    reason: str

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in {"exact_semantic", "template_duplicate", "distinct"}:
            raise ValueError(f"invalid dedup label: {value!r}")
        return value

    @field_validator("id", "reason")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("dedup id/reason must be non-empty")
        return text


class DedupBatchVerdicts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: list[DedupPairVerdict]


class TemplateFamilyVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    family_id: str
    label: str
    confidence: str
    reason: str

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in {
            "eval_template_family",
            "semantic_duplicate",
            "natural_shared_phrase",
        }:
            raise ValueError(f"invalid template family label: {value!r}")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: str) -> str:
        text = value.strip().lower()
        if text not in {"low", "medium", "high"}:
            raise ValueError("template family confidence must be low, medium or high")
        return text

    @field_validator("family_id", "reason")
    @classmethod
    def validate_family_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("template family id/reason must be non-empty")
        return text


class TemplateFamilyBatchVerdicts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: list[TemplateFamilyVerdict]


@dataclass(frozen=True)
class _Candidate:
    left: int
    right: int
    similarity: float


@dataclass(frozen=True)
class _TemplateFamilyCandidate:
    family_id: str
    members: tuple[int, ...]
    shared_phrases: tuple[str, ...]


def _row_text(row: dict[str, Any]) -> str:
    inp = row.get("input")
    return str(inp.get("text") or "") if isinstance(inp, dict) else ""


def _semantic_signature(row: dict[str, Any]) -> dict[str, Any] | None:
    meta = row.get("meta")
    signature = meta.get("semantic_signature") if isinstance(meta, dict) else None
    if not isinstance(signature, dict):
        profile = meta.get("complexity_profile") if isinstance(meta, dict) else None
        signature = profile.get("semantic_signature") if isinstance(profile, dict) else None
    if not isinstance(signature, dict):
        return None
    required = {"goal", "subject_type", "operations", "data_dimensions", "temporal_shape", "output_shape"}
    return signature if required <= set(signature) else None


def _signature_tokens(signature: dict[str, Any]) -> set[str]:
    """Tokenize signature values only; common JSON field names are not useful blockers."""
    values: list[str] = []
    for key in ("goal", "subject_type", "operations", "data_dimensions", "temporal_shape", "output_shape"):
        value = signature.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    return tokenize_question(" ".join(values), entity_slot=False)


def _signature_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _quality_key(row: dict[str, Any], index: int) -> tuple[int, int, int, int, int]:
    """Quality, naturalness, completeness, then stable source order."""
    meta = row.get("meta")
    profile = meta.get("complexity_profile") if isinstance(meta, dict) else None
    quality = profile.get("question_quality") if isinstance(profile, dict) else None
    quality_score = {"high": 2, "medium": 1, "low": 0}.get(str(quality), 0)
    signature = _semantic_signature(row)
    completeness = 0
    if signature is not None:
        completeness = sum(bool(signature.get(key)) for key in signature)
    value_profile = meta.get("value_profile") if isinstance(meta, dict) else None
    template_severity = (
        value_profile.get("template_severity") if isinstance(value_profile, dict) else "none"
    )
    naturalness = {"none": 2, "light": 1, "severe": 0}.get(str(template_severity), 0)
    # Surviving embedded-prompt rows are ranked last. The value gate normally removes them.
    prompt_free = int(not (isinstance(meta, dict) and meta.get("contains_embedded_prompt")))
    return (
        -quality_score,
        -naturalness,
        -prompt_free,
        -completeness,
        index,
    )


def _row_phrases(text: str, min_words: int, min_chars: int) -> set[str]:
    """提取问句的候选长表达（单档最短长度）。

    含中日韩字符的问句按字符 n-gram（长度 min_chars），否则按词 n-gram（长度
    min_words）。共享更长的表达必然共享其最短长度子串，因此单档即可捕获所有
    ≥min 的完全相同表达。过短文本返回空集（不参与模板表达去重）。
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if _CJK_RE.search(normalized):
        compact = re.sub(r"\s+", "", normalized)
        if len(compact) < min_chars:
            return set()
        return {compact[i : i + min_chars] for i in range(len(compact) - min_chars + 1)}
    words = normalized.split()
    if len(words) < min_words:
        return set()
    return {" ".join(words[i : i + min_words]) for i in range(len(words) - min_words + 1)}


def _phrase_family_candidates(
    rows: list[dict[str, Any]], cfg: DedupConfig
) -> list[_TemplateFamilyCandidate]:
    """Generate shared-phrase candidates without making a deletion decision."""
    phrase_members: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        for phrase in _row_phrases(
            _row_text(row), cfg.phrase_dedup_min_words, cfg.phrase_dedup_min_chars
        ):
            phrase_members.setdefault(phrase, []).append(index)
    by_members: dict[tuple[int, ...], set[str]] = {}
    for phrase, members in phrase_members.items():
        unique_members = tuple(sorted(set(members)))
        if len(unique_members) >= cfg.phrase_dedup_min_shared:
            by_members.setdefault(unique_members, set()).add(phrase)
    ordered = sorted(by_members.items(), key=lambda item: (item[0][0], -len(item[0]), item[0]))
    return [
        _TemplateFamilyCandidate(
            family_id=f"family-{index}",
            members=members,
            shared_phrases=tuple(sorted(phrases, key=lambda value: (-len(value), value))[:8]),
        )
        for index, (members, phrases) in enumerate(ordered)
    ]


def _family_payload(
    candidate: _TemplateFamilyCandidate, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    ranked = sorted(candidate.members, key=lambda index: _quality_key(rows[index], index))
    if len(ranked) <= _FAMILY_SAMPLE_SIZE:
        sampled = ranked
    else:
        edge = list(candidate.members[:2]) + list(candidate.members[-2:])
        sampled = list(dict.fromkeys(ranked[: _FAMILY_SAMPLE_SIZE - len(edge)] + edge))
    return {
        "family_id": candidate.family_id,
        "member_count": len(candidate.members),
        "shared_phrases": list(candidate.shared_phrases),
        "questions": [_row_text(rows[index]) for index in sampled],
    }


def _parse_family_batch(
    raw: str, expected_ids: set[str]
) -> dict[str, TemplateFamilyVerdict]:
    parsed = TemplateFamilyBatchVerdicts.model_validate(parse_json_object(raw))
    by_id = {item.family_id: item for item in parsed.items}
    if len(by_id) != len(parsed.items) or set(by_id) != expected_ids:
        raise ValueError("template family response ids must match the complete input batch")
    return by_id


async def review_template_families(
    rows: list[dict[str, Any]],
    cfg: DedupConfig,
    *,
    client: Any,
    llm_cfg: LLMConfig,
    cache: dict[str, dict[str, Any]],
    cache_path: Any,
    cache_lock: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
    set[int],
    set[int],
]:
    """Adjudicate shared-phrase families before any row is removed.

    Low-confidence families are retained but their hard members are returned in
    ``force_normal``. Call failures are counted so the pipeline can fail without
    publishing partial outputs.
    """
    if len(rows) < cfg.phrase_dedup_min_shared or not cfg.phrase_dedup_enabled:
        return list(rows), [], {
            "template_family_candidates": 0,
            "template_family_rejected": 0,
            "template_family_rejected_rows": 0,
            "template_family_duplicates": 0,
            "template_family_failed": 0,
        }, set(), set()
    candidates = _phrase_family_candidates(rows, cfg)
    if not candidates:
        return list(rows), [], {
            "template_family_candidates": 0,
            "template_family_rejected": 0,
            "template_family_rejected_rows": 0,
            "template_family_duplicates": 0,
            "template_family_failed": 0,
        }, set(), set()

    prompt = resolve_prompt("template_family")
    verdicts: dict[str, TemplateFamilyVerdict] = {}
    uncached: list[tuple[_TemplateFamilyCandidate, str]] = []
    for candidate in candidates:
        payload_json = json.dumps(
            _family_payload(candidate, rows),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        key = make_cache_key(payload_json, step="template_family", model=llm_cfg.model, prompt=prompt)
        cached = cache.get(key)
        if cached is not None:
            try:
                verdicts[candidate.family_id] = TemplateFamilyVerdict.model_validate(cached)
                continue
            except (ValueError, RuntimeError):
                cache.pop(key, None)
        uncached.append((candidate, key))

    batches = [
        uncached[index : index + _FAMILY_BATCH_SIZE]
        for index in range(0, len(uncached), _FAMILY_BATCH_SIZE)
    ]

    async def worker(batch: list[tuple[_TemplateFamilyCandidate, str]]) -> dict[str, Any]:
        items = [_family_payload(candidate, rows) for candidate, _ in batch]
        expected_ids = {str(item["family_id"]) for item in items}
        user_prompt = "请判定以下模板族候选，只输出严格 JSON：\n" + json.dumps(
            {"families": items}, ensure_ascii=False, separators=(",", ":")
        )
        raw = await client.complete(system_prompt=prompt, user_prompt=user_prompt)
        parsed = _parse_family_batch(raw, expected_ids)
        for candidate, key in batch:
            verdict = parsed[candidate.family_id]
            await put_cache(
                cache,
                cache_path,
                key,
                verdict.model_dump(),
                meta={
                    "step": "template_family",
                    "prompt_id": "template_family",
                    "model": llm_cfg.model,
                    "family_id": candidate.family_id,
                    "member_count": len(candidate.members),
                },
                lock=cache_lock,
            )
        return parsed

    results = await run_concurrent(batches, worker, description="template family review")
    failed = 0
    for batch, result in zip(batches, results):
        if result is None:
            failed += len(batch)
            continue
        verdicts.update(result)

    removed_eval: set[int] = set()
    rejected_family_count = 0
    duplicate_groups: list[_TemplateFamilyCandidate] = []
    force_normal: set[int] = set()
    semantic_protected: set[int] = set()
    for candidate in candidates:
        verdict = verdicts.get(candidate.family_id)
        if verdict is None:
            continue
        if verdict.confidence == "low":
            force_normal.update(candidate.members)
            semantic_protected.update(candidate.members)
        elif verdict.label == "eval_template_family":
            rejected_family_count += 1
            removed_eval.update(candidate.members)
        elif verdict.label == "semantic_duplicate":
            duplicate_groups.append(candidate)
        elif verdict.label == "natural_shared_phrase":
            semantic_protected.update(candidate.members)

    removed = set(removed_eval)
    representative_for: dict[int, int] = {}
    for candidate in duplicate_groups:
        active = [index for index in candidate.members if index not in removed]
        if len(active) < 2:
            continue
        representative = min(active, key=lambda index: _quality_key(rows[index], index))
        for index in active:
            if index != representative:
                removed.add(index)
                representative_for[index] = representative

    dropped: list[dict[str, Any]] = []
    for index in sorted(removed):
        representative = representative_for.get(index)
        dropped.append(
            {
                "source_case_id": rows[index].get("source_case_id", ""),
                "trace_id": rows[index].get("trace_id", ""),
                "text": _row_text(rows[index])[:200],
                "method": "template_family_llm",
                "decision": "semantic_duplicate" if representative is not None else "eval_template_family",
                "dedup_of_trace_id": (
                    rows[representative].get("trace_id", "") if representative is not None else ""
                ),
            }
        )
    force_normal.difference_update(removed)
    semantic_protected.difference_update(removed)
    return (
        [row for index, row in enumerate(rows) if index not in removed],
        dropped,
        {
            "template_family_candidates": len(candidates),
            "template_family_rejected": rejected_family_count,
            "template_family_rejected_rows": len(removed_eval),
            "template_family_duplicates": len(representative_for),
            "template_family_failed": failed,
        },
        {id(rows[index]) for index in force_normal},
        {id(rows[index]) for index in semantic_protected},
    )


def _parse_batch(raw: str, expected_ids: set[str]) -> dict[str, DedupPairVerdict]:
    parsed = DedupBatchVerdicts.model_validate(parse_json_object(raw))
    by_id = {item.id: item for item in parsed.items}
    if len(by_id) != len(parsed.items) or set(by_id) != expected_ids:
        raise ValueError("dedup response ids must match the complete input batch")
    return by_id


def _pair_payload(candidate: _Candidate, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": f"{candidate.left}:{candidate.right}",
        "left": {
            "question": _row_text(rows[candidate.left]),
            "semantic_signature": _semantic_signature(rows[candidate.left]),
        },
        "right": {
            "question": _row_text(rows[candidate.right]),
            "semantic_signature": _semantic_signature(rows[candidate.right]),
        },
        "candidate_similarity": round(candidate.similarity, 6),
    }


async def _judge_candidates(
    candidates: list[_Candidate],
    rows: list[dict[str, Any]],
    *,
    client: Any,
    llm_cfg: LLMConfig,
    cache: dict[str, dict[str, Any]],
    cache_path: Any,
    cache_lock: Any,
) -> tuple[dict[tuple[int, int], DedupPairVerdict], int]:
    """Judge candidate pairs in batches while caching each pair independently."""
    prompt = resolve_prompt("dedup_pair")
    verdicts: dict[tuple[int, int], DedupPairVerdict] = {}
    uncached: list[tuple[_Candidate, str, str]] = []
    for candidate in candidates:
        payload = _pair_payload(candidate, rows)
        pair_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        key = make_cache_key(pair_json, step="dedup_pair", model=llm_cfg.model, prompt=prompt)
        cached = cache.get(key)
        if cached is not None:
            try:
                verdicts[(candidate.left, candidate.right)] = DedupPairVerdict.model_validate(cached)
                continue
            except (ValueError, RuntimeError):
                cache.pop(key, None)
        uncached.append((candidate, key, pair_json))

    batches = [uncached[i : i + _PAIR_BATCH_SIZE] for i in range(0, len(uncached), _PAIR_BATCH_SIZE)]

    async def worker(batch: list[tuple[_Candidate, str, str]]) -> dict[str, Any]:
        items = [_pair_payload(candidate, rows) for candidate, _, _ in batch]
        ids = {str(item["id"]) for item in items}
        user_prompt = "请判定以下候选问句对，只输出严格 JSON：\n" + json.dumps(
            {"pairs": items}, ensure_ascii=False, separators=(",", ":")
        )
        try:
            raw = await client.complete(system_prompt=prompt, user_prompt=user_prompt)
            parsed = _parse_batch(raw, ids)
        except Exception as exc:  # noqa: BLE001 判定失败必须 fail-open 保留两条
            return {"error": str(exc)[:200], "batch": batch, "parsed": {}}
        for candidate, cache_key, _ in batch:
            item = parsed[f"{candidate.left}:{candidate.right}"]
            await put_cache(
                cache,
                cache_path,
                cache_key,
                item.model_dump(),
                meta={
                    "step": "dedup_pair",
                    "prompt_id": "dedup_pair",
                    "model": llm_cfg.model,
                    "left_trace_id": rows[candidate.left].get("trace_id", ""),
                    "right_trace_id": rows[candidate.right].get("trace_id", ""),
                },
                lock=cache_lock,
            )
        return {"error": None, "batch": batch, "parsed": parsed}

    failed = 0
    results = await run_concurrent(batches, worker, description="semantic dedup pairs")
    for batch, result in zip(batches, results):
        if result is None or result["error"] is not None:
            failed += len(batch)
            continue
        for candidate, _, _ in result["batch"]:
            verdicts[(candidate.left, candidate.right)] = result["parsed"][
                f"{candidate.left}:{candidate.right}"
            ]
    return verdicts, failed


async def semantic_dedup_rows(
    rows: list[dict[str, Any]],
    cfg: DedupConfig,
    *,
    client: Any,
    llm_cfg: LLMConfig,
    cache: dict[str, dict[str, Any]],
    cache_path: Any,
    cache_lock: Any,
    protected_row_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Template-level dedup with direct-evidence, representative-star clusters.

    Exact normalized text is merged deterministically. Signed rows use a
    semantic-signature inverted index followed by independent LLM pair verdicts.
    Rows without a signature fall back to the legacy lexical implementation.
    A row can join a cluster only when it was judged duplicate directly against
    that cluster's representative, so A≈B and B≈C never implies A≈C.
    """
    n = len(rows)
    protected = {
        index
        for index, row in enumerate(rows)
        if protected_row_ids is not None and id(row) in protected_row_ids
    }
    if n < 2:
        return list(rows), [], {
            "semantic_dedup_candidates": 0,
            "semantic_dedup_removed": 0,
            "semantic_dedup_failed": 0,
        }
    ranked = sorted(range(n), key=lambda i: _quality_key(rows[i], i))
    rank = {index: position for position, index in enumerate(ranked)}
    removed: set[int] = set()
    dropped: list[tuple[int, dict[str, Any]]] = []

    # 1) Exact normalized text: choose the best representative, then record a
    # direct representative edge for every removed row.
    exact_groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        normalized = normalize_question(_row_text(row))
        if normalized:
            exact_groups.setdefault(normalized, []).append(i)
    for members in exact_groups.values():
        if len(members) < 2:
            continue
        rep = min(members, key=lambda i: rank[i])
        for index in members:
            if index == rep:
                continue
            removed.add(index)
            dropped.append(
                (
                    index,
                    {
                        "source_case_id": rows[index].get("source_case_id", ""),
                        "trace_id": rows[index].get("trace_id", ""),
                        "text": _row_text(rows[index])[:200],
                        "dedup_of_trace_id": rows[rep].get("trace_id", ""),
                        "method": "exact_text",
                        "decision": "exact_semantic",
                        "reason": "规范化原文完全一致",
                        "candidate_similarity": 1.0,
                        "similarity": 1.0,
                        "direct_edge_similarity": 1.0,
                        "representative_similarity": 1.0,
                        "semantic_signature": _semantic_signature(rows[index]),
                        "representative_semantic_signature": _semantic_signature(rows[rep]),
                    },
                )
            )

    active = [i for i in range(n) if i not in removed]
    signatures = {i: _semantic_signature(rows[i]) for i in active}
    signed = [i for i in active if signatures[i] is not None]
    unsigned = [i for i in active if signatures[i] is None and i not in protected]

    # 2) Missing signature: explicit legacy fallback, scoped to missing rows.
    if len(unsigned) > 1:
        unsigned_rows = [rows[i] for i in unsigned]
        lexical_kept, lexical_dropped = dedup_rows(unsigned_rows, cfg)
        kept_ids = {id(row) for row in lexical_kept}
        dropped_by_trace = {str(item.get("trace_id") or ""): item for item in lexical_dropped}
        for index in unsigned:
            if id(rows[index]) in kept_ids:
                continue
            removed.add(index)
            legacy = dropped_by_trace.get(str(rows[index].get("trace_id") or ""), {})
            dropped.append(
                (
                    index,
                    {
                        **legacy,
                        "source_case_id": rows[index].get("source_case_id", ""),
                        "trace_id": rows[index].get("trace_id", ""),
                        "text": _row_text(rows[index])[:200],
                        "method": f"lexical_fallback:{legacy.get('method', 'unknown')}",
                        "semantic_signature": None,
                    },
                )
            )

    # 3) Signature candidates. Exact signature equality is included even if a
    # blocker token is absent; otherwise use rare-token inverted blocking.
    signed = [i for i in signed if i not in removed]
    token_sets = {i: _signature_tokens(signatures[i] or {}) for i in signed}
    inverted: dict[str, list[int]] = {}
    signature_groups: dict[str, list[int]] = {}
    for i in signed:
        for token in token_sets[i]:
            inverted.setdefault(token, []).append(i)
        signature_json = json.dumps(signatures[i], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        signature_groups.setdefault(signature_json, []).append(i)

    candidate_map: dict[tuple[int, int], _Candidate] = {}
    for i in signed:
        anchors = heapq.nsmallest(
            _SEMANTIC_ANCHOR_K,
            (token for token in token_sets[i] if len(inverted[token]) > 1),
            key=lambda token: len(inverted[token]),
        )
        pool = {j for token in anchors for j in inverted[token] if j != i}
        scored = sorted(
            (
                (_signature_similarity(token_sets[i], token_sets[j]), j)
                for j in pool
            ),
            key=lambda item: (-item[0], rank[item[1]]),
        )
        for similarity, j in scored[: cfg.max_candidates_per_row]:
            if similarity < cfg.semantic_candidate_threshold:
                continue
            left, right = sorted((i, j))
            if left in protected and right in protected:
                continue
            candidate_map[(left, right)] = _Candidate(left, right, similarity)
    for members in signature_groups.values():
        if len(members) < 2:
            continue
        # Exact signature groups need only direct edges to the quality-best
        # representative. Building the full clique is quadratic and violates
        # the per-row candidate cap without adding evidence for star clusters.
        rep = min(members, key=lambda index: rank[index])
        for index in members:
            if index == rep:
                continue
            a, b = (rep, index) if rep < index else (index, rep)
            if a in protected and b in protected:
                continue
            candidate_map[(a, b)] = _Candidate(a, b, 1.0)

    candidates = sorted(candidate_map.values(), key=lambda item: (item.left, item.right))
    verdicts, failed = await _judge_candidates(
        candidates,
        rows,
        client=client,
        llm_cfg=llm_cfg,
        cache=cache,
        cache_path=cache_path,
        cache_lock=cache_lock,
    )

    # Representative-star clustering: processing quality-first ensures the
    # chosen representative satisfies the declared selection policy.
    representatives: list[int] = []
    candidate_by_pair = {(c.left, c.right): c for c in candidates}
    for index in sorted(signed, key=lambda i: rank[i]):
        if index in protected:
            representatives.append(index)
            continue
        matching: list[tuple[int, DedupPairVerdict, _Candidate]] = []
        for rep in representatives:
            pair = (index, rep) if index < rep else (rep, index)
            verdict = verdicts.get(pair)
            if verdict is None or verdict.label == "distinct":
                continue
            matching.append((rep, verdict, candidate_by_pair[pair]))
        if not matching:
            representatives.append(index)
            continue
        rep, verdict, candidate = min(matching, key=lambda item: rank[item[0]])
        removed.add(index)
        dropped.append(
            (
                index,
                {
                    "source_case_id": rows[index].get("source_case_id", ""),
                    "trace_id": rows[index].get("trace_id", ""),
                    "text": _row_text(rows[index])[:200],
                    "dedup_of_trace_id": rows[rep].get("trace_id", ""),
                    "direct_pair_trace_ids": [
                        rows[index].get("trace_id", ""),
                        rows[rep].get("trace_id", ""),
                    ],
                    "method": "semantic_signature_llm",
                    "decision": verdict.label,
                    "reason": verdict.reason,
                    "candidate_similarity": candidate.similarity,
                    "similarity": candidate.similarity,
                    "direct_edge_similarity": candidate.similarity,
                    "representative_similarity": candidate.similarity,
                    "semantic_signature": signatures[index],
                    "representative_semantic_signature": signatures[rep],
                },
            )
        )

    dropped.sort(key=lambda item: item[0])
    return (
        [row for i, row in enumerate(rows) if i not in removed],
        [item for _, item in dropped],
        {
            "semantic_dedup_candidates": len(candidates),
            "semantic_dedup_removed": sum(
                1
                for _, item in dropped
                if item.get("method") in {"exact_text", "semantic_signature_llm"}
            ),
            "semantic_dedup_failed": failed,
        },
    )


def _shared_slot_counts_equal(left: dict[str, int], right: dict[str, int]) -> bool:
    """两侧共同出现的槽类型计数必须一致。

    Jaccard 层不能用全集相等：股票名词典不完备时槽化是非对称的（"nvidia" 在词典
    而 "amd" 不在 → 一侧 {<stock>:1}、另一侧 {}），硬性全集相等会漏掉本该合并的
    同型查询。但共享类型计数不一致（2 只 vs 3 只股票）仍是硬性不合并——实体数量
    不同是比较请求本身的差异。
    """
    return all(left.get(t, 0) == right.get(t, 0) for t in set(left) & set(right))


def dedup_rows(rows: list[dict[str, Any]], cfg: DedupConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop near-duplicate rows by rule layers:

    1. Template layer (equivalence classes): identical non-slot token skeleton
       AND identical per-type slot counts — "帮我分析一下<stock>的走势" vs
       "...<stock2>..." — merge directly (2-entity vs 3-entity comparisons are
       different requests and never merge). Equivalence relation → O(n).
    2. Jaccard layer: exact token-set Jaccard >= threshold on the slotted text,
       compared through an inverted index (blocking); pairs must also agree on
       the counts of their *shared* slot types, so the 2-vs-3-stock case cannot
       slip through with set-Jaccard 1.0.
    3. Pure-slot rows (no non-slot tokens): set similarity cannot distinguish
       entities ("1234" vs "5678" both collapse to {<num>}), so only exact
       normalized-text duplicates merge (method "exact_text").

    Union-find clustering and longest-representative selection are unchanged.
    """
    def _text(row: dict[str, Any]) -> str:
        inp = row.get("input")
        return str(inp.get("text") or "") if isinstance(inp, dict) else ""

    n = len(rows)
    token_sets = [tokenize_question(_text(r), entity_slot=cfg.entity_slot) for r in rows]
    non_slot = [s - SLOT_PLACEHOLDERS for s in token_sets]
    comparable = [bool(s) for s in non_slot]
    # 实体槽计数（按类型）：集合会折叠同型槽，模板层与 Jaccard 层都必须要求
    # 计数一致，否则"比较 A 和 B"与"比较 A、B 和 C"被当作同模板合并。
    slot_counts_list = [slot_counts(_text(r), entity_slot=cfg.entity_slot) for r in rows]

    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    # ---- layer 1: template equivalence classes (non-slot skeleton + slot counts) ----
    # 组键 = (非槽 token 集合, 槽位计数)。同一骨架但实体槽数量不同（如 2 只 vs
    # 3 只股票的比较）不得合并——槽计数进键，等价类按计数自动分开。
    template_groups: dict[tuple[frozenset[str], tuple[tuple[str, int], ...]], list[int]] = {}
    for i, ns in enumerate(non_slot):
        if comparable[i] and len(ns) >= _MIN_TEMPLATE_TOKENS:
            key = (frozenset(ns), tuple(sorted(slot_counts_list[i].items())))
            template_groups.setdefault(key, []).append(i)

    template_members: set[int] = set()
    group_of: dict[int, tuple[frozenset[str], tuple[tuple[str, int], ...]]] = {}
    for key, members in template_groups.items():
        if len(members) < 2:
            continue
        # 同一骨架且槽计数相同的行:实体不同 → 模板合并;槽也相同 → 完全同句,同样合并。
        # 组内全部并入最长代表(代表由最终聚簇阶段统一选择,这里只并查集)。
        for i in members[1:]:
            _union(members[0], i)
        template_members.update(members)
        for i in members:
            group_of[i] = key

    # ---- layer 2: Jaccard between representatives and singletons ----
    # 组内行已合并;组与组之间、组与单例行之间仍需近重复比较。
    # 代表 = 每组文本最长的行(与最终代表选择一致,保证并簇稳定)。
    group_reps: list[int] = []
    rep_of: dict[int, int] = {}
    for members in template_groups.values():
        if len(members) < 2:
            continue
        rep = max(members, key=lambda idx: (question_length_without_punctuation(_text(rows[idx])), -idx))
        group_reps.append(rep)
        rep_of[rep] = rep
    singletons = [i for i in range(n) if comparable[i] and i not in template_members]
    compare_items = group_reps + singletons

    # Inverted index over ALL comparable rows (not just representatives):
    # a token's candidate set must reflect every row that carries it, or the
    # rarest-token anchor becomes a singleton and misses similar partners.
    inverted: dict[str, list[int]] = {}
    for i in range(n):
        if comparable[i]:
            for tok in non_slot[i]:
                inverted.setdefault(tok, []).append(i)

    for i in compare_items:
        # selective blocking: union the k rarest shared-token candidate sets.
        # A single rarest anchor can be a private token of i's own template
        # group and would miss similar partners in other groups; a union of a
        # few rare anchors covers them while staying small in practice.
        anchors = heapq.nsmallest(
            _ANCHOR_K,
            (tok for tok in non_slot[i] if len(inverted[tok]) > 1),
            key=lambda tok: len(inverted[tok]),
        )
        for j in {idx for tok in anchors for idx in inverted[tok]}:
            if j <= i or _find(i) == _find(j):
                continue  # same template group already merged in layer 1
            # 实体槽计数不一致（如 2 只 vs 3 只股票）:即使集合 Jaccard 高也不合并——
            # 实体数量不同是比较请求本身的差异,不是"只换标的"的同模板变体。
            if not _shared_slot_counts_equal(slot_counts_list[i], slot_counts_list[j]):
                continue
            a, b = len(token_sets[i]), len(token_sets[j])
            # 尺寸界预过滤:J >= t 要求交集 >= t(a+b)/(1+t),而交集 <= min(a,b)。
            if min(a, b) * (1 + cfg.threshold) < cfg.threshold * (a + b):
                continue
            if exact_token_jaccard(token_sets[i], token_sets[j]) >= cfg.threshold:
                _union(i, j)

    # ---- 纯槽位行查重（exact-text）----
    # non_slot 为空的行集合近似无法区分实体："1234" 与 "5678" 都折叠成 {<num>}、
    # Jaccard=1.0，但显然是不同查询，绝不能合并；只有原文（规范化后）完全相同
    # 才视为重复。空文本不参与（空行永不丢弃）。
    exact_text_groups: dict[str, list[int]] = {}
    for i in range(n):
        if comparable[i]:
            continue
        text = normalize_question(_text(rows[i]))
        if text:
            exact_text_groups.setdefault(text, []).append(i)
    for members in exact_text_groups.values():
        if len(members) < 2:
            continue
        for i in members[1:]:
            _union(members[0], i)

    # ---- cluster -> drop non-representatives ----
    groups: dict[int, list[int]] = {}
    for i in range(n):
        if comparable[i]:
            groups.setdefault(_find(i), []).append(i)
    for members in exact_text_groups.values():
        if len(members) < 2:
            continue
        groups.setdefault(_find(members[0]), []).extend(members)

    dropped_indices: set[int] = set()
    dropped: list[tuple[int, dict[str, Any]]] = []
    for indices in groups.values():
        if len(indices) == 1:
            continue
        rep = max(indices, key=lambda idx: (question_length_without_punctuation(_text(rows[idx])), -idx))
        for idx in indices:
            if idx == rep:
                continue
            dropped_indices.add(idx)
            sim = exact_token_jaccard(token_sets[idx], token_sets[rep])
            method = (
                "template_merge"
                if group_of.get(idx) is not None and group_of.get(idx) == group_of.get(rep)
                else "exact_text"
                if not comparable[idx]
                else "token_jaccard"
            )
            dropped.append(
                (
                    idx,
                    {
                        "source_case_id": rows[idx].get("source_case_id", ""),
                        "trace_id": rows[idx].get("trace_id", ""),
                        "text": (rows[idx].get("input") or {}).get("text", "")[:200],
                        "dedup_of_trace_id": rows[rep].get("trace_id", ""),
                        "similarity": sim,
                        "method": method,
                    },
                )
            )

    dropped.sort(key=lambda pair: pair[0])
    # 没被删的行(含不可比/空文本、单例组)都在 kept,保持原索引顺序。
    kept = [rows[i] for i in range(n) if i not in dropped_indices]
    return kept, [d for _, d in dropped]
