from __future__ import annotations

from typing import Any

from query_pipeline.config.models import DedupConfig
from query_pipeline.rules.normalize import (
    SLOT_PLACEHOLDERS,
    exact_token_jaccard,
    question_length_without_punctuation,
    tokenize_question,
)


def dedup_rows(rows: list[dict[str, Any]], cfg: DedupConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop near-duplicate rows by exact token-set Jaccard on the slotted text.

    实体槽化使"同模板不同实体"的查询骨架一致(按意图/模板合并);token 集合次序
    无关,能抓住同句改写。分组用并查集(传递),每组保留最长最完整的代表。
    Dropped rows carry provenance (representative trace_id + similarity + method)
    for the debug file. Returns (kept, dropped).
    """
    def _text(row: dict[str, Any]) -> str:
        inp = row.get("input")
        return str(inp.get("text") or "") if isinstance(inp, dict) else ""

    n = len(rows)
    token_sets = [tokenize_question(_text(r), entity_slot=cfg.entity_slot) for r in rows]
    # 仅含槽位 token 的退化查询(如纯数字)不参与比较,避免 "1234" vs "5678" 都变 {<num>} 而误并。
    comparable = [bool(s and (s - SLOT_PLACEHOLDERS)) for s in token_sets]

    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        if not comparable[i]:
            continue
        for j in range(i + 1, n):
            if not comparable[j]:
                continue
            a, b = len(token_sets[i]), len(token_sets[j])
            # 尺寸界预过滤:J >= t 要求交集 >= t(a+b)/(1+t),而交集 <= min(a,b)。
            if min(a, b) * (1 + cfg.threshold) < cfg.threshold * (a + b):
                continue
            if exact_token_jaccard(token_sets[i], token_sets[j]) >= cfg.threshold:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        if comparable[i]:
            groups.setdefault(_find(i), []).append(i)

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
            dropped.append(
                (
                    idx,
                    {
                        "source_case_id": rows[idx].get("source_case_id", ""),
                        "trace_id": rows[idx].get("trace_id", ""),
                        "text": (rows[idx].get("input") or {}).get("text", "")[:200],
                        "dedup_of_trace_id": rows[rep].get("trace_id", ""),
                        "similarity": exact_token_jaccard(token_sets[idx], token_sets[rep]),
                        "method": f"token_jaccard_threshold_{cfg.threshold}",
                    },
                )
            )

    dropped.sort(key=lambda pair: pair[0])
    # 没被删的行(含不可比/空文本、单例组)都在 kept,保持原索引顺序。
    kept = [rows[i] for i in range(n) if i not in dropped_indices]
    return kept, [d for _, d in dropped]
