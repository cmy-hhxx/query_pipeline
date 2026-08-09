from __future__ import annotations

import heapq
from typing import Any

from query_pipeline.config.models import DedupConfig
from query_pipeline.rules.normalize import (
    SLOT_PLACEHOLDERS,
    exact_token_jaccard,
    question_length_without_punctuation,
    tokenize_question,
)

# 模板合并门槛:非槽 token 集合完全一致(措辞骨架相同)且实体不同 → 同模板。
# 非槽 token 过少(如 <4 个词)时骨架太弱,不启用,避免短句误并。
_MIN_TEMPLATE_TOKENS = 4
_ANCHOR_K = 5


def dedup_rows(rows: list[dict[str, Any]], cfg: DedupConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop near-duplicate rows by two rule layers:

    1. Template layer (equivalence classes): identical non-slot token skeleton
       with differing entity slots — "帮我分析一下<stock>的走势" vs
       "...<stock2>..." — merge directly. This is an equivalence relation, so
       it is clustered in O(n) without pairwise comparison.
    2. Jaccard layer: exact token-set Jaccard >= threshold on the slotted
       text, compared between template-group representatives and singleton
       rows through an inverted index (blocking), so ~100k rows stay fast.

    Union-find clustering and longest-representative selection are unchanged.
    """
    def _text(row: dict[str, Any]) -> str:
        inp = row.get("input")
        return str(inp.get("text") or "") if isinstance(inp, dict) else ""

    n = len(rows)
    token_sets = [tokenize_question(_text(r), entity_slot=cfg.entity_slot) for r in rows]
    non_slot = [s - SLOT_PLACEHOLDERS for s in token_sets]
    comparable = [bool(s) for s in non_slot]

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

    # ---- layer 1: template equivalence classes (non-slot skeleton) ----
    template_groups: dict[frozenset[str], list[int]] = {}
    for i, ns in enumerate(non_slot):
        if comparable[i] and len(ns) >= _MIN_TEMPLATE_TOKENS:
            template_groups.setdefault(frozenset(ns), []).append(i)

    template_members: set[int] = set()
    group_of: dict[int, frozenset[str]] = {}
    for key, members in template_groups.items():
        if len(members) < 2:
            continue
        # 同一骨架的行:实体槽不同 → 模板合并;槽也相同 → 完全同句,同样合并。
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
            a, b = len(token_sets[i]), len(token_sets[j])
            # 尺寸界预过滤:J >= t 要求交集 >= t(a+b)/(1+t),而交集 <= min(a,b)。
            if min(a, b) * (1 + cfg.threshold) < cfg.threshold * (a + b):
                continue
            if exact_token_jaccard(token_sets[i], token_sets[j]) >= cfg.threshold:
                _union(i, j)

    # ---- cluster -> drop non-representatives ----
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
            sim = exact_token_jaccard(token_sets[idx], token_sets[rep])
            method = (
                "template_merge"
                if group_of.get(idx) is not None and group_of.get(idx) == group_of.get(rep)
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
