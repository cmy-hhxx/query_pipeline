from __future__ import annotations

import heapq
from typing import Any

from query_pipeline.config.models import DedupConfig
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
