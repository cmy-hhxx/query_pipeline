from __future__ import annotations

import random
from typing import Any

from query_pipeline.config.models import DedupConfig
from query_pipeline.rules.normalize import normalize_question

_PRIME = (1 << 61) - 1  # Mersenne prime for permutation hashing
_NUM_BANDS = 16  # LSH bands; rows per band = num_perm // _NUM_BANDS
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_FNV_MASK = (1 << 64) - 1

# Fixed random permutations shared by every signature (deterministic across runs).
_rng = random.Random(42)
_PERM_CACHE: dict[int, tuple[list[int], list[int]]] = {}


def _fnv1a(text: str) -> int:
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & _FNV_MASK
    return h


def _shingles(text: str, n_gram: int) -> set[str]:
    if len(text) <= n_gram:
        return {text} if text else set()
    return {text[i : i + n_gram] for i in range(len(text) - n_gram + 1)}


def _permutations(num_perm: int) -> tuple[list[int], list[int]]:
    cached = _PERM_CACHE.get(num_perm)
    if cached is None:
        a = [_rng.randrange(1, _PRIME) for _ in range(num_perm)]
        b = [_rng.randrange(0, _PRIME) for _ in range(num_perm)]
        cached = (a, b)
        _PERM_CACHE[num_perm] = cached
    return cached


def minhash_signature(text: str, *, n_gram: int, num_perm: int) -> list[int] | None:
    """MinHash signature over character n-grams of the normalized text.

    Returns None for empty text so it can never collide with a real query.
    """
    normalized = normalize_question(text).lower()
    shingles = _shingles(normalized, n_gram)
    if not shingles:
        return None
    a, b = _permutations(num_perm)
    signature = [_PRIME] * num_perm
    for shingle in shingles:
        h = _fnv1a(shingle)
        for i in range(num_perm):
            permuted = (a[i] * h + b[i]) % _PRIME
            if permuted < signature[i]:
                signature[i] = permuted
    return signature


def jaccard_estimate(left: list[int], right: list[int]) -> float:
    """Fraction of signature positions that match — unbiased Jaccard estimate."""
    if len(left) != len(right):
        return 0.0
    return sum(1 for a, b in zip(left, right) if a == b) / len(left)


def _band_keys(signature: list[int]) -> list[tuple[int, ...]]:
    rows = max(1, len(signature) // _NUM_BANDS)
    return [tuple(signature[b * rows : (b + 1) * rows]) for b in range(_NUM_BANDS) if b * rows < len(signature)]


def dedup_rows(rows: list[dict[str, Any]], cfg: DedupConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop near-duplicate rows by MinHash Jaccard, keeping the first occurrence.

    LSH banding only pairs rows sharing a signature slice, so the comparison
    count stays near-linear instead of O(n^2). Dropped rows carry provenance
    (representative trace_id + similarity + method) for the debug file.
    Returns (kept, dropped).
    """
    signatures = [
        minhash_signature(r.get("input", {}).get("text", "") if isinstance(r.get("input"), dict) else "", n_gram=cfg.n_gram, num_perm=cfg.num_perm)
        for r in rows
    ]
    buckets: dict[tuple[int, ...], list[int]] = {}
    for index, signature in enumerate(signatures):
        if signature is None:
            continue
        for key in _band_keys(signature):
            buckets.setdefault(key, []).append(index)

    kept_indices: set[int] = set()
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        signature = signatures[index]
        dup_of: int | None = None
        dup_similarity = 0.0
        if signature is not None:
            candidates = {k for key in _band_keys(signature) for k in buckets.get(key, ()) if k < index}
            for candidate in sorted(candidates):
                if candidate not in kept_indices:
                    continue
                candidate_signature = signatures[candidate]
                if candidate_signature is None:
                    continue  # unreachable: only non-None signatures enter buckets
                similarity = jaccard_estimate(signature, candidate_signature)
                if similarity >= cfg.threshold:
                    dup_of = candidate
                    dup_similarity = similarity
                    break
        if dup_of is None:
            kept_indices.add(index)
            kept.append(row)
        else:
            dropped.append(
                {
                    "source_case_id": row.get("source_case_id", ""),
                    "trace_id": row.get("trace_id", ""),
                    "text": (row.get("input") or {}).get("text", "")[:200],
                    "dedup_of_trace_id": rows[dup_of].get("trace_id", ""),
                    "similarity": dup_similarity,
                    "method": f"minhash_threshold_{cfg.threshold}",
                }
            )
    return kept, dropped
