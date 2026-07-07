from __future__ import annotations

from datasketch import MinHash

from query_pipeline.rules.normalize import normalize_question


def char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def build_minhash(text: str, *, num_perm: int) -> MinHash:
    normalized = normalize_question(text).lower()
    shingles = char_ngrams(normalized, 3)
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh
