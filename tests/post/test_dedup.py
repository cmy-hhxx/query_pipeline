"""Dedup: exact/near/template/rephrase merging, entity slotting, blocking scale."""

from __future__ import annotations

import time
import unittest
from typing import Any

from query_pipeline.config.models import DedupConfig
from query_pipeline.post.dedup import dedup_rows
from query_pipeline.rules.normalize import exact_token_jaccard, tokenize_question

_BASE = "how does the fed rate hike affect bond prices this year and what should investors watch for"
_NEAR = _BASE.replace("affect", "impact")
_DISTINCT = "compare vanguard and fidelity etf expense ratios"

def _row(text: str, trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "source_case_id": "t1",
        "input": {"text": text, "image": "", "file": ""},
        "meta": {"reason": "r"},
    }

class DedupTest(unittest.TestCase):
    def test_exact_and_near_duplicates_dropped(self) -> None:
        rows = [
            _row(_BASE, "r1"),
            _row(_BASE, "r2"),  # exact duplicate
            _row(_NEAR, "r3"),  # near duplicate (word swap)
            _row(_DISTINCT, "r4"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())

        self.assertEqual([r["trace_id"] for r in kept], ["r1", "r4"])
        self.assertEqual([d["trace_id"] for d in dropped], ["r2", "r3"])
        self.assertEqual(dropped[0]["dedup_of_trace_id"], "r1")
        self.assertEqual(dropped[1]["dedup_of_trace_id"], "r1")
        self.assertEqual(dropped[0]["similarity"], 1.0)
        self.assertGreaterEqual(dropped[1]["similarity"], 0.8)
        self.assertEqual(dropped[0]["method"], "template_merge")  # identical skeleton group

    def test_near_duplicate_survives_high_threshold(self) -> None:
        kept, dropped = dedup_rows([_row(_BASE, "r1"), _row(_NEAR, "r2")], DedupConfig(threshold=0.99))
        self.assertEqual([r["trace_id"] for r in kept], ["r1", "r2"])
        self.assertEqual(dropped, [])

    def test_distinct_queries_kept(self) -> None:
        rows = [_row(_DISTINCT, "r1"), _row(_BASE, "r2")]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_empty_text_never_dropped(self) -> None:
        rows = [_row("", "r1"), _row("", "r2"), _row("   ", "r3")]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 3)
        self.assertEqual(dropped, [])

    def test_tokenization_deterministic(self) -> None:
        a = tokenize_question(_BASE)
        b = tokenize_question(_BASE)
        self.assertEqual(a, b)
        self.assertEqual(exact_token_jaccard(a, b), 1.0)

    def test_threshold_range_validated(self) -> None:
        with self.assertRaises(ValueError):
            DedupConfig(threshold=1.5)
        with self.assertRaises(ValueError):
            DedupConfig(threshold=-0.1)

    def test_near_duplicate_similarity_above_default_threshold(self) -> None:
        sim = exact_token_jaccard(tokenize_question(_BASE), tokenize_question(_NEAR))
        self.assertGreaterEqual(sim, 0.8)

    def test_template_variants_merged(self) -> None:
        rows = [
            _row("Forecast $EYE for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios", "r1"),
            _row("Forecast $AA for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios", "r2"),
            _row("Forecast $NVDA for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios", "r3"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 2)
        self.assertEqual({d["dedup_of_trace_id"] for d in dropped}, {kept[0]["trace_id"]})
        self.assertEqual(dropped[0]["similarity"], 1.0)

    def test_rephrase_merged(self) -> None:
        a = "Check NEARUSDT for the rest of today on 4hr time frame in futures market and recommend as to entry/exit points"
        b = "How is NEARUSDT going to perform for the rest of today on 4hr time frame in futures market and recommend as to entry/exit points"
        kept, dropped = dedup_rows([_row(a, "r1"), _row(b, "r2")], DedupConfig())
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        # 代表选更长更完整的改写版本(r2),被删的指向它。
        self.assertEqual(kept[0]["trace_id"], "r2")
        self.assertEqual(dropped[0]["dedup_of_trace_id"], "r2")
        self.assertGreaterEqual(dropped[0]["similarity"], 0.8)

    def test_distinct_intent_kept(self) -> None:
        rows = [
            _row("Forecast $AAPL for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios with confidence levels", "r1"),
            _row("Forecast NVDA trend this week", "r2"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_content_word_not_slotted(self) -> None:
        a = "The price is stalled 1955 now, Do again fundamental analysis [strong/weak], do sentiment/catalyst/outlook analysis [positive/negative] and Do chart analysis on indonesian stock IDX:WIFI chart and give"
        b = "The price is rallying now, Do again fundamental analysis [strong/weak], do sentiment/catalyst/outlook analysis [positive/negative] and Do chart analysis on indonesian stock IDX:CBDK chart and give me "
        # 真实语料中该对 J=0.786<0.8:stalled/rallying 是内容词,不被槽化,保持分开。
        ta = tokenize_question(a)
        tb = tokenize_question(b)
        self.assertIn("stalled", ta)
        self.assertIn("rallying", tb)
        self.assertNotEqual(ta, tb)
        kept, dropped = dedup_rows([_row(a, "r1"), _row(b, "r2")], DedupConfig())
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_entity_slotting_switch(self) -> None:
        a = "What is the P/E of $AAPL at 100"
        b = "What is the P/E of $MSFT at 200"
        rows = [_row(a, "r1"), _row(b, "r2")]
        kept, dropped = dedup_rows(rows, DedupConfig(entity_slot=True))
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        kept2, dropped2 = dedup_rows(rows, DedupConfig(entity_slot=False))
        self.assertEqual(len(kept2), 2)
        self.assertEqual(dropped2, [])

    def test_number_only_difference_merged(self) -> None:
        rows = [
            _row("The price is stalled 1955 now, do again fundamental analysis on this stock", "r1"),
            _row("The price is stalled 875 now, do again fundamental analysis on this stock", "r2"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped[0]["similarity"], 1.0)

    def test_representative_selection_and_determinism(self) -> None:
        rows = [
            _row("Forecast  SNDK for the next 1 day, — bull / base / bear scenarios with confidence levels and what would trigger each.", "r1"),
            _row("Forecast $AAPL for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios with confidence levels and what would trigger each.", "r2"),
            _row("Forecast $MSFT for the next 1 day, 1 week, and 1 month — bull / base / bear scenarios with confidence levels and what would trigger each.", "r3"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["trace_id"], "r2")
        self.assertEqual({d["dedup_of_trace_id"] for d in dropped}, {"r2"})
        kept2, dropped2 = dedup_rows(rows, DedupConfig())
        self.assertEqual([r["trace_id"] for r in kept2], [r["trace_id"] for r in kept])
        self.assertEqual([d["trace_id"] for d in dropped2], [d["trace_id"] for d in dropped])

    def test_cjk_near_duplicate_merged(self) -> None:
        rows = [
            _row("帮我分析一下nvidia的估值并给出建议", "r1"),
            _row("帮我分析一下amd的估值并给出建议", "r2"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)

    def test_pure_slot_query_never_dropped(self) -> None:
        rows = [_row("1234", "r1"), _row("5678", "r2")]
        kept, dropped = dedup_rows(rows, DedupConfig())
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_chained_pair_merged_transitively(self) -> None:
        # A~B (8/9) and B~C (0.8) both >= threshold; A~C (0.7) below it.
        # Transitive union-find merges all three into one cluster, so the directly
        # >= threshold pair B~C does NOT both survive. One representative stays
        # (longest text: C, 19 chars vs B's 18); the other two are dropped.
        a = "w1 w2 w3 w4 w5 w6 w7 w8"
        b = "w1 w2 w3 w4 w5 w6 w7 w8 w9"
        c = "w1 w2 w3 w4 w5 w6 w7 w9 w10"
        rows = [_row(a, "rA"), _row(b, "rB"), _row(c, "rC")]
        kept, dropped = dedup_rows(rows, DedupConfig())

        self.assertEqual([r["trace_id"] for r in kept], ["rC"])
        self.assertEqual([d["trace_id"] for d in dropped], ["rA", "rB"])
        self.assertEqual({d["dedup_of_trace_id"] for d in dropped}, {"rC"})
        # dropped[1] is B, a direct >= threshold near-dup of representative C.
        self.assertGreaterEqual(dropped[1]["similarity"], 0.8)

class StockNameSlottingTest(unittest.TestCase):
    def test_chinese_stock_swap_merged(self) -> None:
        rows = [
            _row("帮我分析一下贵州茅台的走势，结合基本面和技术面给出操作建议", "a"),
            _row("帮我分析一下宁德时代的走势，结合基本面和技术面给出操作建议", "b"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig(enabled=True, threshold=0.80))
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped[0]["trace_id"], "b")
        self.assertEqual(dropped[0]["method"], "template_merge")

    def test_english_stock_swap_merged(self) -> None:
        rows = [
            _row("Analyze nvidia valuation and give buy/sell advice", "a"),
            _row("Analyze microsoft valuation and give buy/sell advice", "b"),
        ]
        kept, dropped = dedup_rows(rows, DedupConfig(enabled=True, threshold=0.80))
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped[0]["method"], "template_merge")

    def test_different_skeleton_kept(self) -> None:
        rows = [
            _row("帮我分析一下贵州茅台的走势，结合基本面和技术面给出操作建议", "a"),
            _row("帮我分析一下宁德时代的资金流向和筹码分布", "b"),
        ]
        kept, _ = dedup_rows(rows, DedupConfig(enabled=True, threshold=0.80))
        self.assertEqual(len(kept), 2)

    def test_same_skeleton_no_entities_kept(self) -> None:
        # identical sentences share the template group and merge
        rows = [_row("帮我分析一下贵州茅台的走势", "a"), _row("帮我分析一下贵州茅台的走势", "b")]
        kept, dropped = dedup_rows(rows, DedupConfig(enabled=True, threshold=0.80))
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped[0]["method"], "template_merge")

    def test_weak_skeleton_not_merged(self) -> None:
        # fewer than 4 non-slot tokens: template rule disabled
        rows = [
            _row("A 怎么样", "a"),
            _row("B 怎么样", "b"),
        ]
        kept, _ = dedup_rows(rows, DedupConfig(enabled=True, threshold=0.80))
        self.assertEqual(len(kept), 2)

class BlockingScaleTest(unittest.TestCase):
    def test_100k_rows_fast(self) -> None:
        # 10k templates x 10 entity variants = 100k rows; every variant of a
        # template must merge, and the inverted-index blocking must keep it fast.
        entities = ["贵州茅台", "宁德时代", "比亚迪", "五粮液", "中际旭创", "长电科技", "东方财富", "紫金矿业", "天齐锂业", "隆基绿能"]
        # 10k distinct templates: each carries 5 unique literal tokens (alpha{t}..),
        # keeping pairwise Jaccard below the 0.80 threshold (otherwise the Jaccard
        # layer legitimately merges them — the test must isolate the template layer).
        rows: list[dict] = []
        for t in range(10_000):
            unique = " ".join(f"{w}{t}" for w in ("alpha", "beta", "gamma", "delta", "eps"))
            for e in entities:
                rows.append(_row(f"帮我分析一下{e}的走势，{unique}和技术面给出操作建议", f"t{t}_{e}"))
        self.assertEqual(len(rows), 100_000)
        start = time.monotonic()
        kept, dropped = dedup_rows(rows, DedupConfig(enabled=True, threshold=0.80))
        elapsed = time.monotonic() - start
        # each 10-variant template group keeps 1 representative
        self.assertEqual(len(kept), 10_000)
        self.assertEqual(len(dropped), 90_000)
        self.assertLess(elapsed, 15.0, f"dedup took {elapsed:.1f}s for 100k rows")

if __name__ == "__main__":
    unittest.main()
