"""Enhanced dedup: stock-name slotting, template merge, blocking scale."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.config.models import DedupConfig
from query_pipeline.post.dedup import dedup_rows


def _row(text: str, trace_id: str) -> dict:
    return {"source_case_id": "s", "trace_id": trace_id, "input": {"text": text}}


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
