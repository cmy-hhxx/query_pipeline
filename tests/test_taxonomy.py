"""Taxonomy parsing — single source of truth is templates/categories.md."""

from __future__ import annotations

import unittest

from query_pipeline.taxonomy import COMPLEX_PREFIX, load_taxonomy, parse_categories

_SAMPLE = """\
01    complex-topic/01-data-metrics-calculation    复杂取数计算
02    complex-topic/02-forecasting-and-projection    预测类
07    complex-topic/07-strategy-evaluation    策略触发任务类
01-event-and-concept-stock-selection    事件与概念选股
14-complex-stock-selection    复杂选股
16-macro-information-qa    宏观信息问答
"""

class TaxonomyTest(unittest.TestCase):
    def test_parse_complex_and_normal(self) -> None:
        tax = parse_categories(_SAMPLE)
        self.assertEqual(len(tax.complex), 3)
        self.assertEqual(len(tax.normal), 3)
        c01 = tax.complex["01"]
        self.assertEqual(c01.id, "01")
        self.assertEqual(c01.slug, "data-metrics-calculation")
        self.assertEqual(c01.name, "复杂取数计算")
        self.assertEqual(c01.difficulty, "hard")
        self.assertEqual(c01.path, "complex-topic/01-data-metrics-calculation")

    def test_normal_category_path_has_no_prefix(self) -> None:
        n01 = parse_categories(_SAMPLE).normal["01"]
        self.assertEqual(n01.path, "01-event-and-concept-stock-selection")
        self.assertEqual(n01.difficulty, "normal")

    def test_historical_slug_07_preserved(self) -> None:
        tax = parse_categories(_SAMPLE)
        self.assertEqual(tax.complex["07"].slug, "strategy-evaluation")
        self.assertEqual(tax.complex["07"].path, "complex-topic/07-strategy-evaluation")

    def test_real_file_loads_9_and_16(self) -> None:
        tax = load_taxonomy()
        self.assertEqual(len(tax.complex), 9)
        self.assertEqual(len(tax.normal), 16)
        # every complex slug must match the fin_bench directory name (join key)
        for cat in tax.complex.values():
            self.assertTrue(cat.path.startswith(COMPLEX_PREFIX))

    def test_duplicate_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_categories("01    complex-topic/01-a    甲\n01    complex-topic/01-b    乙\n01-x   丙")

    def test_missing_normal_set_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_categories("01    complex-topic/01-a    甲")

    def test_unparseable_line_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_categories("garbage line without structure")

    def test_blank_and_comment_lines_skipped(self) -> None:
        tax = parse_categories("# 注释\n\n01    complex-topic/01-a    甲\n\n01-x    乙")
        self.assertEqual(len(tax.complex), 1)
        self.assertEqual(len(tax.normal), 1)

    def test_get_by_difficulty(self) -> None:
        tax = parse_categories(_SAMPLE)
        self.assertEqual(tax.get("hard", "01").slug, "data-metrics-calculation")
        self.assertEqual(tax.get("normal", "14").slug, "complex-stock-selection")
        with self.assertRaises(KeyError):
            tax.get("normal", "02")  # id exists only in complex set

if __name__ == "__main__":
    unittest.main()
