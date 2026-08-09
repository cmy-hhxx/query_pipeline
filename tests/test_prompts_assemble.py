"""Runtime prompt assembly from templates/*.md (no LLM involved)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_pipeline.prompts import resolve_prompt
from query_pipeline.prompts.assemble import (
    build_complex_classify_prompt,
    build_normal_classify_prompt,
    build_verify_prompt,
    parse_bad_cases,
    parse_complex_few_shot,
    parse_normal_few_shot,
)

_COMPLEX_SAMPLE = """\
## 9 类复杂金融问句

### 01 复杂取数计算类
特征：需要多指标组合筛选或回测策略。核心是「算」。

所有例子：
1. 回测一个基于5周均线的短线择时策略。
2. 帮我筛选满足条件的股票。

### 02 预测类
特征：以时间维度推演为核心。

典型例子：
- 中际旭创未来1年股价能涨到多少？
"""

_NORMAL_SAMPLE = """\
## 01-event-and-concept-stock-selection | 事件与概念选股

定义：事件驱动的选股任务。

适用场景：
- 用户要求找出受益股票。

排除场景：
- 普通行情查询。

边界规则：
- 01 的核心是事件到标的的映射。

易混类别：
- 03-stock-diagnosis-and-data-lookup: 03 是查数。

## 决策步骤

1. 阅读 query。
"""

_BAD_SAMPLE = """\
结合市场信息分析一下605499股票走势 -- 仅为单项行情
科技硬件、涨价链各推5只个股 -- 仅按显式条件列名单
给我整理pcb的龙头企业，要求财务状态好，订单多
以及 问句复制 AI 的输入，这个也要删
看下来几个点吧
1、简单取数计算的，这个务必限制；
"""


class ParseComplexTest(unittest.TestCase):
    def test_parses_headers_and_examples(self) -> None:
        specs = parse_complex_few_shot(_COMPLEX_SAMPLE)
        self.assertEqual(set(specs), {"01", "02"})
        self.assertIn("多指标组合筛选", specs["01"].definition)
        self.assertEqual(specs["01"].examples, ("回测一个基于5周均线的短线择时策略。", "帮我筛选满足条件的股票。"))
        self.assertEqual(specs["02"].examples, ("中际旭创未来1年股价能涨到多少？",))

    def test_real_file_has_9_specs_with_content(self) -> None:
        text = Path(ROOT / "templates" / "complex_few_shot.md").read_text(encoding="utf-8")
        specs = parse_complex_few_shot(text)
        self.assertEqual(len(specs), 9)
        for spec in specs.values():
            self.assertTrue(spec.definition, f"category {spec.id} missing definition")
            self.assertTrue(spec.examples, f"category {spec.id} missing examples")


class ParseNormalTest(unittest.TestCase):
    def test_parses_sections(self) -> None:
        specs = parse_normal_few_shot(_NORMAL_SAMPLE)
        self.assertEqual(set(specs), {"01"})
        spec = specs["01"]
        self.assertEqual(spec.slug, "event-and-concept-stock-selection")
        self.assertEqual(spec.name, "事件与概念选股")
        self.assertEqual(set(spec.sections), {"定义", "适用场景", "排除场景", "边界规则", "易混类别"})
        self.assertIn("事件驱动的选股任务", spec.sections["定义"])

    def test_real_file_has_16_specs_with_all_sections(self) -> None:
        text = Path(ROOT / "templates" / "normal_few_shot.md").read_text(encoding="utf-8")
        specs = parse_normal_few_shot(text)
        self.assertEqual(len(specs), 16)
        for spec in specs.values():
            self.assertIn("定义", spec.sections)
            self.assertIn("易混类别", spec.sections)


class BadCasesTest(unittest.TestCase):
    def test_annotation_lines_skipped(self) -> None:
        cases = parse_bad_cases(_BAD_SAMPLE)
        self.assertEqual(
            cases,
            (
                "结合市场信息分析一下605499股票走势",
                "科技硬件、涨价链各推5只个股",
                "给我整理pcb的龙头企业，要求财务状态好，订单多",
            ),
        )

    def test_real_file_has_negative_examples(self) -> None:
        text = Path(ROOT / "templates" / "bad_cases_for_complex.md").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(parse_bad_cases(text)), 10)


class BuildPromptTest(unittest.TestCase):
    def test_complex_classify_prompt_embeds_taxonomy(self) -> None:
        prompt = build_complex_classify_prompt()
        self.assertIn("complex-topic/01-data-metrics-calculation", prompt)
        self.assertIn("category_id", prompt)
        self.assertIn("复杂取数计算", prompt)

    def test_normal_classify_prompt_embeds_taxonomy(self) -> None:
        prompt = build_normal_classify_prompt()
        self.assertIn("01-event-and-concept-stock-selection", prompt)
        self.assertIn("易混类别", prompt)
        self.assertIn("16-macro-information-qa", prompt)

    def test_verify_prompt_injects_bad_cases(self) -> None:
        base = "判断是否复杂。"
        injected = build_verify_prompt(base)
        self.assertIn(base, injected)
        self.assertIn("已被确认为不复杂", injected)
        self.assertIn("605499", injected)

    def test_registry_serves_assembled_prompts(self) -> None:
        self.assertIn("complex-topic/", resolve_prompt("classify_complex"))
        self.assertIn("易混类别", resolve_prompt("classify_normal"))
        self.assertIn("已被确认为不复杂", resolve_prompt("verify_complex"))
        self.assertIn("已被确认为不复杂", resolve_prompt("verify_recheck"))


if __name__ == "__main__":
    unittest.main()
