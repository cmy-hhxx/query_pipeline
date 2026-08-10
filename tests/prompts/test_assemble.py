"""Runtime prompt assembly from templates/*.md (no LLM involved)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (tests/prompts/ -> project root)

from query_pipeline.prompts import PROMPTS, resolve_prompt
from query_pipeline.taxonomy import templates_dir
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
        text = (templates_dir() / "complex_few_shot.md").read_text(encoding="utf-8")
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
        text = (templates_dir() / "normal_few_shot.md").read_text(encoding="utf-8")
        specs = parse_normal_few_shot(text)
        self.assertEqual(len(specs), 16)
        for spec in specs.values():
            self.assertIn("定义", spec.sections)
            self.assertIn("易混类别", spec.sections)

    def test_real_file_no_pollution_after_doc_heading(self) -> None:
        # 回归：`# 决策步骤` 之后的正文曾静默并入类别 16 的"易混类别"，
        # 生产 classify prompt 因此混入无关后处理指令。解析真实模板必须零污染。
        text = (templates_dir() / "normal_few_shot.md").read_text(encoding="utf-8")
        specs = parse_normal_few_shot(text)
        cat16 = specs["16"].sections["易混类别"]
        self.assertNotIn("needs_review", cat16)
        self.assertNotIn("决策步骤", cat16)
        self.assertNotIn("阅读 query", cat16)

    def test_doc_heading_terminates_current_category(self) -> None:
        # `#` 级标题终结当前类别（save），后续新类别头正常开启。
        specs = parse_normal_few_shot(
            "## 01-good | 名字\n定义：a\n# 中间文档标题\n## 02-better | 名字\n定义：b"
        )
        self.assertEqual(set(specs), {"01", "02"})
        self.assertEqual(specs["01"].sections["定义"], "a")
        self.assertEqual(specs["02"].sections["定义"], "b")

    def test_body_after_doc_heading_raises(self) -> None:
        # 文档级标题之后的正文不得静默并入上一类别（fail-loud）。
        from query_pipeline.prompts.assemble import parse_normal_few_shot

        with self.assertRaises(ValueError):
            parse_normal_few_shot("## 01-good | 名字\n定义：a\n# 决策步骤\n1. 阅读 query")

class FailLoudTest(unittest.TestCase):
    def test_malformed_normal_header_raises(self) -> None:
        # 缺 slug/name 的 `## 02`：必须抛错，不得静默并入前一类别
        from query_pipeline.prompts.assemble import parse_normal_few_shot

        with self.assertRaises(ValueError):
            parse_normal_few_shot("## 01-good | 名字\n定义：a\n## 02\n定义：b")

    def test_malformed_complex_header_raises(self) -> None:
        from query_pipeline.prompts.assemble import parse_complex_few_shot

        with self.assertRaises(ValueError):
            parse_complex_few_shot("### 01 好\n特征：a\n### 没有编号\n特征：b")

    def test_document_title_skipped(self) -> None:
        from query_pipeline.prompts.assemble import parse_complex_few_shot

        specs = parse_complex_few_shot("## 文档标题\n### 01 好\n特征：a")
        self.assertEqual(set(specs), {"01"})

    def test_complex_missing_hash_does_not_absorb(self) -> None:
        # 回归：`## 02`（类别头少一个 #）曾静默把 02 的内容并入 01
        # （实证 "特征：a 特征：b"，02 消失）。文档级标题必须终结当前类别，
        # 其后正文 fail-loud。
        from query_pipeline.prompts.assemble import parse_complex_few_shot

        with self.assertRaises(ValueError):
            parse_complex_few_shot("### 01 好\n特征：a\n## 02\n特征：b")
        # 无正文的文档级标题仅终结类别，不报错
        specs = parse_complex_few_shot("### 01 好\n特征：a\n## 02")
        self.assertEqual(set(specs), {"01"})

    def test_missing_spec_raises_in_builder(self) -> None:
        # taxonomy 有类别而 few_shot 缺 spec：build 必须 fail-loud
        from unittest.mock import patch

        from query_pipeline.prompts.assemble import _read

        good = _read("normal_few_shot.md")
        # 删掉 16 的整个 section（保留其它类别）
        truncated = good.rsplit("## 16-", 1)[0]
        with patch("query_pipeline.prompts.assemble._read", return_value=truncated):
            with self.assertRaises(ValueError):
                build_normal_classify_prompt()


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
        text = (templates_dir() / "bad_cases_for_complex.md").read_text(encoding="utf-8")
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

class PromptContractTest(unittest.TestCase):
    def test_prompt_contracts(self) -> None:
        segment_prompt = resolve_prompt("segment")
        self.assertIn("segments", segment_prompt)
        self.assertIn("start", segment_prompt)
        self.assertIn("end", segment_prompt)
        self.assertIn("topic", segment_prompt)
        self.assertIn("同一个主题不能再次出现", segment_prompt)
        self.assertIn("宏观", segment_prompt)

        judge_prompt = resolve_prompt("complex_judge")
        for category_id, name in {
            "01": "复杂取数计算",
            "05": "资产配置",
            "07": "策略触发任务类",
            "09": "动作类",
        }.items():
            self.assertIn(f"{category_id} {name}", judge_prompt)
        self.assertIn("is_complex", judge_prompt)
        self.assertIn("category_id", judge_prompt)
        self.assertIn("reason", judge_prompt)
        # category definitions + priority rules embedded (guards 08/09 boundary collapse)
        self.assertIn("长期帮我盯着并迭代", judge_prompt)
        self.assertIn("→ 优先 09", judge_prompt)
        # few_shot.md examples fused in (07 remapped to trigger/setup semantics;
        # backtest-audit questions fall under 03 now)
        self.assertIn("每类典型示例", judge_prompt)
        self.assertIn("回测一个基于5周均线的短线择时策略", judge_prompt)
        self.assertIn("审计一个多因子策略", judge_prompt)
        self.assertIn("07/08/09 边界", judge_prompt)
        # screening caliber: pure filters are non-complex; validation+trend-point tasks stay 01
        self.assertIn("仅按显式条件过滤", judge_prompt)
        self.assertIn("validate the BAR columns", judge_prompt)


class LazyPromptBuildTest(unittest.TestCase):
    def test_import_does_not_touch_templates(self) -> None:
        # 第四轮 #6：模板 fail-loud 不得在模块导入期发生——模板目录缺失/损坏时，
        # import query_pipeline.prompts 与 --help 仍可用，只有真正 resolve 才报错。
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "templates"
            empty.mkdir()
            env = dict(os.environ, QUERY_PIPELINE_TEMPLATES=str(empty))
            code = (
                "import query_pipeline.prompts as p\n"
                "print('import ok')\n"
                "try:\n"
                "    p.resolve_prompt('segment')\n"
                "    print('resolve ok')\n"
                "except FileNotFoundError as exc:\n"
                "    print('resolve failed:', type(exc).__name__)\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, env=env, cwd=str(ROOT),
            )
            self.assertIn("import ok", proc.stdout, proc.stderr)
            self.assertIn("resolve failed: FileNotFoundError", proc.stdout, proc.stderr)

    def test_patch_dict_works_before_first_build(self) -> None:
        # 惰性构建 + setdefault：patch.dict 预置的 prompt 优先，其余键由真实模板补齐。
        from unittest.mock import patch

        with patch("query_pipeline.prompts.PROMPTS", {**PROMPTS, "segment": "patched"}):
            self.assertEqual(resolve_prompt("segment"), "patched")
            self.assertIn("价值", resolve_prompt("value_gate"))
