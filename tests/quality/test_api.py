"""QC API + CLI: overview/detail lookups and the end-to-end no-LLM run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from query_pipeline.quality.api import overview, record_detail
from query_pipeline.quality.cli import main as cli_main

def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "capture_mode": "full_link",
        "user_cohort": "regular",
        "source_case_id": "thread_a",
        "answer_key": "",
        "trace_id": "trace_001",
        "category": "complex-topic/03-analysis-research",
        "input": {
            "text": "今天8月3日，分析金安国际这只股票，从k线、市盈率、换手率等指标分析明天是否可以重仓？",
            "image": "",
            "file": "",
        },
        "context": [],
        "chain": [
            {
                "plan": "先识别股票代码",
                "tools": [
                    {"name": "stock_ner_parse", "input": {"query": "金安国际"}, "output": '{"code":"600318"}'}
                ],
            }
        ],
        "tools": ["stock_ner_parse"],
        "raw_answer": "金安国际今日走势稳健，市盈率处于合理区间，换手率适中，但主力筹码集中度仍需观察。"
        "建议明日轻仓试仓，不宜重仓。",
        "text_answer": "金安国际今日走势稳健，市盈率处于合理区间，换手率适中，但主力筹码集中度仍需观察。"
        "建议明日轻仓试仓，不宜重仓。",
        "multimodal": [],
        "model_version": "",
        "release_id": "",
        "agent_mode": "",
        "translation": None,
        "user_id": "u1",
        "difficulty_level": "hard",
        "first_token_time_ms": 1000,
        "finish_answer_time_ms": 2000,
        "input_tokens": 100,
        "output_tokens": 50,
        "request_time_ms": 1785854845000,
        "meta": {
            "reason": "需要多指标综合分析",
            "request_time": "2026-08-04 10:47:25",
        },
    }
    row.update(overrides)
    return row

class ApiTest(unittest.TestCase):
    def _write_run(self, root: Path) -> Path:
        source = root / "outputs" / "aime" / "cleaned_queries.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(_row(), ensure_ascii=False) + "\n", encoding="utf-8")

        qc = root / "outputs" / "aime" / "qc" / "0807"
        qc.mkdir(parents=True, exist_ok=True)
        overview_data = {
            "dataset": "aime",
            "date": "0807",
            "source": str(source),
            "total": 1,
            "skipped_bad_lines": 0,
            "status_counts": {"pass": 1, "fail": 0, "needs_review": 0},
        }
        (qc / "overview.json").write_text(json.dumps(overview_data), encoding="utf-8")
        results = [{"trace_id": "trace_001", "status": "pass", "sampled": False, "rules": [], "judge": None}]
        (qc / "results.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in results), encoding="utf-8"
        )
        return qc

    def test_overview_and_record_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root)
            data = overview("aime", "0807", root=root)
            self.assertEqual(data["total"], 1)
            detail = record_detail("aime", "0807", "trace_001", root=root)
            self.assertEqual(detail["record"]["trace_id"], "trace_001")
            self.assertEqual(detail["qc"]["status"], "pass")
            with self.assertRaises(KeyError):
                record_detail("aime", "0807", "nope", root=root)

    def test_qc_dir_is_date_scoped(self) -> None:
        # date 是路径参数：不同日期的 QC 产物互不覆盖（旧实现 date 死参数，
        # 0806/0807 两次运行写同一目录互相覆盖）。
        from query_pipeline.quality.paths import qc_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d0806 = qc_dir("aime", "0806", root)
            d0807 = qc_dir("aime", "0807", root)
            self.assertNotEqual(d0806, d0807)
            self.assertEqual(d0806.parent, d0807.parent)  # 同一数据集 qc/ 下
            self.assertEqual(d0806.name, "0806")
            self.assertEqual(d0807.name, "0807")

class CliE2ETest(unittest.TestCase):
    def test_run_no_llm_produces_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "cleaned_queries.jsonl"
            source.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in [_row(), _row(trace_id="t2", category="99-bad")])
                + "\n",
                encoding="utf-8",
            )
            qc = tmp / "qc"
            rc = cli_main(
                [
                    "run",
                    "--dataset", "aime",
                    "--date", "9999",
                    "--input", str(source),
                    "--qc-dir", str(qc),
                    "--no-llm",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((qc / "overview.json").exists())
            self.assertTrue((qc / "results.jsonl").exists())
            self.assertTrue((qc / "report.md").exists())
            overview_data = json.loads((qc / "overview.json").read_text(encoding="utf-8"))
            self.assertEqual(overview_data["status_counts"]["fail"], 1)
            self.assertIn("## 状态概览", (qc / "report.md").read_text(encoding="utf-8"))

    def test_missing_input_returns_nonzero(self) -> None:
        rc = cli_main(
            ["run", "--dataset", "aime", "--date", "9999", "--input", "/nonexistent/x.jsonl", "--no-llm"]
        )
        self.assertEqual(rc, 1)

