from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from query_pipeline.quality.paths import project_root, qc_dir, source_path


def overview(dataset: str, date: str, *, root: Path | None = None) -> dict[str, Any]:
    """Read a QC run's overview. Raise FileNotFoundError if the run never happened."""
    path = qc_dir(dataset, date, root) / "overview.json"
    if not path.exists():
        raise FileNotFoundError(f"QC overview 不存在：{path}（先运行 query-pipeline-qc run）")
    return json.loads(path.read_text(encoding="utf-8"))


def record_detail(dataset: str, date: str, record_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Drill into one record: the raw output row plus its QC result (if any)."""
    root = root or project_root()
    qc_out = qc_dir(dataset, date, root)
    source = source_path(dataset, date, root)
    overview_path = qc_out / "overview.json"
    if overview_path.exists():
        # Honor the actual source the run used (e.g. a --input override).
        stored_source = json.loads(overview_path.read_text(encoding="utf-8")).get("source")
        if stored_source:
            source = Path(stored_source)
    if not source.exists():
        raise FileNotFoundError(f"输出文件不存在：{source}")
    record: dict[str, Any] | None = None
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("trace_id") or "") == record_id:
                record = row
                break
    if record is None:
        raise KeyError(f"在 {source} 中找不到 trace_id={record_id!r} 的记录")

    results_path = qc_out / "results.jsonl"
    qc: dict[str, Any] | None = None
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("trace_id") == record_id:
                    qc = entry
                    break
    return {"record": record, "qc": qc}
