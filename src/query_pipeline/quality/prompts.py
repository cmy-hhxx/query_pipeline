from __future__ import annotations

import json
from typing import Any

from query_pipeline.taxonomy import COMPLEX_PREFIX, load_taxonomy

# Cache-key step for QC judge calls (separate namespace from pipeline stages).
QC_STEP = "qc_judge"


def build_judge_system_prompt() -> str:
    tax = load_taxonomy()
    lines = [
        f"- {cat.path}（{'复杂' if cat.difficulty == 'hard' else '普通'}，{cat.name}）"
        for cat in tax.all()
    ]
    categories = "\n".join(lines)
    return f"""你是数据质检员，负责抽检被管线分类的问句记录。请只输出严格 JSON，不要 Markdown、代码块或说明。

对给定记录，你需要给出两个判定：
1. question_quality：判断该问句本身是否完整、无截断、语义清晰自洽。取值 "high"（高质量）或 "low"（低质量）。低质量包括：被截断、乱码、重复粘贴、语义含糊自相矛盾、明显残缺。
2. label_ok：判断该问句是否真的属于给定的分类标签。取值 true（属于）或 false（不属于）。

分类体系（id 中文名 / 英文 slug）：
{categories}

输出 JSON 结构：
{{"question_quality": "high" 或 "low", "label_ok": true 或 false, "reason": "一句话理由，中文"}}
"""


def build_judge_payload(row: dict[str, Any]) -> dict[str, Any]:
    inp = row.get("input")
    question = inp.get("text") if isinstance(inp, dict) else ""
    category = str(row.get("category") or "")
    bare = category[len(COMPLEX_PREFIX):] if category.startswith(COMPLEX_PREFIX) else category
    category_id, _, _ = bare.partition("-")
    tax = load_taxonomy()
    cat = next((c for c in tax.all() if c.id == category_id and c.path == category), None)
    if cat is not None:
        label_definition = (
            f"{cat.id} {cat.name}（{cat.path}）"
        )
    else:
        label_definition = category

    chain = row.get("chain")
    chain_hops = len(chain) if isinstance(chain, list) else 0
    tools: list[str] = []
    if isinstance(chain, list):
        for hop in chain:
            if isinstance(hop, dict) and isinstance(hop.get("tools"), list):
                for tool in hop["tools"]:
                    if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                        tools.append(tool["name"])
    unique_tools = list(dict.fromkeys(tools))

    return {
        "question": question,
        "label": category,
        "label_definition": label_definition,
        "chain_hops": chain_hops,
        "tools": unique_tools[:10],
    }


def build_judge_user_prompt(row: dict[str, Any]) -> str:
    payload = build_judge_payload(row)
    return "请对以下问句记录做质检，只输出严格 JSON：\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
