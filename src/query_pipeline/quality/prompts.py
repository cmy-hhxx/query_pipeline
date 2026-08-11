from __future__ import annotations

import json
from typing import Any

from query_pipeline.taxonomy import COMPLEX_PREFIX, load_taxonomy
from query_pipeline.prompts.assemble import load_complex_quality_policy

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

对给定记录，你需要给出三个判定：
1. question_quality：判断该问句本身是否完整、自然、可执行、可验证且语义清晰自洽。取值 "high"（高质量）或 "low"（低质量）。低质量包括：被截断、乱码、重复粘贴、语义含糊自相矛盾、明显残缺；只有标签/维度没有可执行关系；短到缺少对象、范围或任务；绝对化且不可验证的收益/最优目标；严重角色扮演、固定分步框架、原始数据表、精度容差、批量策略引擎或长指令清单主导文本。自然多条件、多句背景、换行、正常表格、中英文混杂或附带格式不应判低质；正常复杂问句也不因缺少次要实现细节判低质。
2. label_ok：判断该问句是否真的属于给定的分类标签。取值 true（属于）或 false（不属于）。
3. difficulty_ok：严格按文末《Complex 问句质量政策》判断 difficulty_level。自然、非模板化且包含至少 3 个彼此独立实质条件的量化筛选可以是 hard；单点、单条件、榜单、单公式、泛泛推荐和绝对化目标必须是 normal；严重 eval 模板与嵌入提示词不应出现在数据中。给定 complexity_profile 时，核对 route、complex_features、exclusion_reasons 与问句内 evidence 是否一致。

分类体系（id 中文名 / 英文 slug）：
{categories}

《Complex 问句质量政策》：
{load_complex_quality_policy()}

输出 JSON 结构：
{{"question_quality": "high" 或 "low", "label_ok": true 或 false, "difficulty_ok": true 或 false, "reason": "一句话理由，中文"}}
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
        "difficulty_level": row.get("difficulty_level"),
        "complexity_profile": (row.get("meta") or {}).get("complexity_profile")
        if isinstance(row.get("meta"), dict)
        else None,
        "chain_hops": chain_hops,
        "tools": unique_tools[:10],
    }


def build_judge_user_prompt(row: dict[str, Any]) -> str:
    payload = build_judge_payload(row)
    return "请对以下问句记录做质检，只输出严格 JSON：\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
