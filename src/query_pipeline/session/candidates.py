from __future__ import annotations

from query_pipeline.config.models import RuleGateConfig
from query_pipeline.models.turn import Turn
from query_pipeline.rules.reject import generic_reject_reason


# 未显式设置的旋钮按输入格式补齐（chat 工具调用分布平坦，>=7 次仅覆盖 ~1%）。
# suggest.py 直接导入本表（唯一事实源），is_default 标记与实际执行门槛一致。
FORMAT_DEFAULTS = {"session": (7, 2), "chat": (3, 2)}


def effective_gate(cfg: RuleGateConfig, fmt: str) -> RuleGateConfig:
    """按输入格式补齐 None 旋钮；显式传入的旋钮保持原值。"""
    calls, tools = FORMAT_DEFAULTS.get(fmt, FORMAT_DEFAULTS["session"])
    return RuleGateConfig(
        enabled=cfg.enabled,
        reject_rules=cfg.reject_rules,
        min_chain_tool_calls=calls if cfg.min_chain_tool_calls is None else cfg.min_chain_tool_calls,
        min_chain_steps=cfg.min_chain_steps,
        min_unique_tools=tools if cfg.min_unique_tools is None else cfg.min_unique_tools,
    )


def is_eligible(turn: Turn) -> bool:
    if turn.status not in (None, "completed"):
        return False
    if turn.outcome not in (None, "success"):
        return False
    if not (turn.question and turn.question.strip()):
        return False
    return bool(turn.answer and turn.answer.strip())


def chain_tool_calls(turn: Turn) -> int:
    if turn.chain:
        total = 0
        for step in turn.chain:
            if not isinstance(step, dict):
                continue
            tools = step.get("tools")
            if isinstance(tools, list):
                total += len(tools)
        return total
    # chain absent: fall back to the raw tool_count (authoritative call count).
    return turn.tool_count or 0


def chain_steps(turn: Turn) -> int:
    if turn.chain:
        return len(turn.chain)
    # a toolful but chain-less turn performed at least one step by definition.
    return 1 if (turn.tool_count or 0) > 0 else 0


def extract_tool_names(turn: Turn) -> list[str]:
    """Ordered unique tool names: prefer chain, else tool_names fallback string."""
    names: list[str] = []
    seen: set[str] = set()

    for step in turn.chain:
        if not isinstance(step, dict):
            continue
        tools = step.get("tools")
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if isinstance(tool, dict):
                name = str(tool.get("name") or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)

    if not names and turn.tool_names:
        for name in turn.tool_names.split(","):
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def unique_tools(turn: Turn) -> int:
    return len(extract_tool_names(turn))


def select_candidates(turns: list[Turn], cfg: RuleGateConfig) -> list[int]:
    """Indices that pass reject rules + chain/tool AND-gates."""
    gate = effective_gate(cfg, "session")  # 直接调用者无格式上下文 → session 默认
    assert gate.min_chain_tool_calls is not None and gate.min_unique_tools is not None
    candidates: list[int] = []
    for idx, turn in enumerate(turns):
        if not is_eligible(turn):
            continue
        if gate.reject_rules:
            reason = generic_reject_reason(turn.question)
            if reason:
                continue
        if (
            chain_tool_calls(turn) >= gate.min_chain_tool_calls
            and chain_steps(turn) >= gate.min_chain_steps
            and unique_tools(turn) >= gate.min_unique_tools
        ):
            candidates.append(idx)
    return candidates


def select_last_only(turns: list[Turn], cfg: RuleGateConfig | None = None) -> list[int]:
    """Chat semantics: only the trailing turn is a candidate, gated on eligibility
    plus (when cfg given) reject rules and chain/tool AND-gates — chat records
    always carry judge_data.chain, so the tool gates apply there too."""
    if not turns:
        return []
    idx = len(turns) - 1
    turn = turns[idx]
    if not is_eligible(turn):
        return []
    if cfg is not None:
        gate = effective_gate(cfg, "session")
        assert gate.min_chain_tool_calls is not None and gate.min_unique_tools is not None
        if gate.reject_rules:
            reason = generic_reject_reason(turn.question)
            if reason:
                return []
        if not (
            chain_tool_calls(turn) >= gate.min_chain_tool_calls
            and chain_steps(turn) >= gate.min_chain_steps
            and unique_tools(turn) >= gate.min_unique_tools
        ):
            return []
    return [idx]
