from __future__ import annotations

from query_pipeline.config.models import RuleGateConfig
from query_pipeline.models.turn import Turn
from query_pipeline.rules.reject import generic_reject_reason


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
    candidates: list[int] = []
    for idx, turn in enumerate(turns):
        if not is_eligible(turn):
            continue
        if cfg.reject_rules:
            reason = generic_reject_reason(turn.question)
            if reason:
                continue
        if (
            chain_tool_calls(turn) >= cfg.min_chain_tool_calls
            and chain_steps(turn) >= cfg.min_chain_steps
            and unique_tools(turn) >= cfg.min_unique_tools
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
        if cfg.reject_rules:
            reason = generic_reject_reason(turn.question)
            if reason:
                return []
        if not (
            chain_tool_calls(turn) >= cfg.min_chain_tool_calls
            and chain_steps(turn) >= cfg.min_chain_steps
            and unique_tools(turn) >= cfg.min_unique_tools
        ):
            return []
    return [idx]
