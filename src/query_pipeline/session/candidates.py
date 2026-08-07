from __future__ import annotations

from typing import Any

from query_pipeline.config.models import Step1Config
from query_pipeline.rules.reject import generic_reject_reason


def is_eligible(turn: dict[str, Any]) -> bool:
    status = turn.get("status")
    outcome = turn.get("outcome")
    if status not in (None, "completed"):
        return False
    if outcome not in (None, "success"):
        return False
    answer = turn.get("answer")
    return bool(answer and str(answer).strip())


def chain_tool_calls(turn: dict[str, Any]) -> int:
    chain = turn.get("chain")
    if not isinstance(chain, list):
        return 0
    total = 0
    for step in chain:
        if not isinstance(step, dict):
            continue
        tools = step.get("tools")
        if isinstance(tools, list):
            total += len(tools)
    return total


def chain_steps(turn: dict[str, Any]) -> int:
    chain = turn.get("chain")
    return len(chain) if isinstance(chain, list) else 0


def extract_tool_names(turn: dict[str, Any]) -> list[str]:
    """Ordered unique tool names for a turn.

    Prefer the tool names actually used in chain steps (the real source of
    truth); fall back to the tool_names field only when the chain carries no
    tools (real data often leaves tool_names empty on deep turns).
    """
    names: list[str] = []
    seen: set[str] = set()

    chain = turn.get("chain")
    if isinstance(chain, list):
        for step in chain:
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

    if not names:
        tool_names = turn.get("tool_names")
        if isinstance(tool_names, str):
            for name in tool_names.split(","):
                name = name.strip()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def unique_tools(turn: dict[str, Any]) -> int:
    return len(extract_tool_names(turn))


def select_candidates(turns: list[dict[str, Any]], cfg: Step1Config) -> list[int]:
    """Return indices of turns that look like complex-question candidates.

    Base filter: reject_rules (low-value / blank / too-short text) via
    generic_reject_reason. Then AND-gate on three complexity signals — a turn
    must clear all of: total chain tool-call count, chain step count, and
    distinct tool-name count.
    """
    candidates: list[int] = []
    for idx, turn in enumerate(turns):
        if not is_eligible(turn):
            continue
        if cfg.reject_rules:
            reason = generic_reject_reason(turn.get("question", ""))
            if reason:
                continue
        if (
            chain_tool_calls(turn) >= cfg.min_chain_tool_calls
            and chain_steps(turn) >= cfg.min_chain_steps
            and unique_tools(turn) >= cfg.min_unique_tools
        ):
            candidates.append(idx)
    return candidates
