"""step1 阈值网格搜索：对 cn_expert_daily_2026-08-07.jsonl 统计所有 (reject, tc, ut, st) 组合下的候选 turn 数。

候选定义（与 src/query_pipeline/session/candidates.py select_candidates 一致）：
  is_eligible(turn) 且 (reject_rules=off 或 无 reject reason) 且
  chain_tool_calls>=tc 且 chain_steps>=st 且 unique_tools>=ut
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from query_pipeline.adapters.session import adapt_turn
from query_pipeline.io.jsonl import read_jsonl_with_bad_lines
from query_pipeline.session.candidates import (
    chain_steps,
    chain_tool_calls,
    generic_reject_reason,
    is_eligible,
    unique_tools,
)

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/iwencai/cn_expert_daily_2026-08-07.jsonl")
BAD = Path("work/iwencai/bad_lines_grid.jsonl")
t0 = time.time()

# 逐 turn 统计: (eligible, rejected, tc, ut, st) -> 计数
from collections import Counter

stats = Counter()
total_turns = 0
records, _ = read_jsonl_with_bad_lines(PATH, BAD)
for rec in records:
    for raw in rec.get("context", []):
        if not isinstance(raw, dict):
            continue
        total_turns += 1
        turn = adapt_turn(raw)
        if not is_eligible(turn):
            continue
        rejected = bool(generic_reject_reason(turn.question))
        stats[(rejected, chain_tool_calls(turn), unique_tools(turn), chain_steps(turn))] += 1
print(f"读取+统计完成: {total_turns} turns, {sum(stats.values())} eligible, {time.time()-t0:.0f}s", file=sys.stderr)

rows = list(stats.items())
base_off = sum(n for (rej, tc, ut, st), n in rows)
base_on = sum(n for (rej, tc, ut, st), n in rows if not rej)

tc_range = [0, 3, 4, 5, 6, 7, 8, 10]
ut_range = [0, 1, 2, 3, 4, 5]
st_range = [0, 1, 2]

print(f"base (reject off)={base_off}  base (reject on)={base_on}\n")

print("== reject_rules: ON, min_chain_steps=1 ==")
print("tc\\ut | " + " | ".join(f"{u}" for u in ut_range))
for tc in tc_range:
    row = []
    for ut in ut_range:
        n = sum(
            n
            for (rej, c, u, s), n in rows
            if not rej and c >= tc and u >= ut and s >= 1
        )
        row.append(f"{n:>6}")
    print(f"{tc:>4}  | " + " | ".join(row))

print()
print("== reject_rules: ON, min_chain_steps=0 (忽略步数) ==")
for tc in tc_range:
    row = []
    for ut in ut_range:
        n = sum(
            n
            for (rej, c, u, s), n in rows
            if not rej and c >= tc and u >= ut
        )
        row.append(f"{n:>6}")
    print(f"{tc:>4}  | " + " | ".join(row))

print()
print("== reject_rules: OFF, min_chain_steps=1 ==")
for tc in tc_range:
    row = []
    for ut in ut_range:
        n = sum(
            n
            for (rej, c, u, s), n in rows
            if c >= tc and u >= ut and s >= 1
        )
        row.append(f"{n:>6}")
    print(f"{tc:>4}  | " + " | ".join(row))

print()
print("== 目标 4000 附近的组合 (reject ON, st>=1) ==")
best = []
for tc in range(3, 9):
    for ut in range(1, 6):
        n = sum(
            n
            for (rej, c, u, s), n in rows
            if not rej and c >= tc and u >= ut and s >= 1
        )
        best.append((n, tc, ut))
for n, tc, ut in sorted(best):
    print(f"tc>={tc} ut>={ut} st>=1 -> {n}  (偏差 {(n-4000)/4000*100:+.1f}%)")
