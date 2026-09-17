"""
Aggregates logs/traces.jsonl into the unit-economics numbers the JD asks
for by name: "cost per conversation, per resolution, per brand."

    python -m eval.cost_report

Also runs a small routing comparison: what would this same set of turns
have cost if every call had gone to the strong model instead of the cheap
one? That delta is the number you defend a routing decision with.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from app.config import settings

TRACE_PATH = Path(settings.TRACE_LOG_PATH)


def load_traces() -> list[dict]:
    if not TRACE_PATH.exists():
        return []
    with open(TRACE_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def report():
    traces = load_traces()
    if not traces:
        print(f"No traces found at {TRACE_PATH}. Run some /chat turns or the eval suite first.")
        return

    total_cost = sum(t.get("total_cost_usd", 0) for t in traces)
    n = len(traces)
    resolved = [t for t in traces if not t.get("escalated")]
    n_resolved = len(resolved)

    by_tenant = defaultdict(lambda: {"count": 0, "cost": 0.0})
    for t in traces:
        by_tenant[t["tenant_id"]]["count"] += 1
        by_tenant[t["tenant_id"]]["cost"] += t.get("total_cost_usd", 0)

    print(f"\n{'='*60}")
    print("COST REPORT")
    print(f"{'='*60}")
    print(f"Total turns logged:          {n}")
    print(f"Total cost:                  ${total_cost:.6f}")
    print(f"Avg cost / conversation:     ${(total_cost/n if n else 0):.6f}")
    print(f"Resolved without escalation: {n_resolved}/{n} ({n_resolved/n*100 if n else 0:.1f}%)")
    print(f"Avg cost / RESOLVED turn:    ${(total_cost/n_resolved if n_resolved else 0):.6f}")
    print(f"\nBy tenant:")
    for tenant, stats in by_tenant.items():
        avg = stats["cost"] / stats["count"] if stats["count"] else 0
        print(f"  {tenant:<15} turns={stats['count']:<5} total=${stats['cost']:.6f}  avg=${avg:.6f}")

    # Model routing comparison: cheap-first vs. always-strong
    cheap_price = settings.PRICING.get(settings.CHEAP_MODEL, {"input": 0, "output": 0})
    strong_price = settings.PRICING.get(settings.STRONG_MODEL, {"input": 0, "output": 0})

    total_in = sum(sum(c["input_tokens"] for c in t.get("llm_calls", [])) for t in traces)
    total_out = sum(sum(c["output_tokens"] for c in t.get("llm_calls", [])) for t in traces)

    actual_cost = total_cost
    hypothetical_strong_cost = (total_in / 1000) * strong_price["input"] + (total_out / 1000) * strong_price["output"]

    print(f"\nROUTING COMPARISON (cheap-first actual vs. always-strong hypothetical)")
    print(f"  Actual cost (current routing): ${actual_cost:.6f}")
    print(f"  If EVERY call used '{settings.STRONG_MODEL}':  ${hypothetical_strong_cost:.6f}")
    if actual_cost > 0:
        print(f"  Savings from routing: {((hypothetical_strong_cost - actual_cost) / max(hypothetical_strong_cost, 1e-9) * 100):.1f}%")
    print()


if __name__ == "__main__":
    report()
