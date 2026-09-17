"""
Regression eval suite.

    python -m eval.run_eval                  # run once, print + save results
    python -m eval.run_eval --compare old.json new.json   # diff two runs

This is the artifact for "we know whether a prompt/model change made things
better or worse, with evidence" — not a vibe check. Each case checks:
  - intent routing correctness (where applicable)
  - escalation behavior (did it escalate when it should have / shouldn't have)
  - keyword grounding (did the answer actually contain the policy fact)

Keyword matching is a blunt instrument on purpose — it's fast, free, and
deterministic, which matters for a suite you'll run on every prompt change.
The natural next step (documented, not built here) is an LLM-as-judge pass
for semantic correctness on top of this.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.runner import run_agent  # noqa: E402

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_PATH = Path(__file__).parent / "last_run.json"


def run_case(case: dict) -> dict:
    session_id = str(uuid.uuid4())
    start = time.time()
    result = run_agent(tenant_id=case["tenant_id"], session_id=session_id, message=case["message"])
    latency = (time.time() - start) * 1000

    checks = {}

    if case.get("expected_intent") is not None:
        checks["intent_correct"] = result.get("intent") == case["expected_intent"]

    checks["escalation_correct"] = result.get("escalated") == case["expect_escalation"]

    keywords = case.get("expected_keywords", [])
    if keywords:
        answer_low = result.get("answer", "").lower()
        matched = [kw for kw in keywords if kw.lower() in answer_low]
        checks["keywords_matched"] = f"{len(matched)}/{len(keywords)}"
        checks["keywords_pass"] = len(matched) >= 1  # at least one grounding keyword present

    passed = all(v for k, v in checks.items() if isinstance(v, bool))

    return {
        "id": case["id"],
        "message": case["message"],
        "tenant_id": case["tenant_id"],
        "passed": passed,
        "checks": checks,
        "answer": result.get("answer"),
        "intent": result.get("intent"),
        "escalated": result.get("escalated"),
        "latency_ms": round(latency, 1),
        "note": case.get("note"),
    }


def run_all() -> dict:
    cases = json.loads(EVAL_SET_PATH.read_text())
    results = [run_case(c) for c in cases]
    n_passed = sum(1 for r in results if r["passed"])
    summary = {
        "total": len(results),
        "passed": n_passed,
        "failed": len(results) - n_passed,
        "pass_rate": round(n_passed / len(results), 3) if results else 0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1) if results else 0,
        "results": results,
    }
    return summary


def print_report(summary: dict):
    print(f"\n{'='*70}")
    print(f"EVAL RESULTS: {summary['passed']}/{summary['total']} passed "
          f"({summary['pass_rate']*100:.1f}%)  |  avg latency {summary['avg_latency_ms']}ms")
    print(f"{'='*70}")
    for r in summary["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']:<40} intent={str(r['intent']):<16} escalated={r['escalated']}")
        if not r["passed"]:
            print(f"       checks: {r['checks']}")
            print(f"       answer: {r['answer'][:120]}")
    print()


def compare(path_a: str, path_b: str):
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    a_by_id = {r["id"]: r for r in a["results"]}
    b_by_id = {r["id"]: r for r in b["results"]}

    print(f"\nCOMPARE: {path_a} (pass_rate={a['pass_rate']}) vs {path_b} (pass_rate={b['pass_rate']})\n")
    for case_id in a_by_id:
        pa, pb = a_by_id[case_id]["passed"], b_by_id.get(case_id, {}).get("passed")
        if pa != pb:
            arrow = "IMPROVED" if pb else "REGRESSED"
            print(f"  [{arrow}] {case_id}: {pa} -> {pb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    parser.add_argument("--save", default=str(RESULTS_PATH))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
    else:
        summary = run_all()
        print_report(summary)
        Path(args.save).write_text(json.dumps(summary, indent=2))
        print(f"Saved results -> {args.save}")
