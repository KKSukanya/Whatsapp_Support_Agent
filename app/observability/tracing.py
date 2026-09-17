"""
Observability.

Everything that happens in a turn — which node ran, what it decided, tool
calls, tokens, latency, cost, guardrail flags — gets appended as one JSON
line to logs/traces.jsonl. The design goal from the JD: "a trace should
tell the story without a walkthrough." Run `python -m eval.cost_report` to
aggregate this file into cost-per-conversation numbers, or open it
directly — it's one JSON object per turn, easy to pipe into jq/pandas.

If LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY is set, LangGraph's
built-in LangSmith integration traces automatically in parallel to this —
see README for setup. This local JSONL logger is intentionally
provider-independent so the project doesn't require a LangSmith account to
demo.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class TurnTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tenant_id: str = ""
    session_id: str = ""
    user_message: str = ""
    intent: str | None = None
    nodes_visited: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    guardrail_flags: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
    escalated: bool = False
    final_answer: str = ""
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    error: str | None = None
    _start: float = field(default_factory=time.time, repr=False)

    def log_node(self, name: str):
        self.nodes_visited.append(name)

    def log_tool_call(self, name: str, input_: dict, output: dict | None):
        self.tool_calls.append({"tool": name, "input": input_, "output": output})

    def log_guardrail(self, category: str, severity: str, action: str):
        self.guardrail_flags.append({"category": category, "severity": severity, "action": action})

    def log_llm_call(self, purpose: str, result):
        self.llm_calls.append(
            {
                "purpose": purpose,
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "latency_ms": round(result.latency_ms, 1),
            }
        )
        self.total_cost_usd = round(self.total_cost_usd + result.cost_usd, 6)

    def finalize(self):
        self.total_latency_ms = round((time.time() - self._start) * 1000, 1)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return d

    def write(self, path: str = None):
        path = path or settings.TRACE_LOG_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")


@contextmanager
def start_trace(tenant_id: str, session_id: str, user_message: str):
    trace = TurnTrace(tenant_id=tenant_id, session_id=session_id, user_message=user_message)
    try:
        yield trace
    except Exception as e:
        trace.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        trace.finalize()
        trace.write()
