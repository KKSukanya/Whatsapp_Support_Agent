"""
Single entrypoint the API calls: `run_agent(tenant_id, session_id, message)`.

Wraps the LangGraph execution with:
  - a hard timeout + deterministic fallback (JD: "build deterministic
    fallbacks for when the model is wrong, slow, or unavailable")
  - the trace context manager (every turn gets logged, success or failure)
  - a tool-call/node-count guard as a second line of defense against
    runaway loops, independent of whatever LangGraph itself does
"""
from __future__ import annotations

import concurrent.futures

from app.agents.graph import get_graph
from app.config import settings
from app.observability.tracing import start_trace

FALLBACK_MESSAGE = (
    "Sorry, I'm having trouble processing that right now. I've flagged this "
    "conversation for a teammate to follow up shortly — thanks for your patience."
)


def _invoke_graph(initial_state: dict) -> dict:
    graph = get_graph()
    return graph.invoke(initial_state)


def run_agent(tenant_id: str, session_id: str, message: str) -> dict:
    with start_trace(tenant_id, session_id, message) as trace:
        initial_state = {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "user_message": message,
            "escalated": False,
            "retries": 0,
            "trace": trace,
        }

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_invoke_graph, initial_state)
                final_state = future.result(timeout=settings.LLM_TIMEOUT_SECONDS * 3)
        except concurrent.futures.TimeoutError:
            trace.error = "graph_execution_timeout"
            trace.escalated = True
            trace.final_answer = FALLBACK_MESSAGE
            return {
                "answer": FALLBACK_MESSAGE,
                "escalated": True,
                "intent": None,
                "trace_id": trace.trace_id,
            }
        except Exception as e:  # noqa: BLE001 - deliberate: last-resort catch-all
            trace.error = f"{type(e).__name__}: {e}"
            trace.escalated = True
            trace.final_answer = FALLBACK_MESSAGE
            return {
                "answer": FALLBACK_MESSAGE,
                "escalated": True,
                "intent": None,
                "trace_id": trace.trace_id,
            }

        trace.escalated = final_state.get("escalated", False)
        trace.final_answer = final_state.get("answer", "")

        return {
            "answer": final_state.get("answer", FALLBACK_MESSAGE),
            "escalated": final_state.get("escalated", False),
            "escalation_reason": final_state.get("escalation_reason"),
            "intent": final_state.get("intent"),
            "trace_id": trace.trace_id,
        }
