"""
Agent architecture.

Boundaries, decided deliberately (this is exactly the kind of call the JD
wants owned, not handed down as a spec):

  Router          -> classification only. Never generates customer-facing
                      text. One job, cheap model, easy to eval in isolation.
  policy_qa node  -> RAG-grounded Q&A over tenant policy docs.
  order_status    -> tool-call node. Never lets the LLM state a status
                      without a tool result to ground it (see output_guard).
  discount_request-> deterministic node, NOT an LLM call at all. Discount
                      logic is a business rule ("escalate, always"), and
                      rule > LLM call for anything with a fixed, auditable
                      answer.
  escalate        -> deterministic handoff message + flag for a human.

Everything is wired as a LangGraph StateGraph so tool-call loops, retries,
and node history are structural (visible in `state`), not implicit in
prompt text.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.config import settings
from app.guardrails.input_guard import scan_input, highest_severity
from app.guardrails.output_guard import validate_order_answer, contains_unauthorized_discount
from app.llm.client import get_llm
from app.observability.tracing import TurnTrace
from app.rag.vector_store import get_vector_store
from app.tools.order_lookup import lookup_order, extract_order_id


class AgentState(TypedDict, total=False):
    tenant_id: str
    session_id: str
    user_message: str
    intent: str
    retrieved_context: str
    tool_result: dict | None
    answer: str
    escalated: bool
    escalation_reason: str
    retries: int
    trace: TurnTrace  # not serialized by LangGraph checkpointing; runtime-only


ROUTER_SYSTEM_PROMPT = (
    "You classify the user intent for a D2C brand's WhatsApp support agent. "
    "Classify the user intent into exactly one label: order_status, policy_qa, "
    "discount_request, or escalate. Respond with only the label."
)

POLICY_QA_SYSTEM_PROMPT = (
    "You are a helpful, concise WhatsApp support agent for a D2C brand. "
    "Answer ONLY using the provided context. If the context does not contain "
    "the answer, say you're not sure and offer to connect the customer with a "
    "human. Never invent policy details, prices, or dates. Keep answers under "
    "4 sentences, in a friendly WhatsApp tone."
)

ORDER_STATUS_SYSTEM_PROMPT = (
    "You are a WhatsApp support agent. You will be given an order lookup "
    "result as context. Summarize ONLY the fields present in that result "
    "(status, items, expected delivery, tracking link if present). Never "
    "state a status or date that is not literally present in the context."
)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def node_input_guardrail(state: AgentState) -> AgentState:
    trace: TurnTrace = state["trace"]
    trace.log_node("input_guardrail")

    flags = scan_input(state["user_message"])
    severity = highest_severity(flags)

    for f in flags:
        trace.log_guardrail(f.category, f.severity, action="pending")

    if severity == "high":
        for f in flags:
            f  # already logged above
        return {
            **state,
            "escalated": True,
            "escalation_reason": "Input guardrail triggered (high severity): "
            + ", ".join(sorted({f.category for f in flags})),
            "answer": (
                "I want to make sure you get accurate help here, so I'm looping in "
                "a member of our team for this one. They'll follow up shortly."
            ),
        }
    return state


def node_router(state: AgentState) -> AgentState:
    trace: TurnTrace = state["trace"]
    trace.log_node("router")

    llm = get_llm(tier="cheap")
    result = llm.complete(system=ROUTER_SYSTEM_PROMPT, user=state["user_message"])
    trace.log_llm_call("routing", result)

    intent = result.text.strip().lower()
    valid = {"order_status", "policy_qa", "discount_request", "escalate"}
    if intent not in valid:
        intent = "policy_qa"  # safe default: ground in docs rather than guess

    trace.intent = intent
    return {**state, "intent": intent}


def node_policy_qa(state: AgentState) -> AgentState:
    trace: TurnTrace = state["trace"]
    trace.log_node("policy_qa")

    store = get_vector_store()
    chunks = store.retrieve(tenant_id=state["tenant_id"], query=state["user_message"])

    # Confidence gate: a weak top-score means the corpus likely doesn't
    # cover this question at all. Answering anyway is how you get a
    # confidently-wrong policy statement — escalate instead of guessing.
    top_score = chunks[0].score if chunks else 0.0
    if not chunks or top_score < settings.RAG_MIN_SCORE:
        trace.log_guardrail("weak_retrieval_match", "medium", action="escalated")
        return {
            **state,
            "escalated": True,
            "escalation_reason": f"No confident policy match (top_score={top_score:.2f}).",
            "answer": (
                "I couldn't find anything specific about that in our policies — "
                "let me get a teammate to help you directly."
            ),
        }

    context = "\n\n".join(f"[{c.heading}] {c.text}" for c in chunks)
    llm = get_llm(tier="cheap")
    result = llm.complete(system=POLICY_QA_SYSTEM_PROMPT, user=state["user_message"], context=context)
    trace.log_llm_call("policy_qa", result)

    if contains_unauthorized_discount(result.text):
        return {
            **state,
            "escalated": True,
            "escalation_reason": "Output guardrail: unauthorized discount claim blocked.",
            "answer": (
                "I can't apply or confirm discount codes myself — I'm connecting you "
                "with a teammate who can check current offers for you."
            ),
        }

    return {**state, "retrieved_context": context, "answer": result.text}


def node_order_status(state: AgentState) -> AgentState:
    trace: TurnTrace = state["trace"]
    trace.log_node("order_status")

    order_id = extract_order_id(state["user_message"])
    if not order_id:
        return {
            **state,
            "answer": "Could you share your order ID (e.g. GC1001) so I can look that up for you?",
        }

    tool_result = lookup_order(state["tenant_id"], order_id)
    trace.log_tool_call("lookup_order", {"tenant_id": state["tenant_id"], "order_id": order_id}, tool_result)

    if tool_result is None:
        return {
            **state,
            "tool_result": None,
            "escalated": True,
            "escalation_reason": f"Order id '{order_id}' not found for tenant.",
            "answer": (
                f"I couldn't find an order matching {order_id} on our end. "
                "I'll flag this to a teammate to double check for you."
            ),
        }

    context = str(tool_result)
    llm = get_llm(tier="cheap")
    result = llm.complete(system=ORDER_STATUS_SYSTEM_PROMPT, user=state["user_message"], context=context)
    trace.log_llm_call("order_status", result)

    grounding = validate_order_answer(result.text, tool_result)
    if not grounding.grounded:
        trace.log_guardrail("ungrounded_order_claim", "high", action="blocked_and_escalated")
        return {
            **state,
            "tool_result": tool_result,
            "escalated": True,
            "escalation_reason": f"Output guardrail: {grounding.reason}",
            "answer": (
                "Let me get a teammate to confirm your order details precisely, "
                "just to be safe."
            ),
        }

    return {**state, "tool_result": tool_result, "answer": result.text}


def node_discount_request(state: AgentState) -> AgentState:
    """Deterministic — no LLM call. Discount handling is a fixed business
    rule (always escalate), so an LLM call here would only add latency,
    cost, and a chance of the model improvising something it shouldn't."""
    trace: TurnTrace = state["trace"]
    trace.log_node("discount_request")
    return {
        **state,
        "escalated": True,
        "escalation_reason": "Discount requests are always routed to a human by policy.",
        "answer": (
            "I'm not able to generate or confirm discount codes myself, but I'm "
            "connecting you with a teammate who can check current offers for you!"
        ),
    }


def node_escalate(state: AgentState) -> AgentState:
    trace: TurnTrace = state["trace"]
    trace.log_node("escalate")
    if state.get("answer"):
        return state
    return {
        **state,
        "escalated": True,
        "answer": "I'll connect you with a member of our team who can help with that.",
    }


def route_after_guardrail(state: AgentState) -> str:
    return "escalate" if state.get("escalated") else "router"


def route_after_router(state: AgentState) -> str:
    return {
        "order_status": "order_status",
        "policy_qa": "policy_qa",
        "discount_request": "discount_request",
        "escalate": "escalate",
    }.get(state.get("intent", "policy_qa"), "policy_qa")


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("input_guardrail", node_input_guardrail)
    graph.add_node("router", node_router)
    graph.add_node("policy_qa", node_policy_qa)
    graph.add_node("order_status", node_order_status)
    graph.add_node("discount_request", node_discount_request)
    graph.add_node("escalate", node_escalate)

    graph.set_entry_point("input_guardrail")

    graph.add_conditional_edges(
        "input_guardrail", route_after_guardrail, {"router": "router", "escalate": "escalate"}
    )
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "order_status": "order_status",
            "policy_qa": "policy_qa",
            "discount_request": "discount_request",
            "escalate": "escalate",
        },
    )

    graph.add_edge("policy_qa", END)
    graph.add_edge("order_status", END)
    graph.add_edge("discount_request", END)
    graph.add_edge("escalate", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
