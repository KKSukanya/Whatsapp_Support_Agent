"""
Thin LLM wrapper.

Why this exists: the whole agent graph should never care whether it's
talking to OpenAI, Gemini, or a canned response. It calls `llm.complete(...)`
and gets back text + token/cost accounting. That's also what makes the
project runnable and demoable with zero API keys (LLM_PROVIDER=mock),
which matters for a portfolio piece someone will actually try to run.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from app.config import settings


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


def _estimate_tokens(text: str) -> int:
    """
    Cheap, dependency-free token estimate (~4 chars/token for English).
    Good enough for relative cost comparisons; swap for tiktoken in
    production if you want exact OpenAI tokenization.
    """
    return max(1, int(len(text) / 4))


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = settings.PRICING.get(model, settings.PRICING["mock"])
    return round(
        (input_tokens / 1000) * pricing["input"]
        + (output_tokens / 1000) * pricing["output"],
        6,
    )


class MockLLM:
    """
    Deterministic, rule-based stand-in for a real model. It's intentionally
    simple: enough to make the agent graph, RAG grounding, and guardrails
    behave realistically end-to-end without any API key or network call.
    """

    model_name = "mock"

    def complete(self, system: str, user: str, context: str = "") -> LLMResult:
        start = time.time()
        text = self._respond(system, user, context)
        latency = (time.time() - start) * 1000
        in_tok = _estimate_tokens(system + user + context)
        out_tok = _estimate_tokens(text)
        return LLMResult(
            text=text,
            model=self.model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=0.0,
            latency_ms=latency,
        )

    def _respond(self, system: str, user: str, context: str) -> str:
        low = user.lower()

        # Router prompt asks for an intent label
        if "classify the user intent" in system.lower():
            if re.search(r"\b(order|track|shipment|where.?s my|delivery status)\b", low):
                return "order_status"
            if re.search(r"\b(discount|coupon|promo code|offer code)\b", low):
                return "discount_request"
            if re.search(r"\b(return|refund|exchange|cancel|policy|shipping|faq|waterproof|warranty|size|fit)\b", low):
                return "policy_qa"
            if re.search(r"\b(human|agent|representative|talk to someone)\b", low):
                return "escalate"
            return "policy_qa"

        # RAG-grounded answer: skip markdown heading lines (# ..., **...**),
        # blank lines, and bracketed [Heading] prefixes to surface the
        # actual policy sentence instead of just a section title.
        if context:
            lines = [ln.strip() for ln in context.strip().split("\n") if ln.strip()]
            content_line = ""
            for ln in lines:
                stripped = re.sub(r"^\[.+?\]\s*", "", ln)  # drop "[Heading] " prefix
                is_heading = bool(re.match(r"^#{1,3}\s", stripped)) or (
                    stripped.startswith("**") and stripped.endswith("**")
                )
                if stripped and not is_heading:
                    content_line = stripped
                    break
            snippet = (content_line or lines[0])[:400]
            return (
                f"Based on our policy: {snippet}\n\n"
                f"Let me know if you'd like more detail on this."
            )

        # Order status answer (context will contain the tool result already
        # formatted by the caller in that case, handled above)
        return (
            "I don't have enough information sourced from our systems to answer "
            "that confidently, so I'll connect you with a human teammate who can help."
        )


class OpenAILLM:
    """Real OpenAI-backed implementation. Only imported/instantiated when needed."""

    def __init__(self, model: str):
        from openai import OpenAI  # local import so mock mode never needs the package

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = model

    def complete(self, system: str, user: str, context: str = "") -> LLMResult:
        start = time.time()
        messages = [{"role": "system", "content": system}]
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": user})

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        latency = (time.time() - start) * 1000
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else _estimate_tokens(system + user + context)
        out_tok = usage.completion_tokens if usage else _estimate_tokens(text)
        return LLMResult(
            text=text,
            model=self.model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=_cost(self.model_name, in_tok, out_tok),
            latency_ms=latency,
        )


def get_llm(tier: str = "cheap"):
    """
    tier: "cheap" | "strong" — this is the cost-routing hook. Callers ask for
    a capability tier, not a specific model, so swapping the underlying model
    per tier is a one-line config change (see app/config.py).
    """
    provider = settings.effective_provider
    if provider == "mock":
        return MockLLM()
    model = settings.CHEAP_MODEL if tier == "cheap" else settings.STRONG_MODEL
    return OpenAILLM(model)
