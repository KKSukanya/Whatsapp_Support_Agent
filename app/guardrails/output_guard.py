"""
Output guardrails.

The failure mode this exists to prevent: the model confidently stating an
order status, delivery date, or discount code that it invented rather than
read from a tool result. That's the single most expensive category of bug
in this domain (JD: "the failure modes that cost customers money").

Strategy: for order-status answers, the agent is only allowed to state
values that literally appear in the tool result dict it was given. We check
this post-hoc by looking for other order-like tokens (statuses, other order
IDs) that weren't in the source-of-truth payload. It's a blunt check, not
semantic verification — documented as a known limitation, with the
follow-up being an LLM-as-judge grounding check for anything this misses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

KNOWN_STATUSES = {"shipped", "processing", "delivered", "cancelled", "returned", "refunded"}


@dataclass
class GroundingResult:
    grounded: bool
    reason: str


def validate_order_answer(answer_text: str, tool_result: dict | None) -> GroundingResult:
    if tool_result is None:
        # No tool was called but the answer still asserts a concrete status —
        # that's exactly the hallucination pattern we're guarding against.
        if any(status in answer_text.lower() for status in KNOWN_STATUSES):
            return GroundingResult(
                grounded=False,
                reason="Answer states an order status but no order-lookup tool was called.",
            )
        return GroundingResult(grounded=True, reason="No status claim made without a source.")

    allowed_status = tool_result.get("status", "").lower()
    mentioned_statuses = [s for s in KNOWN_STATUSES if s in answer_text.lower()]

    for s in mentioned_statuses:
        if s != allowed_status:
            return GroundingResult(
                grounded=False,
                reason=f"Answer mentions status '{s}' which does not match tool result '{allowed_status}'.",
            )

    other_order_ids = re.findall(r"\b[A-Z]{2}\d{4,}\b", answer_text)
    true_id = tool_result.get("order_id", "")
    for oid in other_order_ids:
        if oid != true_id:
            return GroundingResult(
                grounded=False,
                reason=f"Answer references order id '{oid}' not present in the source-of-truth tool result.",
            )

    return GroundingResult(grounded=True, reason="Claims match tool result.")


def contains_unauthorized_discount(answer_text: str) -> bool:
    """The agent must never state a specific discount percentage/code unless
    it was explicitly retrieved from the tenant's official promotions doc."""
    return bool(re.search(r"\b\d{1,2}%\s*(off|discount)\b", answer_text.lower())) or bool(
        re.search(r"\bcode[:\s]+[A-Z0-9]{4,}\b", answer_text)
    )
