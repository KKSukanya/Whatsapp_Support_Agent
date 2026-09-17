"""
Input guardrails.

Deliberately NOT an LLM call — "is this message trying to manipulate the
system prompt" is exactly the kind of check that should be a cheap,
auditable rule, not a second model call that itself could be fooled or that
doubles latency/cost on every single turn. This is the "reach for the
smallest thing that works before reaching for a bigger model" principle
from the JD, applied literally.

Real systems layer this with an LLM-based classifier for the cases regex
can't catch — see docs/architecture.md for where that would plug in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above)\s*instructions",
    r"disregard (all|any|the)?\s*(previous|prior|above)\s*(instructions|prompt)",
    r"you are now",
    r"act as (if|a)",
    r"new instructions?:",
    r"reveal (your|the) (system prompt|instructions)",
    r"what (is|are) your (system prompt|instructions)",
    r"repeat (the words|everything) above",
    r"forget (that|what) (you were|i) (told|said)",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"dan mode",
    r"developer mode",
    r"override your (rules|guidelines|policy)",
]

DISCOUNT_MANIPULATION_PATTERNS = [
    r"give me a discount code",
    r"generate (a|me a)? ?\d{0,2}%? ?(discount|coupon)( code)?",
    r"apply a? ?\d{1,2}%? ?discount",
    r"make up a coupon",
    r"i('m| am) a (vip|influencer|reviewer)",
]

PII_REQUEST_PATTERNS = [
    r"(credit card|card number|cvv|otp|password|aadhaar|pan card)",
]


@dataclass
class GuardrailFlag:
    category: str
    pattern: str
    severity: str  # "low" | "medium" | "high"


def scan_input(text: str) -> list[GuardrailFlag]:
    low = text.lower()
    flags: list[GuardrailFlag] = []

    for pat in INJECTION_PATTERNS:
        if re.search(pat, low):
            flags.append(GuardrailFlag(category="prompt_injection", pattern=pat, severity="high"))

    for pat in DISCOUNT_MANIPULATION_PATTERNS:
        if re.search(pat, low):
            flags.append(GuardrailFlag(category="discount_manipulation", pattern=pat, severity="medium"))

    for pat in PII_REQUEST_PATTERNS:
        if re.search(pat, low):
            flags.append(GuardrailFlag(category="sensitive_pii", pattern=pat, severity="high"))

    return flags


def highest_severity(flags: list[GuardrailFlag]) -> str | None:
    order = {"high": 3, "medium": 2, "low": 1}
    if not flags:
        return None
    return max(flags, key=lambda f: order[f.severity]).severity
