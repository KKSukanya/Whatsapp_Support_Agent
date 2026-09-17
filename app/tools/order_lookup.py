"""
Mock Shopify order-lookup tool. In production this hits the Shopify Admin
API (or an internal order service) scoped to the brand's store; here it
reads a per-tenant JSON file so the whole project runs with no external
accounts. The interface (`lookup_order`) is what the agent graph depends
on — swapping the implementation for a real API call doesn't touch any
other file.
"""
from __future__ import annotations

import json
import os

from app.config import settings


def lookup_order(tenant_id: str, order_id: str) -> dict | None:
    """Returns the order dict if found for this tenant, else None.

    Tenant isolation here isn't just a filter — the file path itself is
    scoped per tenant, so there's no code path where tenant A's order ID
    could accidentally resolve against tenant B's data.
    """
    order_id = order_id.strip().upper()
    path = os.path.join(settings.TENANTS_DIR, tenant_id, "orders.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        orders = json.load(f)
    return orders.get(order_id)


def extract_order_id(text: str) -> str | None:
    """Very small heuristic extractor: looks for a token matching our fake
    order-id format (2 letters + 4+ digits, e.g. GC1001, US2001)."""
    import re

    match = re.search(r"\b([A-Za-z]{2}\d{4,})\b", text)
    return match.group(1).upper() if match else None
