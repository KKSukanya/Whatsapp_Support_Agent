"""
Run with: pytest tests/ -v

The one test that matters most here: cross-tenant retrieval isolation.
A bug in this test is the worst possible bug in this system — it means
Brand A's customer could see Brand B's policies or pricing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.vector_store import VectorStore
from app.tools.order_lookup import lookup_order


def test_tenant_isolation_no_cross_contamination():
    store = VectorStore()
    store.build()

    glowcart_results = store.retrieve(tenant_id="glowcart", query="return policy", top_k=10)
    urbanstride_results = store.retrieve(tenant_id="urbanstride", query="return policy", top_k=10)

    assert all(c.tenant_id == "glowcart" for c in glowcart_results)
    assert all(c.tenant_id == "urbanstride" for c in urbanstride_results)

    glowcart_sources = {c.source for c in glowcart_results}
    urbanstride_sources = {c.source for c in urbanstride_results}
    # Same filenames exist for both tenants (returns.md etc.) — the real
    # check is that the *text content* differs, not just tenant_id tagging.
    glowcart_texts = {c.text for c in glowcart_results}
    urbanstride_texts = {c.text for c in urbanstride_results}
    assert glowcart_texts.isdisjoint(urbanstride_texts)


def test_unknown_tenant_returns_empty():
    store = VectorStore()
    store.build()
    results = store.retrieve(tenant_id="does_not_exist", query="return policy")
    assert results == []


def test_order_lookup_respects_tenant_boundary():
    # US2001 belongs to urbanstride; looking it up under glowcart must fail.
    assert lookup_order("urbanstride", "US2001") is not None
    assert lookup_order("glowcart", "US2001") is None

    assert lookup_order("glowcart", "GC1001") is not None
    assert lookup_order("urbanstride", "GC1001") is None
