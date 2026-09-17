"""
Run this whenever tenant policy/FAQ docs change:

    python -m app.rag.build_index

In production this is the job you'd put on a schedule (or trigger on a
webhook from the CMS/Shopify metafields) to keep retrieval fresh — see
docs/architecture.md for the refresh-strategy notes.
"""
from app.rag.vector_store import VectorStore

if __name__ == "__main__":
    store = VectorStore()
    n = store.build()
    store.save()
    print(f"Indexed {n} chunks across tenants -> {store.embedder.__class__.__name__}")
    from collections import Counter

    by_tenant = Counter(c.tenant_id for c in store.chunks)
    for tenant, count in by_tenant.items():
        print(f"  {tenant}: {count} chunks")
