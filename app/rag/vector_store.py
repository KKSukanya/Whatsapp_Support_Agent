"""
Vector store with per-tenant isolation.

Design choice worth defending in an interview: tenant filtering happens
BEFORE similarity search, not as a post-hoc filter on ranked results. If you
filter after ranking, a bug that returns top-K=4 globally and then filters
can return zero results for a tenant with sparse docs while silently having
"used up" its budget on another tenant's chunks. Filtering first also means
a retrieval bug can never leak another brand's policy/pricing into a
customer conversation — that's a security property, not just relevance.

This is intentionally in-memory + persisted to a numpy .npz file rather than
a full vector DB, so the project runs with zero external services. Swapping
in Chroma/Pinecone means replacing this file only — `retrieve()` is the
contract the rest of the app depends on.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from app.config import settings
from app.rag.chunking import Chunk, chunk_markdown
from app.rag.embeddings import get_embedder, _tokenize


@dataclass
class RetrievedChunk:
    text: str
    source: str
    tenant_id: str
    heading: str
    score: float


class VectorStore:
    def __init__(self):
        self.embedder = get_embedder()
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def build(self, tenants_dir: str = None):
        """Ingest every tenant's policy docs. Call once at startup or via
        `python -m app.rag.build_index` after editing policy files."""
        tenants_dir = tenants_dir or settings.TENANTS_DIR
        all_chunks: list[Chunk] = []

        for tenant_id in sorted(os.listdir(tenants_dir)):
            policies_dir = os.path.join(tenants_dir, tenant_id, "policies")
            if not os.path.isdir(policies_dir):
                continue
            for fname in sorted(os.listdir(policies_dir)):
                if not fname.endswith(".md"):
                    continue
                path = os.path.join(policies_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                all_chunks.extend(chunk_markdown(text, source=fname, tenant_id=tenant_id))

        self.chunks = all_chunks
        texts = [c.text for c in all_chunks]

        if hasattr(self.embedder, "fit"):
            self.embedder.fit(texts)
        self.matrix = self.embedder.embed_documents(texts)
        return len(all_chunks)

    def save(self, path: str = None):
        path = path or settings.VECTOR_STORE_DIR
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "matrix.npy"), self.matrix)
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump([asdict(c) for c in self.chunks], f)
        # TF-IDF vocab needs persisting too so query-time vectors line up
        if hasattr(self.embedder, "vocab"):
            with open(os.path.join(path, "vocab.json"), "w") as f:
                json.dump(
                    {
                        "vocab": self.embedder.vocab,
                        "idf": self.embedder.idf.tolist(),
                    },
                    f,
                )

    def load(self, path: str = None) -> bool:
        path = path or settings.VECTOR_STORE_DIR
        matrix_path = os.path.join(path, "matrix.npy")
        chunks_path = os.path.join(path, "chunks.json")
        if not (os.path.exists(matrix_path) and os.path.exists(chunks_path)):
            return False

        self.matrix = np.load(matrix_path)
        with open(chunks_path) as f:
            self.chunks = [Chunk(**c) for c in json.load(f)]

        vocab_path = os.path.join(path, "vocab.json")
        if os.path.exists(vocab_path) and hasattr(self.embedder, "vocab"):
            with open(vocab_path) as f:
                data = json.load(f)
            self.embedder.vocab = data["vocab"]
            self.embedder.idf = np.array(data["idf"])
        return True

    def retrieve(self, tenant_id: str, query: str, top_k: int = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.RAG_TOP_K
        if self.matrix is None:
            raise RuntimeError("Vector store not built/loaded. Call build() or load() first.")

        # --- hard tenant filter FIRST ---
        tenant_indices = [i for i, c in enumerate(self.chunks) if c.tenant_id == tenant_id]
        if not tenant_indices:
            return []

        query_vec = self.embedder.embed_query(query)
        sub_matrix = self.matrix[tenant_indices]

        norms = np.linalg.norm(sub_matrix, axis=1) * (np.linalg.norm(query_vec) or 1)
        norms[norms == 0] = 1e-9
        cosine_scores = (sub_matrix @ query_vec) / norms

        # --- hybrid scoring: cosine (TF-IDF) blended with content-word coverage ---
        #
        # Two real bugs this fixes (see docs/retrieval_postmortem.md for the
        # full writeup):
        #   1. Pure TF-IDF cosine lets one rare, IDF-heavy token (e.g. a
        #      product name mentioned in an unrelated FAQ) outrank a chunk
        #      that actually shares more of the query's meaning, because
        #      rare tokens get outsized weight regardless of how many query
        #      terms actually overlap.
        #   2. With short chunks, a single shared common word can produce a
        #      deceptively high cosine score purely from vector-norm
        #      artifacts (sparse vectors normalize aggressively). If ZERO
        #      real content words overlap, that's not a match — no amount
        #      of cosine similarity should override that.
        #
        # Fix: compute plain lexical overlap on content words (stopwords
        # stripped) and blend it with cosine; force score to 0 if there is
        # no content-word overlap at all, regardless of cosine.
        query_content = set(_tokenize(query, keep_stopwords=False))
        coverage_scores = np.zeros(len(tenant_indices))
        for i, idx in enumerate(tenant_indices):
            doc_content = set(_tokenize(self.chunks[idx].text, keep_stopwords=False))
            if query_content:
                coverage_scores[i] = len(query_content & doc_content) / len(query_content)

        scores = np.where(
            coverage_scores == 0,
            0.0,
            0.6 * cosine_scores + 0.4 * coverage_scores,
        )

        ranked = np.argsort(-scores)[:top_k]
        results = []
        for r in ranked:
            idx = tenant_indices[r]
            c = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    text=c.text, source=c.source, tenant_id=c.tenant_id,
                    heading=c.heading, score=float(scores[r]),
                )
            )
        return results


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        if not _store.load():
            _store.build()
            _store.save()
    return _store
