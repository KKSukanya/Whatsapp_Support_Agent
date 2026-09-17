"""
Two embedding backends behind one interface:

- TfidfEmbedder: pure numpy, no network, no API key. This is what makes the
  RAG pipeline runnable and gradeable offline. It's a legitimate technique
  (fine for small, keyword-dense corpora like policy docs), just not what
  you'd ship at scale.
- OpenAIEmbedder: real `text-embedding-3-small` vectors for production /
  semantic matching beyond exact keyword overlap.

Swap via LLM_PROVIDER in .env — same call site either way.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from app.config import settings


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "should", "i", "you", "your", "me",
    "my", "we", "our", "it", "its", "this", "that", "to", "of", "for",
    "in", "on", "at", "and", "or", "if", "be", "have", "has", "with",
    "what", "how", "am",
}


def _tokenize(text: str, keep_stopwords: bool = True) -> list[str]:
    """
    keep_stopwords=True is used for the TF-IDF vector itself (IDF already
    down-weights common words numerically). keep_stopwords=False is used
    for the plain lexical-overlap check in vector_store.retrieve(), where
    we specifically want to know whether CONTENT words overlap — a shared
    "do" or "the" shouldn't count as evidence of relevance.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS]


class TfidfEmbedder:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.fitted_docs: list[list[str]] = []

    def fit(self, texts: list[str]):
        docs_tokens = [_tokenize(t) for t in texts]
        self.fitted_docs = docs_tokens
        vocab_set = sorted({tok for doc in docs_tokens for tok in doc})
        self.vocab = {tok: i for i, tok in enumerate(vocab_set)}

        n_docs = len(docs_tokens)
        df = np.zeros(len(self.vocab))
        for doc in docs_tokens:
            for tok in set(doc):
                df[self.vocab[tok]] += 1
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1

    def _vectorize(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        counts = Counter(tokens)
        for tok, cnt in counts.items():
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] = cnt * self.idf[idx]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vectorize(_tokenize(t)) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vectorize(_tokenize(text))


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def get_embedder():
    if settings.effective_provider == "openai":
        return OpenAIEmbedder()
    return TfidfEmbedder()
