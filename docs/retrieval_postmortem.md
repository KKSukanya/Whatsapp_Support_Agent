# Retrieval Postmortem: Two Real Bugs Found While Building This

This is exactly the kind of writeup the JD asks an interviewer to expect:
*"debug retrieval failures properly — is it the chunking, the query, the
index, or the prompt? Know how to tell the difference."* These two bugs
were found by actually running the eval suite (`eval/eval_set.json`)
against the live pipeline, not hypothesized in advance — the suite failed
at 10/13 on first run, and here's the root-cause process.

---

## Bug 1: A rare token dominates ranking over more relevant common tokens

**Symptom:** Query `"Can I return an opened serum?"` on the GlowCart tenant
retrieved a Vitamin-C/Retinol layering FAQ chunk as the top result, instead
of the returns policy chunk that actually answers the question (which
explicitly covers "opened skincare products... hygiene reasons... unless
damaged").

**Diagnosis process:**
1. First check: is this a chunking problem? Inspected the chunks directly
   (`chunk_markdown` output) — the returns policy chunk was intact and
   contained the right sentence. Not chunking.
2. Second check: is this an index/embedding problem? Printed the raw
   cosine similarity scores for both candidate chunks. The FAQ chunk scored
   0.207, the returns chunk scored 0.124 — so the *ranking* was the issue,
   not corruption or missing data. Index was fine, embeddings were fine.
3. Root cause: TF-IDF weights rare tokens heavily by design (that's the
   "IDF" — inverse *document* frequency). The word "serum" appears in only
   one or two chunks across the whole corpus, so it gets a very high
   weight — high enough to outrank a chunk that actually shares more of
   the query's *meaning* ("return", "opened") but whose terms are more
   common across the corpus and therefore individually lower-weighted.
   This is a known, textbook failure mode of pure TF-IDF: **it's a query
   problem, mechanically, but caused by the embedding/ranking method,
   not the query itself.**

**Fix applied:** Added a lexical content-word coverage score (query
content-words ∩ document content-words, stopwords stripped) and blended it
with the TF-IDF cosine score (`0.6 * cosine + 0.4 * coverage` in
`vector_store.py :: retrieve`). This rewards chunks that share more of the
query's actual content words, which counteracts a single rare token
dominating the ranking. After the fix, the returns chunk ranked first.

**Real production fix (documented, not built here, for scope reasons):**
BM25 (which saturates term-frequency contribution and handles this class
of problem more principled than raw TF-IDF) or, better, real semantic
embeddings where "return an opened item" and "hygiene... opened... cannot
be returned" are close in vector space regardless of exact word overlap.

---

## Bug 2: A short-vector false positive from stopword overlap

**Symptom:** Query `"Do you sell winter jackets?"` on the UrbanStride
tenant (a footwear/bag brand with no jackets in the catalog) matched a
shoe-fit FAQ chunk with a cosine score of 0.169 — comfortably above the
0.08 confidence threshold — and the agent answered instead of escalating.

**Diagnosis process:**
1. Printed the actual matched chunk text: `"How do UrbanStride shoes fit -
   true to size? Most customers find our Trail Runner line fits true to
   size..."` — genuinely nothing about jackets.
2. Checked token overlap manually. The only shared token between the query
   and the chunk was **"do"** — a stopword.
3. Root cause: with short document vectors (a few dozen unique terms),
   cosine similarity is highly sensitive to normalization — a single
   shared token, even a near-meaningless one, can produce a non-trivial
   cosine score purely because the vectors are sparse and short. This is a
   **ranking/scoring problem specific to short documents**, not a chunking
   or index problem — the chunk itself is fine, it's the scoring math being
   fooled by vector-length artifacts.

**Fix applied:** Two changes: (1) added a stopword list so common words
like "do", "the", "is" don't count toward content-word coverage; (2) in the
hybrid score, if content-word coverage is exactly zero, the score is forced
to zero regardless of cosine — the reasoning being that if literally no
real content word overlaps, no amount of cosine similarity noise should be
treated as a match. After the fix, this query correctly falls below the
confidence threshold and the agent escalates instead of guessing.

---

## What this changes about how I'd approach retrieval going forward

- **Always check raw scores, not just "did it retrieve something."** Both
  bugs would have looked identical from the outside (agent gave an answer,
  no error) — the only way to catch them was inspecting the ranked
  candidates and their scores directly.
- **An eval suite with intent + escalation-correctness checks catches this
  class of bug automatically.** Bug 2 specifically is exactly what the
  `out_of_scope_no_context` eval case exists to catch — without it, this
  ships silently.
- **Pure lexical methods (TF-IDF/BM25) need an explicit "no real overlap =
  no match" guardrail**, because their failure mode isn't "returns nothing"
  (easy to catch) — it's "returns something plausible-looking that's
  actually wrong" (easy to miss without deliberately checking).
