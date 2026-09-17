# Architecture & Design Decisions

This doc explains the *why* behind the structure, for anyone reading the
code cold (including future-me, or an interviewer).

## Agent boundaries: why 4 nodes instead of 1 big agent

A single agent with every tool bolted on and a long system prompt telling it
"don't do X, don't do Y" is exactly the pattern that produces unpredictable
behavior at scale — the prompt becomes a wishlist, not a constraint.
Instead:

- **Router** does one job: classify intent. It's cheap, fast, and — because
  it's isolated — independently evaluable. You can measure router accuracy
  as its own number, separate from downstream answer quality.
- **policy_qa** and **order_status** are narrow, single-purpose nodes, each
  with its own system prompt and its own guardrail. If order_status starts
  hallucinating, you know exactly which prompt and which guardrail to look
  at — you're not debugging a monolith.
- **discount_request** is *not an LLM call at all*. Discount handling has a
  fixed business answer ("always escalate to a human") — every discount
  question has the same correct response regardless of phrasing, so running
  it through an LLM only adds latency, cost, and a surface area for the
  model to improvise something it shouldn't ("well, I can't give you a
  code, but here's a hint..."). This is the "reach for the smallest thing
  that works before reaching for a bigger model" principle applied
  literally, not just as a talking point.

## Why LangGraph instead of a plain chain

Tool-call loops, retries, and "did we already try this" state need to be
structural, not implicit in prompt text. A `StateGraph` makes the possible
paths through a conversation turn explicit and inspectable — `state["nodes_
visited"]` in the trace log is a literal list of which nodes ran, which is
what makes a trace "tell the story without a walkthrough" (see
`observability/tracing.py`).

## Why tenant filtering happens before similarity search, not after

See the docstring in `app/rag/vector_store.py`, but the short version:
filtering after ranking means a bug in top-K selection can silently starve
a tenant with a sparse corpus, and — much worse — creates a code path where
cross-tenant data could theoretically surface if the post-filter step were
ever skipped or buggy. Filtering first means there is no code path where
tenant B's chunks are ever even *loaded into the candidate set* for tenant
A's query. It's a security property, not a relevance tweak.

## Why the output guardrail checks against the tool result, not against "does this sound right"

"Does this sound plausible" is exactly what a hallucinating model is good
at producing. The only trustworthy signal is: does the claim in the answer
literally appear in the tool result we already know is correct? That's why
`output_guard.validate_order_answer` does string/field matching against the
tool payload rather than asking a second LLM "is this accurate" — a second
LLM call can hallucinate its verification just as easily as the first one
hallucinated the answer.

## Why cost routing is a `tier` parameter, not a model name

`get_llm(tier="cheap" | "strong")` — callers ask for a capability tier, and
the actual model behind that tier is a one-line config change
(`app/config.py :: CHEAP_MODEL` / `STRONG_MODEL`). In this project every
node currently uses the cheap tier (the tasks are all simple enough), but
the hook exists for the real decision this JD asks for: which turns
actually need the stronger, more expensive model, and can you defend that
split with cost numbers (`eval/cost_report.py`).

## What would change for a production version

- **Retrieval**: swap `TfidfEmbedder` for real embeddings + a proper vector
  DB (Chroma/Pinecone/pgvector) with an ANN index, and add a reranking step
  for the top-20 → top-4 narrowing.
- **Refresh strategy**: `build_index.py` would move from a manual CLI
  command to a scheduled job or a webhook triggered by the brand's CMS /
  Shopify metafield updates, so policy changes propagate within minutes,
  not at next deploy.
- **Guardrails**: layer an LLM-based classifier behind the regex layer for
  injection attempts that don't match known patterns — regex catches the
  known-knowns cheaply, an LLM classifier catches novel phrasing at higher
  cost, so you'd only invoke it when the regex layer is uncertain, not on
  every message.
- **State/memory**: conversation history beyond one turn isn't persisted
  here (each `/chat` call is stateless besides `session_id` for trace
  correlation). Production would add a memory layer (recent turns + summary)
  passed into each node's context.
- **Human handoff**: `escalated=True` responses currently just return a
  message — production would push to the actual support queue/CRM via a
  tool call, with the trace_id attached so a human agent sees the full
  reasoning path.
