# WhatsApp Order-Support Agent (Multi-Tenant D2C Portfolio Project)

A production-shaped LLM support agent for a WhatsApp-first D2C SaaS platform —
built to demonstrate the exact ownership areas of an AI Engineer role: agent
architecture, RAG, guardrails, observability/evals, and cost consciousness.

It simulates two D2C brands (**GlowCart**, a skincare store, and
**UrbanStride**, a footwear store) sharing one multi-tenant agent stack,
answering real customer questions about orders and policies over a
WhatsApp-style chat UI.

**Runs with zero API keys** in mock mode (deterministic, free, fully
functional end-to-end) or with real OpenAI models by setting one env var.

---

## 1. What this project demonstrates

| JD requirement | Where it lives |
|---|---|
| Agent architecture, tool calling, multi-step reasoning | `app/agents/graph.py` — LangGraph `StateGraph` with router + 4 task nodes |
| Deciding agent boundaries (LLM call vs. deterministic rule) | `discount_request` node is a fixed rule, never an LLM call — see comments in `graph.py` |
| RAG pipeline: chunking, embedding, retrieval, refresh | `app/rag/` — `chunking.py`, `embeddings.py`, `vector_store.py`, `build_index.py` |
| Multi-tenant data isolation as correctness/security | Hard tenant filter *before* similarity search — see `vector_store.retrieve()`, tested in `tests/test_rag.py` |
| Debugging retrieval failures (chunking vs. query vs. index vs. prompt) | `docs/retrieval_postmortem.md` — two real bugs found and fixed while building this |
| Input/output guardrails, prompt injection defense | `app/guardrails/input_guard.py`, `app/guardrails/output_guard.py` |
| Preventing hallucinated commitments / wrong order info | Output grounding check: agent can never state an order status not present in the tool result — `output_guard.validate_order_answer` |
| Deterministic fallbacks when model is wrong/slow/unavailable | `app/agents/runner.py` — hard timeout + fallback message, wraps every graph execution |
| Tracing, evals, regression suites | `app/observability/tracing.py`, `eval/run_eval.py`, `eval/eval_set.json` |
| Cost consciousness, cost per conversation/resolution, model routing | `eval/cost_report.py`, `get_llm(tier=...)` routing hook in `app/llm/client.py` |

---

## 2. Quick start (mock mode — no API key needed)

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the env file (already defaults to mock mode)
cp .env.example .env

# 4. Build the RAG index from the policy docs
python -m app.rag.build_index

# 5. Run the server
uvicorn app.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000** — you'll get a WhatsApp-style chat UI with a
tenant switcher and one-click sample messages (order lookup, policy
question, discount request, prompt-injection attempt).

In mock mode there's no LLM call cost and no external network dependency —
the "model" is a small deterministic rule-based stand-in (`app/llm/client.py
:: MockLLM`) wired into the *same* LangGraph agent, guardrails, RAG
pipeline, and tracing that the real model would run through. This is what
makes the whole system runnable and gradeable by someone with zero setup.

---

## 3. Running with a real OpenAI model

```bash
# in .env:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
CHEAP_MODEL=gpt-4o-mini
STRONG_MODEL=gpt-4o
```

Rebuild the index so it uses real OpenAI embeddings instead of the offline
TF-IDF fallback:

```bash
python -m app.rag.build_index
```

Restart the server. Every other part of the system (guardrails, tracing,
cost tracking, eval suite) works identically — only `app/llm/client.py` and
`app/rag/embeddings.py` change behavior based on this flag.

### Optional: LangSmith tracing
```bash
# in .env:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=wa-support-agent
```
LangGraph will trace to LangSmith automatically in parallel to the local
JSONL trace log — no code changes needed.

---

## 4. Running the eval suite

```bash
python -m eval.run_eval
```

Runs 13 cases covering intent routing, RAG-grounded Q&A, order lookup,
multi-tenant isolation, discount-request handling, and three adversarial
guardrail cases (prompt injection, PII fishing, discount manipulation).
Saves results to `eval/last_run.json`.

To compare two runs (e.g. before/after a prompt change):
```bash
cp eval/last_run.json eval/before.json
# ... make your prompt change ...
python -m eval.run_eval
python -m eval.run_eval --compare eval/before.json eval/last_run.json
```

## 5. Running the unit tests

```bash
python -m pytest tests/ -v
```
12 tests: guardrail pattern coverage (injection, discount manipulation, PII)
+ output-grounding checks + the multi-tenant isolation regression test
(`tests/test_rag.py::test_tenant_isolation_no_cross_contamination` — the
single most important test in the repo).

## 6. Cost report

Send a few chat messages (or run the eval suite) to generate trace data,
then:

```bash
python -m eval.cost_report
```

Prints cost per conversation, cost per *resolved* conversation, cost by
tenant, and a routing comparison ("what would this have cost if every call
used the strong model instead of the cheap one"). In mock mode all costs
are $0 — set `LLM_PROVIDER=openai` to see real numbers.

---

## 7. Project structure

```
app/
  agents/
    graph.py        # LangGraph StateGraph — the architectural core
    runner.py        # entrypoint: tracing + timeout + fallback wrapper
  rag/
    chunking.py       # heading-aware markdown chunking
    embeddings.py     # TF-IDF (offline) or OpenAI embeddings
    vector_store.py    # per-tenant hard-filtered retrieval, hybrid scoring
    build_index.py     # CLI: rebuild index after policy docs change
  guardrails/
    input_guard.py    # prompt injection / PII / discount-manipulation detection
    output_guard.py    # grounding validation against tool results
  llm/
    client.py         # MockLLM / OpenAILLM behind one interface, cost accounting
  observability/
    tracing.py         # structured JSONL trace logging
  tools/
    order_lookup.py    # mock Shopify order API, tenant-scoped
  main.py             # FastAPI app
  config.py           # all env-driven settings

data/tenants/<brand>/
  orders.json          # mock order DB
  policies/*.md         # RAG knowledge base

eval/
  eval_set.json        # 13 regression test cases
  run_eval.py          # eval runner + before/after comparison
  cost_report.py        # cost aggregation from trace logs

tests/                 # pytest unit tests (guardrails, RAG isolation)
frontend/index.html    # WhatsApp-style demo chat UI
docs/
  architecture.md      # design decisions and tradeoffs
  retrieval_postmortem.md   # two real retrieval bugs found + fixed
```

## 8. Known limitations (documented on purpose)

- **TF-IDF retrieval in mock mode** is a lexical/keyword method, not true
  semantic search — fine for small, keyword-dense policy docs, not what
  you'd ship at real scale. Swap to OpenAI embeddings (`LLM_PROVIDER=openai`)
  or a proper vector DB (Chroma/Pinecone/pgvector) for production; the
  `retrieve()` interface in `vector_store.py` is the only contract other
  code depends on, so this is a contained swap.
- **Output grounding check is a blunt heuristic** (does the answer mention
  a status/order-id not present in the tool result), not a semantic
  fact-check. An LLM-as-judge grounding pass is the natural next layer.
- **Token counts in mock mode are estimated** (chars/4), not exact
  tokenizer counts — swap in `tiktoken` for exact OpenAI accounting in
  production.
- **In-memory vector store**, persisted to a local `.npy`/`.json` file
  rather than a real vector DB — intentional, so the project has zero
  external service dependencies to run.

See `docs/architecture.md` for the full reasoning behind these tradeoffs.
