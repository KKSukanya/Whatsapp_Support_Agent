from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agents.runner import run_agent
from app.config import settings

app = FastAPI(
    title="WhatsApp Order-Support Agent",
    description=(
        "Multi-tenant D2C support agent (portfolio project modeled on a "
        "WhatsApp-first CX platform). LangGraph orchestration, per-tenant "
        "RAG, input/output guardrails, cost + trace observability."
    ),
    version="1.0.0",
)


class ChatRequest(BaseModel):
    tenant_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    escalated: bool
    escalation_reason: str | None = None
    intent: str | None = None
    trace_id: str


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.effective_provider}


@app.get("/tenants")
def list_tenants():
    tenants = sorted(
        d for d in os.listdir(settings.TENANTS_DIR)
        if os.path.isdir(os.path.join(settings.TENANTS_DIR, d))
    )
    return {"tenants": tenants}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    tenants_dir = os.path.join(settings.TENANTS_DIR, req.tenant_id)
    if not os.path.isdir(tenants_dir):
        raise HTTPException(status_code=404, detail=f"Unknown tenant_id '{req.tenant_id}'")

    result = run_agent(tenant_id=req.tenant_id, session_id=req.session_id, message=req.message)
    return ChatResponse(**result)


@app.get("/traces/recent")
def recent_traces(limit: int = 20):
    """Small convenience endpoint so you can eyeball recent traces from the
    browser instead of tailing the JSONL file — handy in a live demo."""
    path = settings.TRACE_LOG_PATH
    if not os.path.exists(path):
        return {"traces": []}
    with open(path) as f:
        lines = f.readlines()[-limit:]
    return {"traces": [json.loads(line) for line in lines]}


# Serve the simple chat UI
if os.path.isdir("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

    @app.get("/")
    def index():
        return FileResponse("frontend/index.html")
