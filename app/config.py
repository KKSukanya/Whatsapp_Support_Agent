"""
Central configuration. Everything is driven by environment variables so the
same code runs in mock mode (no API key, deterministic, free) or live mode
(real OpenAI/Gemini calls) without code changes.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # --- LLM provider -------------------------------------------------
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")  # "mock" | "openai"
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    CHEAP_MODEL: str = os.getenv("CHEAP_MODEL", "gpt-4o-mini")
    STRONG_MODEL: str = os.getenv("STRONG_MODEL", "gpt-4o")

    # Force mock mode automatically if a real provider was requested but no
    # key is present, instead of crashing on first request.
    @property
    def effective_provider(self) -> str:
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            return "mock"
        return self.LLM_PROVIDER

    # --- Observability --------------------------------------------------
    LANGCHAIN_TRACING_V2: bool = _bool("LANGCHAIN_TRACING_V2", False)
    TRACE_LOG_PATH: str = os.getenv("TRACE_LOG_PATH", "logs/traces.jsonl")

    # --- Guardrails -------------------------------------------------------
    MAX_TOOL_CALLS_PER_TURN: int = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "3"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "15"))

    # --- RAG ---------------------------------------------------------------
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
    TENANTS_DIR: str = os.getenv("TENANTS_DIR", "data/tenants")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
    RAG_MIN_SCORE: float = float(os.getenv("RAG_MIN_SCORE", "0.08"))

    # --- Cost (USD per 1K tokens, approximate, editable) -------------------
    PRICING = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "mock": {"input": 0.0, "output": 0.0},
    }


settings = Settings()
