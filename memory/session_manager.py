"""
Session Manager
- SQLite checkpointer: LangGraph thread_id-based persistence (per-session memory)
- Mem0 integration: cross-session semantic memory (user preferences & financial profile)
"""
import uuid
import os
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------------------------------------------------
# SQLite Checkpointer — persistent LangGraph sessions
# Allows resuming any conversation via thread_id.
# Uses langgraph-checkpoint-sqlite.
# ---------------------------------------------------------------------------

_DB_PATH = os.path.join("sessions", "checkpoints.db")


def get_checkpointer() -> SqliteSaver:
    os.makedirs("sessions", exist_ok=True)
    return SqliteSaver.from_conn_string(_DB_PATH)


def new_thread_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Mem0 — cross-session semantic memory
# Extracts facts from conversations and retrieves them on next session.
# Paper: arxiv:2504.19413 (ECAI 2025) — +26% vs OpenAI Memory baseline.
# Falls back gracefully if mem0ai is not installed or Ollama model unavailable.
# ---------------------------------------------------------------------------

_mem0_client = None


def _get_mem0():
    global _mem0_client
    if _mem0_client is not None:
        return _mem0_client
    try:
        from mem0 import Memory
        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "mistral",
                    "temperature": 0,
                    "ollama_base_url": "http://localhost:11434",
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "all-MiniLM-L6-v2"}
            },
            "vector_store": {
                "provider": "faiss",
                "config": {"embedding_model_dims": 384}
            },
        }
        _mem0_client = Memory.from_config(config)
        return _mem0_client
    except Exception as e:
        print(f"[Memory] Mem0 unavailable ({e}). Running without cross-session memory.")
        return None


def save_memory(user_id: str, messages: list):
    """Extract and store facts from the conversation for future sessions."""
    mem = _get_mem0()
    if mem is None:
        return
    try:
        mem.add(messages, user_id=user_id)
    except Exception as e:
        print(f"[Memory] Failed to save memory: {e}")


def load_memory(user_id: str) -> str:
    """Retrieve relevant memories for the user as a context string."""
    mem = _get_mem0()
    if mem is None:
        return ""
    try:
        memories = mem.get_all(user_id=user_id)
        if not memories:
            return ""
        facts = [m["memory"] for m in memories.get("results", [])]
        if not facts:
            return ""
        return "User memory from previous sessions:\n" + "\n".join(f"- {f}" for f in facts)
    except Exception as e:
        print(f"[Memory] Failed to load memory: {e}")
        return ""


def search_memory(user_id: str, query: str) -> str:
    """Search memories relevant to a specific query."""
    mem = _get_mem0()
    if mem is None:
        return ""
    try:
        results = mem.search(query, user_id=user_id)
        facts = [r["memory"] for r in results.get("results", [])]
        return "\n".join(f"- {f}" for f in facts) if facts else ""
    except Exception as e:
        print(f"[Memory] Search failed: {e}")
        return ""
