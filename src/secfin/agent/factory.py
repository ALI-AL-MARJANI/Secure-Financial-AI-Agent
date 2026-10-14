"""Build the agent from configs/agent.yaml (used by the CLI, the API and the agent benchmark)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from secfin.agent.graph import build_graph, search_tool_fn
from secfin.agent.rag import RegulationSearch
from secfin.agent.router import CascadeRouter, KeywordRouter, LLMRouter
from secfin.agent.tools import Bank, make_tools
from secfin.guardrails.audit import AuditLog
from secfin.guardrails.injection import CascadeDetector, ProtectAIDetector, RegexInjectionDetector
from secfin.guardrails.input_guard import InputGuard
from secfin.llm import OllamaLLM
from secfin.runinfo import REPO_ROOT, load_config


def make_input_guard(cfg: dict) -> InputGuard:
    kind = cfg["input_guard"]["detector"]
    if kind == "regex_v1":
        return InputGuard(RegexInjectionDetector("v1"), 0.5)
    if kind == "protectai":
        return InputGuard(ProtectAIDetector(), cfg["input_guard"]["threshold"])
    if kind == "cascade":
        return InputGuard(
            CascadeDetector(
                [RegexInjectionDetector("v1"), ProtectAIDetector()], [0.5, cfg["input_guard"]["threshold"]]
            ),
            0.5,
        )
    raise ValueError(kind)


def make_router(cfg: dict, llm):
    mode = cfg["router"]["mode"]
    return {
        "keyword": lambda: KeywordRouter(),
        "llm": lambda: LLMRouter(llm),
        "cascade": lambda: CascadeRouter(llm),
    }[mode]()


class _KeywordOrRAG:
    """Keyword router that falls back to RAG instead of returning None (used when no LLM routing is wanted)."""

    def __init__(self):
        self.kw = KeywordRouter()

    def route(self, message: str):
        from secfin.agent.router import RouteDecision

        return self.kw.route(message) or RouteDecision("RAG", "keyword-default")


def build_agent(
    config_path: str | Path = REPO_ROOT / "configs/agent.yaml", checkpointer=None, base_url: str | None = None
):
    from langchain_ollama import ChatOllama

    cfg = load_config(config_path)
    base_url = base_url or os.environ.get("OLLAMA_HOST")
    llm_cfg = cfg["llm"]
    text_llm = OllamaLLM(
        llm_cfg["model"], llm_cfg["temperature"], seed=0, num_ctx=llm_cfg["num_ctx"], base_url=base_url
    )
    chat_kwargs = {
        "model": llm_cfg["model"],
        "temperature": llm_cfg["temperature"],
        "num_ctx": llm_cfg["num_ctx"],
    }
    if base_url:
        chat_kwargs["base_url"] = base_url
    chat = ChatOllama(**chat_kwargs)
    search = RegulationSearch(cfg["retrieval"])
    bank = Bank(cfg["demo_bank"], REPO_ROOT / cfg["demo_bank"]["accounts_file"])
    tools = make_tools(bank, search_tool_fn(search))
    graph = build_graph(
        chat_model=chat,
        text_llm=text_llm,
        search=search,
        tools=tools,
        router=make_router(cfg, text_llm) if cfg["router"]["mode"] != "keyword" else _KeywordOrRAG(),
        input_guard=make_input_guard(cfg),
        audit=AuditLog(REPO_ROOT / cfg["audit_dir"]),
        limits=cfg["limits"],
        checkpointer=checkpointer,
        tool_prompt=cfg.get("tool_prompt", "inline"),
    )
    return graph, cfg


def sqlite_checkpointer(path: str | Path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
