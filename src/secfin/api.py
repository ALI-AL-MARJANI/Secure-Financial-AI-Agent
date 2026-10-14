"""FastAPI service. Demo authentication: the customer is taken from the X-Customer-Id header, which a real
deployment would replace with a verified identity from an auth gateway.

  uvicorn secfin.api:app --port 8000
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from secfin.runinfo import REPO_ROOT, load_config


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None


class ConfirmRequest(BaseModel):
    thread_id: str
    approve: bool


class ChatResponse(BaseModel):
    thread_id: str
    answer: str | None
    route: str | None = None
    citations: list[str] = []
    flags: list[str] = []
    pending_confirmation: dict | None = None


def _response(thread: str, state: dict) -> ChatResponse:
    if "__interrupt__" in state:
        return ChatResponse(
            thread_id=thread, answer=None, pending_confirmation=state["__interrupt__"][0].value
        )
    return ChatResponse(
        thread_id=thread,
        answer=state.get("answer"),
        route=state.get("route"),
        citations=list(dict.fromkeys(c["citation"] for c in state.get("contexts", []))),
        flags=(state.get("input_flags") or []) + (state.get("output_flags") or []),
    )


def create_app(graph=None) -> FastAPI:
    holder: dict = {"graph": graph}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if holder["graph"] is None:
            from secfin.agent.factory import build_agent, sqlite_checkpointer

            cfg = load_config(REPO_ROOT / "configs/agent.yaml")
            holder["graph"], _ = build_agent(checkpointer=sqlite_checkpointer(REPO_ROOT / cfg["sessions_db"]))
        yield

    app = FastAPI(title="secfin", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, x_customer_id: str | None = Header(default=None)) -> ChatResponse:
        thread = req.thread_id or str(uuid.uuid4())
        state = holder["graph"].invoke(
            {"messages": [HumanMessage(content=req.message)], "customer_id": x_customer_id},
            {"configurable": {"thread_id": thread}},
        )
        return _response(thread, state)

    @app.post("/confirm", response_model=ChatResponse)
    def confirm(req: ConfirmRequest) -> ChatResponse:
        graph = holder["graph"]
        cfg = {"configurable": {"thread_id": req.thread_id}}
        if not graph.get_state(cfg).interrupts:
            raise HTTPException(409, "no pending confirmation for this thread")
        return _response(req.thread_id, graph.invoke(Command(resume=req.approve), cfg))

    return app


app = create_app()
