from collections import Counter

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from secfin.agent.graph import build_graph
from secfin.agent.tools import Bank, make_tools
from secfin.api import create_app
from secfin.guardrails.audit import AuditLog
from secfin.guardrails.input_guard import InputGuard
from secfin.runinfo import REPO_ROOT, load_config, read_jsonl
from tests.test_graph import DefaultRAG, FakeSearch, ScriptedChat, tool_call

CFG = load_config(REPO_ROOT / "configs/agent.yaml")


def client(tmp_path, replies=()):
    bank = Bank(CFG["demo_bank"], REPO_ROOT / CFG["demo_bank"]["accounts_file"])
    graph = build_graph(
        chat_model=ScriptedChat(replies),
        text_llm=lambda p: "Up to $50 [1].",
        search=FakeSearch(),
        tools=make_tools(bank, lambda q: ""),
        router=DefaultRAG(),
        input_guard=InputGuard(),
        audit=AuditLog(tmp_path, salt="t"),
        limits=CFG["limits"],
        checkpointer=InMemorySaver(),
    )
    return TestClient(create_app(graph))


def test_chat_endpoint_rag(tmp_path):
    with client(tmp_path) as c:
        r = c.post("/chat", json={"message": "What is my liability for a stolen card?"}).json()
    assert r["route"] == "RAG" and r["citations"] and r["thread_id"]


def test_confirm_flow(tmp_path):
    big = {"principal": 20_000_000, "annual_rate": 5, "years": 25}
    with client(tmp_path, [tool_call("simulate_mortgage", big), AIMessage(content="ok")]) as c:
        r = c.post(
            "/chat", json={"message": "simulate a 20 million mortgage"}, headers={"X-Customer-Id": "CUST123"}
        ).json()
        assert r["pending_confirmation"]["tool"] == "simulate_mortgage"
        done = c.post("/confirm", json={"thread_id": r["thread_id"], "approve": True}).json()
        assert done["answer"] == "ok"
        assert c.post("/confirm", json={"thread_id": r["thread_id"], "approve": True}).status_code == 409


# ---------------------------------------------------------------- committed data integrity
def test_golden_set_integrity():
    passages = {p["passage_id"] for p in read_jsonl(REPO_ROOT / "data/corpus/passages.jsonl")}
    golden = read_jsonl(REPO_ROOT / "data/golden/golden.jsonl")
    assert len(golden) >= 150
    types = Counter(g["type"] for g in golden)
    assert types["unanswerable"] >= 20 and types["multi"] >= 20
    for g in golden:
        assert all(p in passages for p in g["gold_passages"]), g["id"]
        assert g["split"] in ("dev", "test")


def test_security_sets_have_both_labels():
    test = read_jsonl(REPO_ROOT / "data/security/injection_test.jsonl")
    labels = Counter(r["label"] for r in test)
    assert labels[1] >= 300 and labels[0] >= 3000
