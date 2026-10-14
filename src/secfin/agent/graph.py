"""LangGraph agent: every guardrail is a node, so blocked paths and tool decisions are visible in traces.

START -> input_guard -(blocked)-> audit -> END
             |
           router -> rag ----------------------------> output_guard -> audit -> END
                 -> tool_agent <-> action_guard (allow / deny / human confirm via interrupt)
                 -> refuse --------------------------> audit
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from secfin.agent.rag import RegulationSearch, format_context, generate_answer
from secfin.guardrails.action_guard import check_action
from secfin.guardrails.audit import AuditLog
from secfin.guardrails.input_guard import InputGuard
from secfin.guardrails.output_guard import check_output
from secfin.llm import TextLLM

REFUSAL_INJECTION = (
    "I can't process this request because it looks like an attempt to override my instructions."
)
REFUSAL_SCOPE = (
    "I can help with US banking rules (electronic transfers, savings disclosures, funds availability), "
    "your demo account data, and loan calculations. I can't help with that request."
)
TOOL_SYSTEM = (
    "The authenticated customer ID is {customer_id}; never use another customer ID. Use the tools for account "
    "data and calculations and never invent figures. Rates are in percent, amounts in US dollars. After tool "
    "results arrive, answer briefly with the figures they returned."
)
# Inline variants chosen on 6 probe queries outside the benchmark set (Mistral 7B: 5/6 native tool calls vs 4/6
# without "Use a tool to answer"). After tool results arrive the nudge is dropped to avoid repeated calls.
TOOL_INLINE_FIRST = "(The authenticated customer ID is {customer_id}. Use a tool to answer.)"
TOOL_INLINE_AFTER = (
    "(The authenticated customer ID is {customer_id}. Answer with the figures returned by the tools.)"
)
MAX_TOOL_ROUNDS = 4


_TEXT_TOOL_CALLS = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\])", re.DOTALL)


def _recover_text_tool_calls(reply: AIMessage, tools_by_name: dict) -> AIMessage:
    """Mistral sometimes prints its native '[TOOL_CALLS] [{"name": ..., "arguments": ...}]' format as text
    instead of the structured field. Convert it back into tool calls (unknown tools are dropped)."""
    if getattr(reply, "tool_calls", None):
        return reply
    m = _TEXT_TOOL_CALLS.search(str(reply.content))
    if not m:
        return reply
    try:
        raw = json.loads(m.group(1))
    except json.JSONDecodeError:
        return reply
    calls = [
        {
            "name": c["name"],
            "args": c.get("arguments", {}),
            "id": f"text-{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }
        for c in raw
        if isinstance(c, dict) and c.get("name") in tools_by_name and isinstance(c.get("arguments", {}), dict)
    ]
    return AIMessage(content="", tool_calls=calls, id=reply.id) if calls else reply


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str | None
    user_text: str
    input_hash: str
    threat_level: str
    input_flags: list[str]
    route: str
    route_method: str
    contexts: list[dict]
    evidence: list[str]
    tool_events: list[dict]
    tool_rounds: int
    answer: str
    output_flags: list[str]
    blocked: bool


def build_graph(
    *,
    chat_model: Any,  # LangChain chat model supporting .bind_tools (tool route)
    text_llm: TextLLM,  # plain text LLM (RAG generation)
    search: RegulationSearch | Any,
    tools: list,
    router: Any,
    input_guard: InputGuard,
    audit: AuditLog,
    limits: dict,
    checkpointer=None,
    tool_prompt: str = "inline",  # inline | system
):
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = chat_model.bind_tools(tools)

    def input_guard_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        decision = input_guard.check(str(last.content))
        update: dict = {
            # Every per-turn field is reset here so nothing from the previous turn leaks into this one.
            "input_hash": audit.digest(str(last.content)),
            "user_text": "",
            "route_method": "",
            "threat_level": decision.threat_level,
            "input_flags": decision.reasons,
            "contexts": [],
            "evidence": [],
            "tool_events": [],
            "tool_rounds": 0,
            "output_flags": [],
            "blocked": not decision.allowed,
            "route": "",
            "answer": "",
        }
        if not decision.allowed:
            update["answer"] = REFUSAL_INJECTION
            # The raw text never stays in the checkpoint: replace it with a placeholder.
            update["messages"] = [HumanMessage(content="[blocked input]", id=last.id)]
        else:
            update["user_text"] = decision.text
            if decision.text != last.content:
                update["messages"] = [HumanMessage(content=decision.text, id=last.id)]
        return update

    def router_node(state: AgentState) -> dict:
        d = router.route(state["user_text"])
        return {"route": d.label, "route_method": d.method}

    def refuse_node(state: AgentState) -> dict:
        return {"answer": REFUSAL_SCOPE, "messages": [AIMessage(content=REFUSAL_SCOPE)]}

    def rag_node(state: AgentState) -> dict:
        docs = search.search(state["user_text"])
        answer = generate_answer(text_llm, state["user_text"], docs)
        return {
            "answer": answer,
            "evidence": [d.text for d in docs],
            "contexts": [{"chunk_id": d.chunk_id, "citation": d.citation} for d in docs],
        }

    def tool_agent_node(state: AgentState) -> dict:
        cid = state.get("customer_id") or "none"
        # Only the current turn is sent: with earlier turns' tool messages in context, Mistral 7B regressed to
        # printing tool calls as text. Past turns stay in the checkpoint but are not used by the tool route.
        start = max(j for j, m in enumerate(state["messages"]) if m.type == "human")
        history = list(state["messages"][start:])
        if tool_prompt == "inline":
            # Mistral 7B via Ollama stops emitting native tool calls when a system message is present
            # (see README, "Tool-calling reliability"); a short instruction goes in front of the current user
            # turn instead. Only the model input changes, not the stored conversation.
            after_tools = history[-1].type == "tool"
            prefix = (TOOL_INLINE_AFTER if after_tools else TOOL_INLINE_FIRST).format(customer_id=cid)
            history[0] = HumanMessage(content=f"{prefix} {history[0].content}")
            reply = model_with_tools.invoke(history)
        else:
            reply = model_with_tools.invoke(
                [SystemMessage(content=TOOL_SYSTEM.format(customer_id=cid)), *history]
            )
        reply = _recover_text_tool_calls(reply, tools_by_name)
        out: dict = {"messages": [reply]}
        if not getattr(reply, "tool_calls", None):
            out["answer"] = str(reply.content)
        return out

    def action_guard_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        messages, events, evidence = [], list(state.get("tool_events", [])), list(state.get("evidence", []))
        for call in last.tool_calls:
            decision = check_action(call["name"], call["args"], state.get("customer_id"), limits)
            if decision.decision == "confirm":
                approved = interrupt(
                    {"tool": call["name"], "args": decision.args, "reasons": decision.reasons}
                )
                if not approved:
                    decision.decision, decision.reasons = (
                        "deny",
                        decision.reasons + ["declined by human reviewer"],
                    )
                else:
                    decision.decision = "allow"
            event = {"tool": call["name"], "decision": decision.decision, "reasons": decision.reasons}
            if decision.decision == "allow":
                result = tools_by_name[call["name"]].invoke(decision.args)
                content = result if isinstance(result, str) else json.dumps(result)
                evidence.append(content)
            else:
                content = f"DENIED by action guard: {'; '.join(decision.reasons)}"
            events.append(event)
            audit.write("tool_call", **event)
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {
            "messages": messages,
            "tool_events": events,
            "evidence": evidence,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
        }

    def output_guard_node(state: AgentState) -> dict:
        route = state.get("route")
        evidence = list(state.get("evidence") or []) if route in ("RAG", "TOOL") else None
        if route == "TOOL":
            evidence.append(state.get("user_text", ""))  # numbers the customer typed may be echoed back
        answer = state.get("answer") or "I could not complete this request with the available tools."
        decision = check_output(answer, evidence, on_ungrounded="withhold" if route == "TOOL" else "annotate")
        out: dict = {"answer": decision.text, "output_flags": decision.flags}
        if state.get("route") == "RAG":
            out["messages"] = [AIMessage(content=decision.text)]
        elif decision.text != state.get("answer"):
            out["messages"] = [AIMessage(content=decision.text, id=state["messages"][-1].id)]
        return out

    def audit_node(state: AgentState) -> dict:
        audit.write(
            "turn",
            input_hash=state.get("input_hash"),
            threat_level=state.get("threat_level"),
            input_flags=state.get("input_flags"),
            route=state.get("route"),
            route_method=state.get("route_method"),
            tools=state.get("tool_events"),
            output_flags=state.get("output_flags"),
            citations=[c["citation"] for c in state.get("contexts", [])],
            blocked=state.get("blocked", False),
        )
        return {}

    g = StateGraph(AgentState)
    for name, fn in [
        ("input_guard", input_guard_node),
        ("router", router_node),
        ("refuse", refuse_node),
        ("rag", rag_node),
        ("tool_agent", tool_agent_node),
        ("action_guard", action_guard_node),
        ("output_guard", output_guard_node),
        ("audit", audit_node),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", lambda s: "audit" if s.get("blocked") else "router")
    g.add_conditional_edges(
        "router", lambda s: {"RAG": "rag", "TOOL": "tool_agent"}.get(s["route"], "refuse")
    )
    g.add_edge("refuse", "audit")
    g.add_edge("rag", "output_guard")
    g.add_conditional_edges(
        "tool_agent",
        lambda s: (
            "action_guard"
            if getattr(s["messages"][-1], "tool_calls", None) and s.get("tool_rounds", 0) < MAX_TOOL_ROUNDS
            else "output_guard"
        ),
    )
    g.add_edge("action_guard", "tool_agent")
    g.add_edge("output_guard", "audit")
    g.add_edge("audit", END)
    return g.compile(checkpointer=checkpointer)


def search_tool_fn(search: RegulationSearch):
    def fn(query: str) -> str:
        docs = search.search(query)
        return format_context(docs) if docs else "No matching regulation text."

    return fn
