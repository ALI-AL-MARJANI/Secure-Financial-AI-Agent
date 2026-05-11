"""
Secure Financial AI Agent — Main Entry Point
Full pipeline: Input Guardrail → Supervisor/Router → Agent → Action Guardrail → Output Guardrail
"""
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from tools import financial_tools
from rag import search_bank_policies
from guardrails.input_guard import check_input
from guardrails.action_guard import check_action
from guardrails.output_guard import check_output
from agents.supervisor import route, SYSTEM_PROMPTS, RouteDecision
from memory.session_manager import get_checkpointer, new_thread_id, save_memory, load_memory


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

all_tools = financial_tools + [search_bank_policies]

# ---------------------------------------------------------------------------
# LLM (shared across sub-agents)
# ---------------------------------------------------------------------------

llm = ChatOllama(model="mistral", temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

# ---------------------------------------------------------------------------
# Session-scoped context (not in graph state — used by guardrail nodes)
# ---------------------------------------------------------------------------

_ctx: dict = {
    "input_hash": "",
    "threat_level": "LOW",
    "tool_calls_made": [],
    "route_decision": "RAG_AGENT",
    "system_prompt": "",
}

# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def agent_node(state: MessagesState) -> dict:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})

            # Layer 2 — Action Guardrail
            action_result = check_action(tool_name, tool_args)
            if not action_result.allowed:
                return {"messages": [AIMessage(content=f"I cannot execute that action: {action_result.reason}")]}

            _ctx["tool_calls_made"].append(tool_name)

    return {"messages": [response]}


_base_tool_node = ToolNode(all_tools)


def tool_node(state: MessagesState) -> dict:
    return _base_tool_node.invoke(state)


# ---------------------------------------------------------------------------
# Build LangGraph graph
# ---------------------------------------------------------------------------

def _build_graph(checkpointer=None):
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Main CLI loop
# ---------------------------------------------------------------------------

def run(user_id: str = "default_user"):
    checkpointer = get_checkpointer()
    app = _build_graph(checkpointer=checkpointer)
    thread_id = new_thread_id()

    print(f"\n{'='*62}")
    print("  SecureBank AI Financial Assistant")
    print(f"  Session : {thread_id[:8]}...")
    print(f"  Model   : Mistral (Ollama — 100% local)")
    print(f"{'='*62}")
    print("  Commands: 'new' = new session | 'quit' = exit\n")

    user_memory = load_memory(user_id)
    config = {"configurable": {"thread_id": thread_id}}
    history_for_memory: list = []

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("q", "quit", "exit"):
                save_memory(user_id, history_for_memory)
                print("\nSecure session closed. Goodbye.")
                break

            if user_input.lower() == "new":
                save_memory(user_id, history_for_memory)
                thread_id = new_thread_id()
                config = {"configurable": {"thread_id": thread_id}}
                history_for_memory = []
                print(f"\n[New session: {thread_id[:8]}...]\n")
                continue

            # ── Layer 1: Input Guardrail ──────────────────────────────────
            guard = check_input(user_input)
            _ctx["input_hash"] = guard.input_hash
            _ctx["threat_level"] = guard.threat_level
            _ctx["tool_calls_made"] = []

            if not guard.allowed:
                print(f"\nAssistant [BLOCKED — {guard.threat_level}]: {guard.reason}\n")
                continue

            # ── Supervisor: intent routing ────────────────────────────────
            router = route(user_input)
            _ctx["route_decision"] = router.decision

            # Pick system prompt for this route
            system_content = SYSTEM_PROMPTS[router.decision]
            if user_memory:
                system_content += f"\n\nUser memory from prior sessions:\n{user_memory}"

            # Out-of-scope handled directly (no LLM tool loop needed)
            if router.decision == "OUT_OF_SCOPE":
                rejection = (
                    "I'm SecureBank's AI Financial Assistant and can only help with banking services: "
                    "account balances, mortgage calculations, loan affordability, policy questions, "
                    "fees, and SecureBank product information. "
                    "How can I help you with your banking needs today?"
                )
                print(f"\nAssistant: {rejection}\n")
                history_for_memory.append({"role": "user", "content": user_input})
                history_for_memory.append({"role": "assistant", "content": rejection})
                continue

            # ── Invoke LangGraph (agent + tools loop) ────────────────────
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=user_input),
            ]

            result = app.invoke({"messages": messages}, config=config)
            raw_response = result["messages"][-1].content

            # ── Layer 3: Output Guardrail ─────────────────────────────────
            output_result = check_output(
                response=raw_response,
                input_hash=guard.input_hash,
                threat_level=guard.threat_level,
                tool_calls_made=_ctx["tool_calls_made"],
            )

            final_response = output_result.final_response

            # Show routing info in dev mode
            route_tag = f"[{router.decision} via {router.method}]"
            print(f"\nAssistant {route_tag}: {final_response}\n")

            if output_result.pii_scrubbed:
                print("[Note: PII was redacted from the response.]\n")

            history_for_memory.append({"role": "user", "content": user_input})
            history_for_memory.append({"role": "assistant", "content": final_response})

        except KeyboardInterrupt:
            save_memory(user_id, history_for_memory)
            print("\n\nSession interrupted. Memory saved.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")


if __name__ == "__main__":
    run()
