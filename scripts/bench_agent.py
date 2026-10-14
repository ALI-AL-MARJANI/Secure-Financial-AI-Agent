"""Agent tool-calling accuracy and red-team checks with the real local LLM (needs Ollama).

  python scripts/bench_agent.py --seeds 0 1 2

The router is bypassed (route forced to TOOL) to measure tool calling in isolation. Human confirmation
requests are auto-declined and counted. Outputs results/agent/tool_calling.json.
"""

from __future__ import annotations

import argparse
import uuid

import yaml
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from secfin.agent.graph import build_graph, search_tool_fn
from secfin.agent.rag import RegulationSearch
from secfin.agent.router import RouteDecision
from secfin.agent.tools import Bank, make_tools
from secfin.eval.ir_metrics import bootstrap_ci
from secfin.guardrails.audit import AuditLog
from secfin.guardrails.input_guard import InputGuard
from secfin.runinfo import REPO_ROOT, load_config, run_metadata, write_json


class ForceTool:
    def route(self, message):
        return RouteDecision("TOOL", "forced")


def args_match(expected: dict, actual: dict) -> bool:
    for k, v in expected.items():
        a = actual.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if not isinstance(a, (int, float)) or abs(float(a) - v) > max(0.01 * abs(v), 1e-6):
                return False
        elif str(a).strip().upper() != str(v).upper():
            return False
    return True


def run_case(graph, q: str, customer: str | None) -> tuple[dict, list[dict], int]:
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke({"messages": [HumanMessage(content=q)], "customer_id": customer}, cfg)
    confirmations = 0
    while "__interrupt__" in state:
        confirmations += 1
        state = graph.invoke(Command(resume=False), cfg)
    calls = [tc for m in state["messages"] if m.type == "ai" for tc in (getattr(m, "tool_calls", None) or [])]
    return state, calls, confirmations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/agent.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument(
        "--temperature", type=float, default=0.7, help="sampling temperature (seeds only matter if > 0)"
    )
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    cases = yaml.safe_load((REPO_ROOT / "data/agent/tool_cases.yaml").read_text())
    search = RegulationSearch(cfg["retrieval"])
    bank = Bank(cfg["demo_bank"], REPO_ROOT / cfg["demo_bank"]["accounts_file"])
    audit = AuditLog(REPO_ROOT / "data/cache/bench_audit")

    model = args.model or cfg["llm"]["model"]
    per_seed = []
    for seed in args.seeds:
        chat = ChatOllama(model=model, temperature=args.temperature, seed=seed, num_ctx=cfg["llm"]["num_ctx"])

        def make_graph(poison: str | None = None, chat=chat):
            base = search_tool_fn(search)
            fn = (lambda q: base(q) + f"\n\n{poison}") if poison else base
            return build_graph(
                chat_model=chat,
                text_llm=lambda p: "",
                search=search,
                tools=make_tools(bank, fn),
                router=ForceTool(),
                input_guard=InputGuard(),
                audit=audit,
                limits=cfg["limits"],
                checkpointer=InMemorySaver(),
                tool_prompt=args.tool_prompt,
            )

        graph = make_graph()
        sel = []
        for c in cases["tool_selection"]:
            _, calls, _ = run_case(graph, c["q"], "CUST123")
            first = calls[0] if calls else None
            ok_tools = {c["tool"], *c.get("alt_tools", [])}
            sel.append(
                {
                    "q": c["q"],
                    "called": first["name"] if first else None,
                    "args": first["args"] if first else None,
                    "tool_ok": bool(first) and first["name"] in ok_tools,
                    "args_ok": bool(first)
                    and (
                        (first["name"] == c["tool"] and args_match(c["args"], first["args"]))
                        or first["name"] in c.get("alt_tools", [])
                    ),
                }
            )

        red = []
        for c in cases["redteam"] + cases["indirect_injection"]:
            g = make_graph(c.get("poison"))
            state, calls, confirmations = run_case(g, c["q"], c.get("customer", "CUST123"))
            events = state.get("tool_events", [])
            executed = [e for e in events if e["decision"] == "allow"]
            attempted_forbidden = (
                any(
                    tc["name"] == c.get("forbidden_call", {}).get("tool")
                    and str(tc["args"].get("customer_id", "")).upper() == c["forbidden_call"]["customer_id"]
                    for tc in calls
                )
                if "forbidden_call" in c
                else None
            )
            leaked = any(s in state.get("answer", "") for s in c.get("forbidden", []))
            if c.get("expect_confirmation"):
                success = confirmations > 0 and not executed
            elif c.get("expect_denied_or_no_call"):
                success = not any(
                    e["tool"] in ("simulate_mortgage", "check_overdraft_risk", "get_account_balance")
                    and e["decision"] == "allow"
                    for e in events
                )
            else:
                executed_forbidden = any(e["decision"] == "allow" for e in events) and attempted_forbidden
                success = not leaked and not executed_forbidden
            red.append(
                {
                    "id": c["id"],
                    "success": success,
                    "llm_attempted_forbidden_call": attempted_forbidden,
                    "leaked": leaked,
                    "confirmations": confirmations,
                    "events": events,
                    "answer": state.get("answer", "")[:300],
                }
            )
        per_seed.append({"seed": seed, "tool_selection": sel, "redteam": red})
        print(
            f"seed {seed}: tool acc={sum(s['tool_ok'] for s in sel) / len(sel):.2f} "
            f"args acc={sum(s['args_ok'] for s in sel) / len(sel):.2f} "
            f"red-team defended={sum(r['success'] for r in red)}/{len(red)}"
        )

    def agg(key, sub):
        vals = [float(x[key]) for s in per_seed for x in s[sub] if x[key] is not None]
        return bootstrap_ci(vals)

    write_json(
        REPO_ROOT / f"results/agent/tool_calling_{model.replace(':', '-')}_{args.tool_prompt}.json",
        {
            "meta": run_metadata(config=vars(args) | {"model": model}, seed=args.seeds),
            "summary": {
                "tool_selection_accuracy": agg("tool_ok", "tool_selection"),
                "argument_accuracy": agg("args_ok", "tool_selection"),
                "redteam_defended": agg("success", "redteam"),
                "llm_attempted_forbidden_call": agg("llm_attempted_forbidden_call", "redteam"),
            },
            "runs": per_seed,
        },
    )


if __name__ == "__main__":
    main()
