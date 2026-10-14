"""Interactive CLI.

python scripts/chat.py --customer CUST123                 # new session
python scripts/chat.py --customer CUST123 --thread <id>   # resume a session (SQLite checkpoints)
"""

from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from secfin.agent.factory import build_agent, sqlite_checkpointer
from secfin.runinfo import REPO_ROOT, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--customer", default=None, help="authenticated demo customer (CUST123 / CUST456 / CUST789)"
    )
    ap.add_argument("--thread", default=None)
    ap.add_argument("--config", default="configs/agent.yaml")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    graph, _ = build_agent(
        REPO_ROOT / args.config, checkpointer=sqlite_checkpointer(REPO_ROOT / cfg["sessions_db"])
    )
    thread = args.thread or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread}}
    print(f"session {thread}  customer={args.customer}  (Ctrl-D to quit)")
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            break
        if not text:
            continue
        state = graph.invoke({"messages": [HumanMessage(content=text)], "customer_id": args.customer}, config)
        while "__interrupt__" in state:
            req = state["__interrupt__"][0].value
            ans = input(
                f"[confirm] {req['tool']} {req['args']} ({'; '.join(req['reasons'])}) - approve? [y/N] "
            )
            state = graph.invoke(Command(resume=ans.strip().lower() == "y"), config)
        print(f"\nassistant [{state.get('route')}/{state.get('route_method', '-')}]> {state['answer']}")
        for citation in dict.fromkeys(c["citation"] for c in state.get("contexts", [])):
            print(f"   - {citation}")
        if state.get("output_flags"):
            print(f"   flags: {state['output_flags']}")


if __name__ == "__main__":
    main()
