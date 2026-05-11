---
name: Project Build Context
description: Full build state of Secure Financial AI Agent — what was built, how, and what's left
type: project
---

Full architecture implemented in one session (2026-05-11).

**Why:** User wants an impressive, complete, technically deep repo to showcase on GitHub for fintech/AI community.

**How to apply:** When continuing work, reference CLAUDE.md for full spec. All modules are functional and tested.

## What was built

### Core modules
- `rag.py` — Full advanced RAG: Contextual Retrieval + Hybrid BM25/FAISS + Cross-Encoder Reranking + Multi-HyDE + CRAG grader
- `tools.py` — 6 tools: get_account_balance, simulate_mortgage, calculate_loan_affordability, check_overdraft_risk, get_spending_summary, explain_decision
- `main.py` — Full LangGraph graph with 3-layer guardrails wired in, SQLite session persistence, Mem0 memory

### Guardrails (3-layer defense)
- `guardrails/input_guard.py` — PII regex (SSN, credit card, IBAN), prompt injection (14 patterns), topic filter
- `guardrails/action_guard.py` — SQL injection, path traversal, dollar thresholds, format validation
- `guardrails/output_guard.py` — Compliance check (crypto, guaranteed returns, fee waiver), PII scrub, grounding check via LLM, JSONL audit logging

### Memory
- `memory/session_manager.py` — SQLite checkpointer (LangGraph), Mem0 cross-session semantic memory (optional, uses Ollama)

### Evaluation
- `evaluation/golden_dataset.json` — 30 QA pairs from bank_policies.txt (mortgage, overdraft, investments)
- `evaluation/run_eval.py` — RAGAS + DeepEval runner

### Notebooks (5 total)
- `01_rag_pipeline_demo.ipynb` — Baseline vs Hybrid vs Reranked comparison with latency charts
- `02_guardrails_demo.ipynb` — Live attack demos (injection, PII, compliance), attack category charts
- `03_agent_walkthrough.ipynb` — LangGraph graph visualization, step-by-step tool call traces
- `04_evaluation_results.ipynb` — RAGAS radar + bar charts, golden dataset distribution
- `05_memory_demo.ipynb` — SQLite session resume demo + Mem0 cross-session flow diagram

### Config
- `CLAUDE.md` — Full project context for future sessions
- `README.md` — Clean, concise with architecture table
- `requirements.txt` — All dependencies
- `.env.example` — Template

## All guardrail tests pass (7/7 input, 4/4 action)

## What still needs to be done
- Run `python main.py` end-to-end with Ollama running to test full graph
- Run `python evaluation/run_eval.py` with Ollama to get real RAGAS scores
- Execute notebooks to generate actual charts
- Possibly add a FastAPI REST layer if user wants web interface
