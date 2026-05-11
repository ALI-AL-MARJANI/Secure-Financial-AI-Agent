# Secure Financial AI Agent

A production-grade, fully local AI banking assistant implementing state-of-the-art techniques from recent research. No external APIs — everything runs via Ollama.

---

## Architecture

```
User Input
    ↓
[Input Guardrail]  ← PII detection · prompt injection · topic filter
    ↓
[Supervisor / Router]  ← intent classification → routes to sub-agent
    ├── [RAG Agent]       policy & compliance questions
    ├── [Tool Agent]      account ops & financial calculations
    └── [Chitchat Guard]  out-of-scope rejection
    ↓
[Action Guardrail]  ← tool argument validation · threshold checks
    ↓
[Tools]  get_balance · simulate_mortgage · loan_affordability · overdraft_risk
    ↓
[Output Guardrail]  ← hallucination check · PII scrub · compliance · audit log
    ↓
Response + Audit Trail
```

---

## RAG Pipeline

| Technique | Paper / Source | Gain |
|-----------|---------------|------|
| Contextual Retrieval | Anthropic (2024) | −49% retrieval failures |
| Hybrid BM25 + FAISS | LangChain EnsembleRetriever | +15–30% recall |
| Cross-Encoder Reranking | `sentence-transformers` | top-50 → top-5 precision |
| CRAG Grader | arxiv:2401.15884 (ICLR 2025) | adaptive fallback on poor retrieval |
| Multi-HyDE | arxiv:2509.16369 (ACL FinNLP 2025) | +11.2% accuracy on financial QA |

---

## Guardrails (3-Layer Defense)

- **Layer 1 — Input**: regex PII scan · prompt injection heuristics · topic relevance
- **Layer 2 — Action**: tool argument sanitization · dollar threshold flags
- **Layer 3 — Output**: context grounding · compliance vs bank policies · audit logging

---

## Stack

- **LLM**: Mistral via Ollama (local, temperature=0)
- **Agent Framework**: LangGraph
- **RAG**: FAISS + BM25 hybrid + cross-encoder reranking
- **Memory**: LangGraph SQLite checkpointer · Mem0 cross-session memory
- **Evaluation**: RAGAS · DeepEval

---

## Project Structure

```
├── main.py                  # LangGraph graph entry point
├── tools.py                 # Financial tools
├── rag.py                   # Advanced RAG pipeline
├── guardrails/
│   ├── input_guard.py       # Layer 1
│   ├── action_guard.py      # Layer 2
│   └── output_guard.py      # Layer 3 + audit logging
├── agents/
│   └── supervisor.py        # Router / Planner
├── memory/
│   └── session_manager.py   # Session persistence + Mem0
├── evaluation/
│   ├── golden_dataset.json  # 30 QA test pairs
│   └── run_eval.py          # RAGAS + DeepEval runner
├── notebooks/               # Visual demos (see below)
└── data/bank_policies.txt
```

---

## Notebooks

| Notebook | What it shows |
|----------|--------------|
| `01_rag_pipeline_demo.ipynb` | Basic vs hybrid vs contextual retrieval — side-by-side metrics |
| `02_guardrails_demo.ipynb` | Live guardrail catches: PII leaks, jailbreaks, prompt injection |
| `03_agent_walkthrough.ipynb` | LangGraph graph execution step-by-step with node visualization |
| `04_evaluation_results.ipynb` | RAGAS scores, faithfulness charts, context precision over iterations |
| `05_memory_demo.ipynb` | Mem0 cross-session memory: user facts persisting across conversations |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill env vars
cp .env.example .env

# Make sure Ollama is running with Mistral
ollama pull mistral

# Run the agent
python main.py
```

---

## References

- [CRAG — Corrective RAG (ICLR 2025)](https://arxiv.org/abs/2401.15884)
- [Contextual Retrieval — Anthropic (2024)](https://www.anthropic.com/news/contextual-retrieval)
- [agentic-guardrails — FareedKhan-dev](https://github.com/FareedKhan-dev/agentic-guardrails)
