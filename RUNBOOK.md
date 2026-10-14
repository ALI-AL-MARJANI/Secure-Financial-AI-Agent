# RUNBOOK

Exact commands to reproduce every table in the README. Times were measured on a MacBook with an Apple M4 (16 GB unified
memory, PyTorch MPS, Ollama with Q4 GGUF models) unless marked "estimate".

## 0. Setup

```bash
uv venv --python 3.11 .venv
uv sync --all-extras                 # pinned by uv.lock
source .venv/bin/activate
pytest                               # 53 fast tests, no GPU / no downloads / no Ollama
```

Disk: the Python environment is ~2 GB, FiQA + embedding caches ~0.3 GB, and each 7-8B Ollama model ~4.7-4.9 GB.
The end-to-end evaluation needs three models (~14 GB): `ollama pull mistral llama3.1:8b qwen2.5:7b`.

## 1. Data (minutes, CPU)

| Step | Command | Time | Output |
|---|---|---|---|
| eCFR corpus | `python scripts/build_corpus.py` | ~10 s | `data/corpus/passages.jsonl`, `manifest.json` |
| Golden set validation + split | `python scripts/build_golden.py` | ~2 s | `data/golden/golden.jsonl`, `stats.json` |
| Injection sets | `python scripts/build_security_sets.py` | ~30 s (downloads ~2 MB) | `data/security/injection_*.jsonl` |
| PII set | `python scripts/build_pii_set.py` | <1 s | `data/security/pii_synthetic.jsonl` |
| Router set | `python scripts/build_router_set.py` | <1 s | `data/router/router_queries.jsonl` |

## 2. Benchmarks reported in the README

| Table | Command | Time (M4) | Output |
|---|---|---|---|
| eCFR retrieval ablation | `python scripts/bench_retrieval.py --dataset ecfr --split test` (and `--split dev`) | ~3 min each | `results/retrieval/ecfr-{test,dev}.json`, `*_per_query.csv` |
| FiQA retrieval ablation | `python scripts/bench_retrieval.py --dataset fiqa --split test` | ~30 min (10 min to embed 57,638 docs on MPS; reranking rows ~5-10 min each) | `results/retrieval/fiqa-test.json` |
| Prompt-injection detectors | `python scripts/bench_injection.py` | ~6 min (ProtectAI on MPS) | `results/guardrails/injection.json`, `injection_errors.jsonl` |
| PII detectors | `python scripts/bench_pii.py` | ~1 min | `results/guardrails/pii.json` |
| Keyword router | `python scripts/bench_router.py` | <1 s | `results/agent/router.json` |
| Figures + tables | `python scripts/make_figures.py` | ~5 s | `results/figures/*.png`, `results/tables.md` |

## 3. Experiments with a local LLM (Ollama)

These runs extend the ablation with LLM components and score answers end to end. Start Ollama (`ollama serve`); every script is resumable and merges its rows into the existing result files.

| # | What | Command | Estimated time on M4 | Output |
|---|---|---|---|---|
| 3.1 | Contextual Retrieval prefixes for 2,288 chunks (Mistral 7B) | `python scripts/build_contextual_chunks.py --model mistral` | 4-6 h (measured ~8-12 s/chunk on 2 chunks incl. model load; estimate) | `data/corpus/contextual_mistral.jsonl` (commit it) |
| 3.2 | LLM retrieval rows on eCFR (contextual, HyDE x 3 seeds, CRAG) | `python scripts/bench_retrieval.py --dataset ecfr --split test --with-llm` then `--split dev` | 2-3 h per split (estimate) | merged into `results/retrieval/ecfr-*.json` |
| 3.3 | HyDE on FiQA, 3 seeds (optional; the HyDE paper evaluates on FiQA) | `python scripts/bench_retrieval.py --dataset fiqa --split test --with-llm` | 6-8 h (5,832 generations; estimate) | merged into `results/retrieval/fiqa-test.json` |
| 3.4 | LLM and cascade routers | `python scripts/bench_router.py --with-llm` | ~15 min | `results/agent/router.json` |
| 3.5 | Agent tool calling + red team, 3 seeds at T=0.7 | `python scripts/bench_agent.py --seeds 0 1 2` | 30-60 min | `results/agent/tool_calling.json` |
| 3.6 | E2E generation, per generator | `python scripts/run_e2e_eval.py --stage generate --generator mistral` (repeat for `llama3.1:8b`, `qwen2.5:7b`) | ~35 min per generator (~15 s/question measured on 3 questions) | `results/e2e/answers_<gen>.jsonl` |
| 3.7 | E2E judging with the local judge (qwen2.5:7b) | `python scripts/run_e2e_eval.py --stage judge --generator mistral` (repeat per generator) | 5-9 h per generator (2-4 min/question measured on 3 questions) | `results/e2e/judged_<gen>__qwen2.5-7b.jsonl` |
| 3.8 | Summary | `python scripts/run_e2e_eval.py --stage summary` | seconds | `results/e2e/summary.json` |
| 3.9 | Judge validation (manual, ~2 h) | `python scripts/judge_agreement.py export`, annotate per `docs/annotation_guide.md`, then `python scripts/judge_agreement.py score` | 2 h of annotation | `results/e2e/judge_agreement.json` |
| 3.10 | Refresh tables | `python scripts/make_figures.py` | seconds | `results/tables.md`, figures |

Suggested order: 3.4 and 3.5 (short), 3.6 for each generator, 3.7 overnight, then 3.1 and 3.2.

## 4. Golden-set review

`python scripts/review_golden.py` walks through every item with its gold passages (criteria in
`docs/annotation_guide.md`); `python scripts/build_golden.py` then rebuilds the split.

## 5. Demo

```bash
ollama pull mistral
python scripts/chat.py --customer CUST123          # CLI, sessions persisted in sessions/checkpoints.db
uvicorn secfin.api:app --port 8000                 # REST API
docker compose up --build                          # app + Ollama (first start pulls mistral, ~4.4 GB)
```

`curl -X POST localhost:8000/chat -H 'X-Customer-Id: CUST123' -H 'Content-Type: application/json' -d '{"message": "What is my liability if my debit card is stolen?"}'`
